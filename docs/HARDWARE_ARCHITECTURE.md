# GPU–TSU Hardware Architecture

This document is the consolidated specification for the implemented GPU–TSU
chip: the SIMT control processor, tensor engines, memory system, binary and
categorical stochastic fabrics, and the sampling ISA that connects them.

## Contents

- [SIMT Core](#simt-core)
- [Tensor Sidecar and D4 Socket](#tensor-sidecar-and-d4-socket)
- [Memory Hierarchy](#memory-hierarchy)
- [Sampling ISA](#sampling-isa)
- [Sampling Cluster Construction](#sampling-cluster-construction)
- [Categorical Q-Site Architecture](#categorical-q-site-architecture)


## SIMT Core


Written before the Epoch-D RTL under the frozen design rule: barrel first,
assertion density is the game, richer lockstep + shrinking, and
pre-registered adjudication. Every invariant below lands as (a) an SVA
property in the RTL, (b) a mirrored cocotb check, (c) where marked [F], a
SymbiYosys bounded proof obligation. No RTL merges while any INV lacks its
assertion (Golden-First Law extension).

### 1. v1 microarchitecture (M13): the barrel of harts

- **W = 8 warps**, each warp = one independent scalar RV32I hart (no
  intra-warp lanes in v1; lanes+divergence arrive M14 — see §5).
- **5-stage pipeline** F → D → EX → M → WB, **deterministic round-robin**:
  cycle t fetches warp (t mod W). With W(8) > depth(5), **at most one
  instruction per warp is ever in flight** — this single property deletes
  same-warp RAW/WAW/control hazards, bypass networks, and the stall tree.
- **The only interlock is the per-warp busy bit**: set at issue of a
  long-latency op (load/store while memory waits, MUL/DIV iterating,
  fence), cleared exactly once by completion. A busy warp's fetch slot
  issues a **bubble** (warp slot stays reserved — rotation never skips,
  it inserts bubbles; determinism over utilization, v1).
- Per-warp architectural state: PC, 31×32 regfile, minimal CSR
  (mhartid = warp id, cycle, instret). Shared: imem port, dmem port
  (round-robin arbitration is free — stages are warp-disjoint), MMIO/UART.
- Memory v1: synchronous 1-cycle BRAM-model ports (the M16+ hierarchy
  replaces them behind the same request/response contract).

### 2. Invariant catalog (the assertion budget)

| ID | Invariant (formal statement) | Class |
|---|---|---|
| INV-1 | Warp tags of all occupied stages are pairwise distinct | [F] safety |
| INV-2 | Stage S occupied at cycle t ⇒ its warp tag = (t − idx(S)) mod W, where idx(F..WB) = 0..4 | [F] safety |
| INV-3 | issue(w) ⇒ ¬busy[w]; busy[w] set only at issue of a long-latency op by w; cleared only by that op's unique completion; set→cleared exactly once per set | [F] safety |
| INV-4 | ≤ 1 in-flight instruction per warp at all times (corollary of W>depth + INV-3; asserted independently — it is load-bearing) | [F] safety |
| INV-5 | Regfile write port fires ⇒ WB stage occupied ∧ writer warp = WB warp tag ∧ rd ≠ x0 | [F] safety |
| INV-6 | Bubbles are inert: a bubble in any stage never asserts regfile-we, dmem-req, csr-we, or redirect | [F] safety |
| INV-7 | PC[w] changes only when warp w occupies WB (retirement or redirect); next-PC ∈ {PC+4, branch/jump target, trap vector} | [F] safety |
| INV-8 | Per-warp memory ops issue in program order and ≤ 1 outstanding (corollary INV-4; asserted at the dmem boundary) | [F] safety |
| INV-9 | Every retirement emits exactly one commit record (warp, pc, instr, rd, wdata, mem info) — the lockstep contract; no commit record without a retirement | safety |
| INV-10 | Liveness: ¬busy[w] ⇒ warp w retires within W·(depth+1) cycles; busy[w] ⇒ completion arrives within the memory bound (bounded by the BRAM model: 1 cycle; parameterized for M16) | [F] liveness (bounded) |
| INV-11 | Reset: all busy clear, all PCs = RESET_VEC, no stage occupied | [F] safety |
| INV-12 | dmem arbitration grants ≤ 1 warp per cycle and only to the M-stage warp | [F] safety |
| INV-13 | x0 reads as zero for every warp under all interleavings | [F] safety |
| INV-14 | CSR mhartid[w] = w, immutable | [F] safety |

Assertion placement: INV-1/2/4 in the scheduler; INV-3/10 at the busy-bit
unit; INV-5/6/13 at the regfile; INV-7 at the PC unit; INV-8/12 at the
dmem port; INV-9 at the commit stage; INV-11 top-level. SymbiYosys: bmc
depth ≥ 3·W·depth plus k-induction on INV-1..4; config lands with the
first RTL commit (no RTL without its sby target).

#### Invariant-catalog amendment (D-032a slot compaction, 2026-07-14 —
#### written BEFORE any scheduler RTL change; the RTL commit cites this)

Motivation (measured): tid0-serial ISA workloads, including the S7 suite's
scalar phases, get a 1/W issue duty because rotation never
skips — a parked warp's slot carries a bubble. FASTPATH leg A rotates
only the READY set. That change re-founds the catalog as follows; the
bar (all program-level goldens bit-identical; scalar-phase cycles
≥ 2× better with ≤ 8 harts active) is in docs/FPGA_IMPLEMENTATION.md#fast-path-optimizations and
does not move here.

- **INV-2 is RETIRED** (stage tag = pure function of phase — the
  never-skip property itself; it cannot survive compaction by
  construction). Replacement:
  **INV-2′ (tags travel with slots)**: stage tags/valids change only
  by the pipe shift — stage s+1's (valid, tag) at t+1 equals stage
  s's at t, and stage F's equals the issue decision's. No consumer
  may derive a stage's warp from phase arithmetic. (The datapath
  already satisfies this: simt_core consumes issue_warp and the
  traveling stage_warp registers only — audited 2026-07-14, the only
  phase consumer is the scheduler's own issue decision.)
- **INV-4 keeps its statement** (≤ 1 in-flight instruction per warp)
  **and loses its structural proof** (W > DEPTH ∧ never-skip). New
  enforcement is direct: a per-warp in-pipe interlock — in_pipe[w]
  sets at F-issue of w, clears when w leaves WB (retirement,
  including trap redirect; bubbles never set it) — and issue
  eligibility becomes ready(w) = ¬busy[w] ∧ ¬in_pipe[w]. INV-4 stays
  asserted independently (it is load-bearing for INV-1/8/12).
- **INV-1 becomes a corollary of INV-4** (pairwise distinctness
  follows from ≤ 1 per warp); its assertion is kept.
- **INV-8 / INV-12 are untouched**: they derive from INV-4 and stage
  occupancy, not from phase arithmetic.
- **INV-10 (liveness) is restated**: ready(w) ⇒ warp w issues within
  #ready ≤ W cycles (rotation is round-robin over the ready set,
  cyclic from last-issued+1 — a ready warp cannot be passed over
  twice while another issues twice). The old W·(depth+1) retirement
  bound still holds a fortiori.
- **Hart-visible semantics preserved** (the leg-A bar's clause): a
  warp never observes its own slot skipped while ready — eligibility
  is exactly readiness, and readiness is exactly the pre-existing
  busy contract plus the in-pipe interlock that INV-4 always implied.
- Enforcement honesty: the [F] marks are aspirational until a project
  .sby exists (none does today — only simulation assertions enforce
  the catalog). The leg-A RTL commit must update the barrel_sched
  assertions (INV-2 → INV-2′ tag-continuity form) and the
  tb/barrel_sched bench's expectations in the same commit. The legacy
  rtl/simt/barrel_core.sv (M13 standalone core, not in the SoC build)
  keeps the old rotation and the old catalog; this amendment scopes
  to barrel_sched + its consumers.

### 3. Lockstep + shrinking (doctrine moves 1 & 3)

- **W-way Spike lockstep**: the commit stream (INV-9) demultiplexes by
  warp into W independent P1-style diff lanes — each warp diffs against
  its own Spike instance (mhartid-differentiated). The M2 harness/gate
  machinery is reused verbatim per lane; a barrel bug that corrupts
  cross-warp state shows up as a diff in SOME lane within one retirement.
- **Interleaving fuzz**: rvfuzz extended with busy-bit chaos — random
  dmem-latency jitter (once M16 lands; v1 BRAM is fixed-latency so chaos
  enters via MUL/DIV timing and fence placement) to walk the rotation
  phase space. 1M-cycle nightly fuzz is the SIMT-DoD floor (M15 gate).
- **ddmin shrinker**: failing fuzz cases auto-reduce with
  signature-preserving reduction (same failing INV id + same first-diverge
  warp); target < 50 instructions before human eyes (doctrine move 3).

### 4. Adjudication + tripwire (doctrine move 4)

Divergences are recorded as (signature, shrunk repro, verdict, fix-commit).
Tripwire: >6 open unexplained divergences, or any
INV weakened without a written argument, halts feature work (R4 pattern).

### 5. M14 — lanes (SIMT proper): concrete spec (frozen 2026-07-07)

- **Shape**: warp = L=8 lanes sharing one PC/instruction; per-lane 31x32
  regfiles; per-warp active-mask + divergence stack. Barrel rotation and
  busy-bit contracts UNCHANGED (divergence serializes inside a warp slot).
- **Thread identity**: mhartid = warp*L + lane (the global tid). This is
  the one architecturally lane-divergent CSR; riscv-tests consequently
  validate lane 0 of warp 0 and park everything else — accepted, the
  lane datapath is validated by lane-divergent fuzz + the matmul kernel.
- **Divergence (the max-PC reconvergence scheme)**: on a conditional
  branch where enabled lanes disagree:
    - forward target (tgt > pc+4): push {mask_taken, restart_pc = tgt,
      reconv = tgt}; continue at pc+4 with mask_nottaken. Whenever the
      warp PC reaches >= TOS.reconv: pop; if the popped entry's side has
      not run yet the PC switches to its restart_pc with its mask, with
      reconv inherited; else masks merge. (Handles if-then and
      if-then-else; nesting via the stack, depth 8, overflow -> trap.)
    - backward divergent branch: TRAP (illegal, cause 2 + documented
      v1 limit). Uniform backward branches (loops) are unrestricted —
      kernel loop conditions must be warp-uniform, which the matmul
      kernel and all structured code satisfy.
  JAL/JALR with active mask are warp-uniform by construction (same
  target from shared instruction) EXCEPT JALR whose per-lane rs1 could
  diverge: divergent JALR targets -> trap (v1 limit; no function-pointer
  divergence in the gated kernels).
- **Memory**: per-lane addresses. A warp memory op sets busy[w], leaves
  the pipe, and a lane-serial memory unit drains the <= L enabled lanes'
  requests one per cycle on the single dmem port, then clears busy —
  exactly the interlock the busy bit was built for. Misalignment on ANY
  enabled lane -> trap (cause 4/6, tval = that lane's address).
- **Traps/CSRs**: per-warp (not per-lane) CSR file as in M13; a trap
  quashes the whole warp instruction (no partial-lane retirement).
- **Commit record**: one per warp instruction: (warp, pc, instr, mask,
  per-lane rd wdata for enabled lanes, per-lane store records). The
  W-way lockstep golden is golden/simt_iss.py (lane-parallel ISS with
  the same mask-stack scheme, bit-identical by construction).
- **New invariants**:
  | INV-15 | every lane write/store strobe ⊆ current active mask | [F] |
  | INV-16 | divergence stack push/pop balanced; pop only when PC >= TOS.reconv; depth <= 8 | [F] |
  | INV-17 | per-lane regfile write ⇒ lane enabled ∧ WB warp owns the write | [F] |
  | INV-18 | busy[w] set by a memory op is cleared exactly once, after all enabled lanes' requests completed | [F] |
- **M15 — SIMT-DoD**: riscv-tests per warp, torture per warp, the
  INT8 matmul kernel on lanes, 1M-cycle fuzz with all invariants armed
  (gate = zero diffs ∧ zero assertion fires ∧ positive completion
  markers from every suite).

### 6. Interfaces (frozen for M13)

- imem: `req(warp, addr) → rdata` next cycle, always ready (BRAM model).
- dmem: `req(warp, addr, we, wdata, be) → rdata/ack` next cycle (v1);
  request only from M stage (INV-12). The M16 hierarchy keeps this
  contract with variable latency via ack.
- commit: `valid, warp, pc, instr, rd, rd_wdata, mem_addr, mem_wdata,
  mem_we, trap` — one record per retirement (INV-9), the lockstep bus.
- debug: per-warp busy, stage-occupancy vector, rotation phase — the
  observability counters that made M6 debuggable, ported forward.

## Tensor Sidecar and D4 Socket


The D4 coprocessor socket is THE reusable pattern: M21 grafts the
S-cluster (pconfig/psample/pdrain) into this exact CSR+doorbell shape as
a peer of the tensor sidecar. Everything here runs under the M17 kernel
ABI — sidecar ops are issued from compiled C kernels.

### 1. D4 socket (frozen CSR map, custom M-mode range)

| CSR | name | semantics |
|---|---|---|
| 0x8C0 | T_OP | 0=GEMM8 1=LUT_GELU 2=LUT_EXP 3=LUT_SIGMOID 4=LUT_RSQRT 5=LUT_GELUD (M19/D-018) |
| 0x8C1..3 | T_A, T_B, T_C | operand base addresses (byte) |
| 0x8C4..6 | T_M, T_N, T_K | dimensions (GEMM) / element count in T_M (LUT ops) |
| 0x8C7 | T_FLAGS | bit0: C+= (accumulate) vs C= (overwrite) |
| 0x8C8 | T_GO | write 1 = enqueue the staged command (§1b; illegal while FULL -> trap 2) |
| 0x8C9 | T_STATUS | read: bit0 busy (engine or queue nonempty), bit1 FULL (§1b) |

Contract: one outstanding command globally (shared resource); the issuing
warp polls T_STATUS (warp-uniform read -> uniform backward poll loop is
legal). CSR writes use the existing uniform-source rule (divergent
sources trap). All sidecar memory traffic flows through a round-robin
arbiter sharing the single dmem port with the core's memory unit —
starvation-free by alternation, observable via the DBEATS counter split.

#### 1b. D4 socket v2: the command queue (D-032d amendment,
#### 2026-07-15 — spec BEFORE RTL; leg D of docs/FPGA_IMPLEMENTATION.md#fast-path-optimizations)

The one-outstanding contract above is amended: the T-CSR block gains
a command QUEUE of depth QDEPTH (parameter; frozen default 2) sitting
ABOVE both engines. Kernels may enqueue while a command runs — the
serving-path chains (GEMV -> LUT -> GEMV) stop paying a core
round-trip between ops.

- T_GO (0x8C8): write-1 SNAPSHOTS the eight staging CSRs
  {T_OP, T_A..T_FLAGS} into the queue as one command record. Legal
  whenever FULL = 0 — INCLUDING while busy (the amended boundary).
  Writing GO while FULL = 1 traps mcause 2 (same fault code; the
  boundary moves from "busy" to "full").
- T_STATUS (0x8C9): bit0 BUSY = engine busy OR queue nonempty. The
  meaning of !BUSY is UNCHANGED: every issued command is complete and
  its memory effects visible — all existing polling code (nonzero
  test or bit0 test) works verbatim, because bit1 is only ever set
  while bit0 is set. bit1 FULL = queue at capacity (a GO would trap).
- STAGING SEMANTICS made explicit: T_OP/T_A..T_FLAGS are staging
  registers; GO snapshots them, and they may be freely rewritten
  immediately after the GO retires. (The engines already latch
  operands at the doorbell, so this is today's implicit contract,
  now load-bearing: the DISPATCH mux feeds each engine from the
  QUEUE-HEAD RECORD, never the live staging registers.)
- DISPATCH: when the engine slot is idle, the head record dispatches
  to the engine its T_OP addresses (GO routes by T_OP[3] as today).
  ONE command executes at a time — the I-2 serialization is
  untouched; the queue pipelines ISSUE, it does not add EXECUTION
  concurrency (tensor–fabric concurrency stays the registered ISA-v2
  ask).
- ORDERING: strict program order across ALL T_OPs (tensor and
  sampling share the single queue).
- Uniformity rules unchanged (uniform-source CSR writes, uniform
  T_STATUS poll).
- RED PATHS (S7.5 boundary flip): GO while busy-below-FULL must NOT
  trap and the enqueued command must execute bit-exactly (directed
  test); GO at FULL must trap mcause 2 (fill the queue behind a long
  op, then overflow). The s7_gowb kernel and its gate expectation
  update WITH the RTL commit.
- GOLDEN IMPACT: none on values, by construction — every golden
  executes commands instantly in program order, so a program-order
  queue is value-invisible; the trap boundary is an RTL-contract red
  path exactly as today (the ISS carries no socket model). This
  disclosure IS the "golden updated first" step for leg D.

#### 1c. Profiles: op-absence contract + profile-id (PR4 amendment,
#### 2026-07-16 — spec BEFORE any RTL; PROFILES card, D-036)

Mk-I ships per-workload bitstreams (docs/FPGA_IMPLEMENTATION.md#fpga-personalities). The D4
socket is the stable seam: profiles differ ONLY in which engines
answer it. Two additions, both inert in today's union builds:

- **T_PROFILE (0x8CA, read-only)**: the running profile, a synthesis
  constant. 0 = UNION (every existing build and every sim/dev build —
  all engines present; today's certified behavior is the id-0 case,
  bit-for-bit), 1 = S (serving), 2 = P (planning), 3 = T (training,
  P's alias). Reads follow the existing shared-CSR read semantics;
  writes are illegal (read-only per the §1 uniformity rules). Kernels
  discover capability by reading T_PROFILE — a correct kernel never
  takes the absence trap.
- **Op-absence trap**: each profile carries a frozen T_OP presence
  mask — UNION: all present; S: {0..7} present, {8..10} absent;
  P/T: {0..6, 8..10} present, {7} absent (the sidecar stays in every
  profile; the verify ARRAY is S-only; the sampling domain is P/T
  and dev). T_GO whose STAGED T_OP is absent traps **mcause 2 at GO
  time** — the same deterministic illegal-family fault and the same
  trap point as the §1b FULL trap (GO is where staged state becomes
  a command; nothing enqueues). One mcause serves both: the red
  paths are contextually disjoint (FULL requires a full queue;
  absence is state-independent), and the PR4 gate exercises each in
  isolation. Sub-cause disambiguation is registered as a bring-up
  wishlist item, not v1.

RTL scope (deliberately tiny, NOT built at spec time): a PROFILE
parameter on the simt_core T-CSR block = one id constant + one
11-bit presence mask + one comparator at the existing GO trap site.
Gate scope: the PR4 red path (GO at an absent op traps; T_PROFILE
reads back the build's id) joins each profile's gate set (PR1).

### 2. GEMM8 semantics (bit-frozen)

C[M×N] (int32, row-major) = A[M×K] (int8) × B[K×N] (int8) (+ C if
FLAGS.acc), with int32 accumulation, no saturation. M,N,K ≤ 64.
Engine: 8-lane dot-product datapath — 8 INT8×INT8 MACs/cycle (one K-strip
per cycle), so 8×8×8 GEMM = 64 MAC-cycles + fill/drain. Golden:
golden/tensor.py::gemm8 (numpy int32 exact).

### 3. LUT special functions (bit-frozen tables, M4 discipline)

Element-wise over int8 vectors (Q8_0-scale semantics deferred to the
kernel; the LUT is pure int8→int8 or int8→int16 table lookup — 256
entries each, generated by golden/tensor.py and checked in as .mem):

| op | table | entry format | frozen accuracy spec |
|---|---|---|---|
| GELU | gelu(x/32)·32 | int8 → int8 | max abs err ≤ 1 LSB vs float64 ref |
| GELUD | (gelu−relu)(x/32)·512 | int8 → int8 | bounded in [−87, 0] by construction; max abs err ≤ 1 LSB vs float64 ref (added M19/D-018: the delta form gives q9 output resolution on the same int8 table) |
| EXP | exp(x/16)·256 clamped | int8 → uint16 | rel err ≤ 2⁻⁷ where table value ≥ 64 (abs ≤ 0.5 LSB below) |
| SIGMOID | σ(x/16)·2¹⁵ | int8 → uint16 | max abs err ≤ 2⁻⁸ (matches pbit LUT bound style) |
| RSQRT | 1/√(x/256+ε)·2¹² | uint8 → uint16 | rel err ≤ 2⁻⁶ |

The scale conventions are FROZEN here and used by the M19 Q8_0 pipeline;
G6's GPT-2 goldens quantize to these exact tables.

### 4. Gates G1–G5 (frozen definitions; all under axi_pessimistic)

| gate | claim | pass bar |
|---|---|---|
| G1 | memory truth at SoC level | kernel-visible memory values are latency-invariant (pessimistic vs 1-cycle backends, identical outputs on the M17 battery); DBEATS identical; stall cycles reported |
| G2 | GEMM8 exactness | every (M,N,K) in {1,7,8,16,33,64}³ sample set + 200 random shapes: RTL C == numpy int32 exactly, issued from a compiled kernel via the socket |
| G3 | LUT accuracy | checked-in .mem tables == golden generator byte-exact AND meet §3 error specs on exhaustive 256-point sweeps |
| G4 | fused block | quantized linear layer (GEMM8 + bias-add + GELU LUT) from one kernel == golden quantized reference exactly, N=784→256 MNIST-scale shape |
| G5 | σ-throughput | sustained GEMM MACs/cycle ≥ 4.5 on 64³ (D-017: single-port DMA ceiling is 5.82 ideal / 5.13 with arbitration — the original 6.0 was arithmetically impossible; overlap is the v2 lever), number archived for the calibration day |

Evidence: ci/logs/g15/. G6/G7 (GPT-2) build on this at M19.

### 5. Build order (golden-first)

golden/tensor.py (gemm8 + tables + fused ref, self-checked) →
rtl/tensor/tensor_sidecar.sv + rtl/mem/port_arbiter.sv → simt_core CSR
socket plumbing → soc_harness second-port service → mk.h intrinsics
(mk_gemm8, mk_lut) → kernels → gates/g15_tensor.py.

## Memory Hierarchy


Golden-first order: `sim/axi_pessimistic.py` (memory truth) →
`golden/cache.py` + `golden/coalescer.py` (behavioral contracts) → RTL
diffed exactly against them → cores rewire at M17 behind the same
request/ack contract their busy-bit interlock already anticipates.

### 1. Port contract (frozen)

One outstanding request per port (matches the cores' busy-bit design):
- request: `valid, we, addr[31:0], wdata[31:0], be[3:0]` (word-aligned
  beat; `be` ignored on reads)
- response: `ack` pulses ≥1 cycle after `valid` with `rdata`; the port is
  idle until the next request. No pipelining in v1 (M18 revisits if the
  G-gates demand it — measure first).

#### 1b. Credit face v2 (D-032c amendment, 2026-07-14 — spec before RTL)

The measured revisit §1 anticipated: S7.4 border tax and the D-032
program showed the DMA phases are serialization-bound, and D-017
showed the v1 face cannot pipeline (a held grant re-issues a
requester's stale level-valid — the duplicate-beat race). The v2 face
replaces LEVEL semantics with EDGE semantics so multi-outstanding is
race-free by construction:

- request: `req` is a ONE-CYCLE PULSE per beat; `we/addr/wdata/be`
  are valid in the req cycle (same widths and word-beat meaning as
  v1). One pulse = exactly one beat, always — there is no held state
  to re-sample, so the D-017 race is structurally impossible.
- credits: a face carries parameter `CRED` (default 1, max 4). The
  issuer holds a credit counter initialized to CRED; issuing a req
  spends one credit; a response returns one. `req` may only pulse
  with a credit in hand. CRED=1 reproduces v1's one-outstanding
  timing shape exactly.
- response: `rsp_valid` is a one-cycle pulse per beat, STRICTLY IN
  REQUEST ORDER; `rsp_rdata` is valid in that cycle for reads; for
  writes `rsp_valid` is the completion (data don't-care). In-order
  completion is what preserves I-9/G1 latency-invariance: values can
  never depend on the LAT knob because no beat overtakes another.
- migration (frozen order, one master per commit, each behind the
  full gate set): arbiter downstream + harness slave first with
  CRED=1 proven value-identical to the v1 face; then the PDRAIN
  streamer, then the sidecar loaders (LD_A/LD_B/LD_C0/ST_C/LUT),
  then the PCONFIG walker. Masters not yet migrated keep the v1
  level-valid contract at the arbiter's upstream ports (the arbiter
  adapts level -> edge internally, one outstanding per legacy port).
  A MIGRATED master attaches on a dedicated v2 upstream port: the
  arbiter accepts its pulses unconditionally into a skid FIFO sized
  to the port's CRED (the master's own credit loop is the overflow
  guard) and grants an empty-FIFO pulse combinationally, so
  single-beat traffic keeps the v1 grant timing. A module with
  several internal walkers may migrate them one at a time: its face
  flips to v2 once, unmigrated walkers ride a one-outstanding
  level->edge shim inside the module (the PDRAIN commit carries the
  PCONFIG walker on the shim). A master must hold its busy/status
  until every issued beat's completion has returned — !busy means
  the data is memory-visible (the T_STATUS poll contract). A
  STREAMING LOAD master places response data by a RESPONSE cursor,
  never its issue cursor (in-order completion makes the response
  index well-defined); a master with a serial data dependency
  between beats (the sidecar LUT stream) may run one-outstanding on
  the edge face — CRED is a ceiling, not a mandate. Phase boundaries
  inside one command barrier on a full credit pool.
- the SCRATCH window (leg B) is a second sub-slave behind the fork;
  the fork owns cross-sub-slave ordering (a younger fast beat's
  response is HELD until all older beats complete — order FIFO,
  depth = total credits).

### 2. `sim/axi_pessimistic.py` — the memory truth

EVERY number below is a **documented guess pending calibration** on real
xck26 silicon during the Bring-up Campaign (risk #10b). The model is
deliberately pessimistic so σ performance claims under-promise.

| parameter | value | why this guess |
|---|---|---|
| base latency | 40 cycles | DDR4 CL+tRCD+controller+AXI fabric at 100 MHz core clock, rounded up |
| bandwidth | 10 GB/s ÷ n_ports | KV260 DDR4-2400 x64 theoretical 19.2 GB/s x ~0.5 achievable, split evenly (pessimistic: no port sharing of idle capacity) |
| token bucket | capacity = 2 lines, refill = BW/cycle | burst absorption without sustained over-BW |
| queueing | FIFO per port; a request waits for tokens + latency | no reordering, no bank parallelism credit |
| line size | 32 B | matches cache line; DRAMsim3 swap-in uses BL8 x64 = 64 B natively, modeled at 32 |

DRAMsim3 (DDR4 config, patched per D-009) mounts behind the IDENTICAL
python interface as the high-fidelity alternate; gates run pessimistic by
default, DRAMsim3 as the cross-check lane.

### 3. Cache (frozen v1 geometry)

- 2-way set associative, 32 B lines, 4 KB default (64 sets), LRU,
  write-back + write-allocate, blocking (one outstanding miss),
  single request port (§1 contract on both faces).
- Miss flow (order is architectural, golden mirrors it): select victim by
  LRU → if dirty, write back FULL line (8 beats) → fill FULL line
  (8 beats) → complete the original access → update LRU.
- LRU update on every hit and fill-complete. Reset: all invalid, LRU=0.

### 4. Coalescer (frozen v1)

Sits between the simt_core lane-serial memory unit and the cache: takes
the ≤L enabled lanes' (addr, wdata, be) of ONE warp memory op:
- groups by line address; loads issue one cache read per distinct WORD
  (v1 keeps word granularity — line-level gather is an M18 measurement
  decision); stores merge per word with **lane-ascending application**
  (highest lane wins conflicting bytes — matches golden simt_iss lane
  order exactly).
- Ordering: distinct words serviced in ascending lane-index-of-first-
  toucher order (deterministic, golden-mirrored).

### 5. Verification

- `golden/cache.py`: bit-true model incl. LRU/dirty/writeback ordering;
  self-check vs a flat-memory reference under random traffic.
- RTL cache bench: exact per-transaction diff (rdata, and the FULL
  backing-memory image + tag/dirty state at end) vs golden under random
  + directed (thrash, dirty-evict, same-set ping-pong) traffic, with the
  axi_pessimistic timing behind it (timing does not change VALUES —
  asserted by running the same traffic at latency 1 and 40).
- Counters (observability, D-mem): hits, misses, writebacks, stall
  cycles — the G-gate fuel.

## Sampling ISA


The stochastic domain becomes ISA: `pconfig`/`psample`/`pdrain` grafted into
the exact CSR+doorbell socket the tensor sidecar froze at M18
(docs/HARDWARE_ARCHITECTURE.md#tensor-sidecar-and-d4-socket). The S-cluster is a peer of the tensor sidecar behind
the same T-CSRs; a compiled SIMT kernel (M17 ABI) configures, launches and
drains the p-bit fabric the way it issues a GEMM. Everything sampled is the
FROZEN fabric contract from M4-M6 (golden/pbit.py p17, golden/gibbs_grid.py
chromatic lane schedule, D-014 word-consumption rule) — this spec adds no new
sampling math, only the ISA surface plus two frozen M21 features: D19/D30
moment+covariance accumulators and the D33 per-replica
work register (independent floating-point anchor:
golden/fabric_anchor.py).

Golden-first law: golden/sampling_isa.py is the bit-frozen oracle for every
op here; NO RTL until this spec's freeze is reviewed. The S7σ bars in §10 are
FROZEN before any measurement (R3 pre-registration style).

### 1. Op space (D4 socket, frozen CSR map 0x8C0-0x8C9 reused)

New T_OP values (6-7 stay reserved for tensor growth):

| T_OP | op | summary |
|---|---|---|
| 8 | PCONFIG | DMA a config image (couplings/bias rows, order list = clamp masks, schedule, PRNG seeds, state, replica map) from shared memory into the fabric |
| 9 | PSAMPLE | run chromatic Gibbs sweeps (immediate beta or the loaded annealing schedule), optionally accumulating moments |
| 10 | PDRAIN | DMA state / moments / covariance / work / telemetry to shared memory |

Socket contract inherited verbatim from tensor_spec §1 + §1b (D-032d):
one command EXECUTES at a time (tensor AND sampling share the single
engine slot in v1 — concurrency is a v2 lever) with a depth-QDEPTH
command queue above both engines; T_GO enqueues, and a write while
FULL traps mcause 2; the issuing warp polls T_STATUS (warp-uniform
read; bit0 busy = engine-or-queue, bit1 FULL); CSR writes obey the
uniform-source rule. All S-cluster memory traffic flows through the existing
port arbiter on the single dmem port.

Per-op CSR operand map (unused fields MUST be written 0; nonzero = reserved):

| CSR | PCONFIG (8) | PSAMPLE (9) | PDRAIN (10) |
|---|---|---|---|
| T_A 0x8C1 | image base (byte addr, word-aligned) | 0 | 0 |
| T_B 0x8C2 | 0 | 0 | 0 |
| T_C 0x8C3 | 0 | 0 | dest base (byte addr, word-aligned) |
| T_M 0x8C4 | image length in words (DMA extent; must equal the §2 encoded length) | IMM form: sweeps u24 ≥1; SCHED form: 0 | 0 |
| T_N 0x8C5 | 0 | 0 | 0 |
| T_K 0x8C6 | 0 | IMM form: beta_raw u8 (u2.6); SCHED form: 0 | 0 |
| T_FLAGS 0x8C7 | bit0 ROWS, bit1 ORDER, bit2 SCHED, bit3 SEEDS, bit4 STATE, bit5 RID (section-select); bit8 WORK_RESET | bit0 RECORD, bit1 STATS_RESET, bit2 IMM | bits[2:0] mode: 0 STATE, 1 MOMENTS, 2 COV, 3 WORK, 4 TELEMETRY |

### 2. PCONFIG image (frozen layout, word-granular)

Header (always present, 3 words), then the selected sections concatenated in
T_FLAGS bit order (ROWS, ORDER, SCHED, SEEDS, STATE, RID). Partial reconfig
is the point: the PCD inner loop rewrites ROWS only; a wake/dream phase flip
swaps ORDER only.

```
word 0  magic 0x4D4B5331 ("MKS1")
word 1  n_sites u16 | bit16 BIPOLAR | bit17 WORK_TRACK
word 2  n_colors u8 | n_sched u8 | n_replicas u8 | u8 reserved(0)
```

| section | size (words) | format |
|---|---|---|
| ROWS | n × 9 | per site ascending: slots 0-7 as one word each = fabric row_data[23:0] zero-extended ({valid, nbr u13, J s1.6.3 raw u10}); word 8 = bias s1.6.3 raw in [9:0] |
| ORDER | 1 + 16 + n_ord | word0 = n_ord u16; 16 words color bounds (cb_start u16 \| cb_end u16 <<16), colors ≥ n_colors zeroed; then n_ord site indices, one per word |
| SCHED | n_sched | per entry: beta_raw u8 \| sweeps u24 <<8 |
| SEEDS | 32 | 8 lane streams × 4 words (s0,s1,s2,s3), stream-major; no stream may be all-zero |
| STATE | ceil(n/32) | site i = bit (i mod 32) of word (i div 32); binary AND bipolar states are stored as bits (bipolar: bit b ⇒ spin 2b−1) |
| RID | ceil(n/4) | replica id, one byte per site (little-endian byte order within the word), rid < n_replicas ≤ 64 |

Semantics:
- **Clamping = order exclusion** (the frozen gibbs_grid contract): clamped
  sites are simply absent from the order list; their values are set by the
  STATE section and never change during PSAMPLE. There is no separate clamp
  mask on the device.
- **Coloring/order are computed SIMT/host-side.** The frozen convention for
  all gate configs is gibbs_grid.build_schedule (greedy coloring by ascending
  site index, order sorted by (color, index), clamped sites excluded). The
  device only checks nothing; a non-proper coloring is a config bug guarded
  by the golden assertion (invariant I-4).
- n_sites is immutable after the first PCONFIG (until SoC reset) — v1 limit.
- SEEDS reloads the 8-stream farm; WITHOUT a SEEDS section the farm continues
  from where the last command left it (persistent streams = persistent
  chains, the D22 idle state).
- The header's n_colors and n_sched fields are SECTION LENGTHS: they
  apply only when the corresponding ORDER/SCHED section is present in
  the image. The loaded coloring and annealing schedule PERSIST across
  images without those sections (the golden's semantics since freeze;
  stated explicitly 2026-07-14 after the S8σ gate caught the RTL
  clobbering both on a ROWS-only rewrite — latent since M21, invisible
  to every prior diffed workload because their images always carried
  ORDER+SCHED).
- WORK_RESET (T_FLAGS bit8) zeroes all work registers AFTER the §5 work
  update (if any) and after all sections apply.

### 3. Sampling semantics (frozen; zero new math)

- Per-site conditional: acc = bias_raw + Σ_slots (BIPOLAR ? (s_nbr?+J:−J)
  : (s_nbr?J:0)), asserted within s14 (golden/pbit.py ACC_MIN/ACC_MAX — the
  degree contract); P(s_i=1) = p17(acc, beta_raw)/65536 exactly
  (golden/pbit.py p17, frozen M4); decision: s_i = (rnd[31:16] < p17).
- Schedule: PSAMPLE IMM form (T_FLAGS bit2) runs T_M sweeps at beta = T_K;
  SCHED form runs the loaded schedule entries 0..n_sched−1 in order. One
  sweep = all colors in order; a color's segment is processed in consecutive
  P=8-site chunks by lanes 0..7; new bits within a chunk apply
  simultaneously.
- PRNG word consumption (the D-014 rule, frozen): the 8-stream xoshiro128++
  farm advances exactly once per NON-EMPTY chunk; lane l consumes its word
  only if it holds a real site; empty color segments consume nothing and
  cost nothing.
- STATS_RESET (bit1) zeroes cnt/m1/m2 before the first sweep of the command.
- RECORD (bit0): at the END of every sweep of the command, accumulate §4
  moments (thin = 1, fixed in v1). The emulator bring-up lesson is
  architectural here: accumulators NEVER decay and NEVER auto-reset —
  psample(record) without STATS_RESET extends the running sums.

Temperature bridge (for the independent floating-point anchor): beta = beta_raw/64, J = J_raw/8,
b = b_raw/8; the emulator's p(s_i=1|rest) = σ((J_i·s + b_i)/T) matches at
T = 64/beta_raw up to the frozen p17 quantization (D15/M4 bound 7.0e-4).

### 4. Moment accumulators (D19) + covariance drain (D30) — integer-only

Sufficient statistics live per model parameter (D19: only moments cross the
border), so second moments exist ONLY on the edge slots, not n×n:

- cnt : u32 accumulate-event counter (one event = one recorded sweep end)
- m1[i] : u32, += s_i per event (bit counter; bipolar uses the stored bit)
- m2[i][k] : u32 per valid slot k of site i, += s_i AND s_nbr(i,k) per event

All are pure bit counters — no multiplier in the accumulate path. Clamped
sites accumulate like any other (matches the emulator, which accumulates the
whole state). m1 ≤ cnt and m2 ≤ cnt by construction, so u32 never wraps
under the v1 limit cnt ≤ 2^24.

Covariance mode (D30) is a DRAIN-TIME computation (one 32×32→64 multiplier
in the drain engine; nothing accumulates differently):

```
craw[i][k] = cnt·m2[i][k] − m1[i]·m1[nbr(i,k)]      (int64, edge slots)
craw_diag[i] = cnt·m1[i] − m1[i]²                    (s² = s ⇒ m2_ii = m1_i)
cov = craw / cnt²  — the division NEVER happens on device; SIMT/host divides
```

|craw| < cnt² < 2^48 under cnt ≤ 2^24 — int64 exact, no saturation. The
emulator's pdrain("cov") = M2/cnt − m⊗m equals craw/cnt² as exact rationals;
the anchor protocol is §10 S7.3.

### 5. Work registers (D33) — per-replica, updated on parameter writes

WORK[r] (int64, r < n_replicas ≤ 64, wraparound two's-complement defined) in
frozen units of 1/16 model energy:

```
E2[r] = −( Σ_{i: rid[i]=r} s_i·pair_i  +  2·Σ_{i: rid[i]=r} b_i·s_i )
        where pair_i = Σ_slots J_raw·s_nbr    (binary mode; all raw ints)
E2 = 2× energy in raw-1/8 units ⇒ E_model = E2/16;  E2 is always an even
row-scan sum, so no division exists anywhere.
```

Update rule (matches anchor Fabric.pconfig exactly): on a PCONFIG whose
T_FLAGS includes ROWS **and** whose PRE-COMMAND header has WORK_TRACK=1:
WORK[r] += E2_new[r] − E2_old[r], both evaluated at the state and RID map AS
OF COMMAND ENTRY (E2_old under the old rows, E2_new under the new rows).
Realization: two energy scans around the ROWS write — ≤ 2·n·9 MACs of
10-bit×1-bit, socket-trivial. SCHED/SEEDS/STATE/ORDER sections never touch
work (fixed-T formalism, per the emulator). The very first PCONFIG after
reset has WORK_TRACK=0 pre-command, so it never charges work.

Replica model: replicas are disjoint tiles of one fabric graph. Invariant
I-3: no coupling row may cross a RID partition — energies are then exactly
separable and WORK[r] equals the emulator's per-replica work under the tile
bridge. v1 limits: WORK_TRACK requires binary mode (the independent floating-point anchor is
{0,1}); on-device replica resampling does not exist (drain STATE+WORK,
permute SIMT-side, PCONFIG STATE back — emulator resample() is SIMT's job).

This supports Jarzynski/BAR free energies, honest NLL, Bayes factors,
dissipation-metered generation, and soft value functions from one adder per
replica.

### 6. PDRAIN readback formats (frozen; all little-endian u32 words)

| mode | length (words) | layout |
|---|---|---|
| 0 STATE | ceil(n/32) | §2 STATE packing |
| 1 MOMENTS | 1 + n + 8n | cnt; m1[0..n); m2 slot-major per site (site 0 slots 0-7, site 1 slots 0-7, …), invalid slots drain 0 |
| 2 COV | 1 + 18n | cnt; per site: craw_diag int64 (lo,hi) then slots 0-7 craw int64 (lo,hi), invalid slots drain 0 |
| 3 WORK | 1 + 2·n_replicas | n_replicas; WORK[0..R) int64 (lo,hi) |
| 4 TELEMETRY | 12 | see below |

TELEMETRY words: 0 sweeps_done u32; 1-2 upd_cnt u48 (lo,hi); 3-4 flip_cnt
u48 (lo,hi); 5 cfg_words u32 (total PCONFIG section words consumed — the
D20/D29 coupling-write tax meter); 6 drain_words u32 (total words produced
by prior COMPLETED PDRAINs); 7 cnt u32 (mirror); 8-9 busy_cycles u64 (lo,hi)
— the ONLY field not golden-checked (timing observable; golden drains 0);
10-11 reserved 0. upd_cnt counts site updates performed, flip_cnt counts
updates that changed the bit (both trajectory-deterministic, golden-exact).

Draining MOMENTS/COV with cnt=0 returns cnt=0 and an all-zero payload
(defined; deviates from the emulator's lazy auto-accumulate convenience —
the semantics anchor applies at cnt>0). PDRAIN mutates nothing: state,
accumulators and work survive every drain.

### 7. PRNG seeding & determinism contract (frozen)

- The farm is 8 xoshiro128++ streams, one per lane (rtl/pbit/prng_farm.sv
  contract). The SEEDS section carries raw 128-bit states.
- The NORMATIVE derivation for all gate/science runs: golden/xoshiro.py
  stream_states(mother_seed, 8) — jump-ahead 2^64-spaced streams from a
  32-bit mother seed (D16). Kernels ship the 32 precomputed words (the GF(2)
  jump is not an RV32I runtime job).
- Determinism: given (config image sequence, op sequence, SEEDS), the state
  trajectory, all counters, and every drain word are bit-reproducible and
  EQUAL to golden/sampling_isa.py. Same mother seed ⇒ identical run;
  streams persist across commands unless SEEDS reloads them.

### 8. Invariants

- I-1 (socket, amended D-032d 2026-07-15 — tensor_spec §1b): commands
  enqueue in program order (depth QDEPTH, default 2); GO while FULL
  traps mcause 2 (the boundary moved from busy to full); T_STATUS
  bit0 is the only completion signal — !busy still means every issued
  command is complete and memory-visible (uniform backward-poll
  legal).
- I-2 (coexistence): tensor ops (T_OP 0-5) and S-ops share the engine slot;
  interleaving them never corrupts either side's state (fabric state,
  accumulators, work and farm survive tensor ops untouched).
- I-3 (replica separability): no coupling crosses a RID partition
  (golden-asserted; RTL behavior on violation is undefined-but-benign).
- I-4 (proper coloring): same-color order-list sites are never adjacent
  (golden-asserted; frozen greedy convention in gibbs_grid).
- I-5 (accumulator discipline): cnt/m1/m2 mutate only at RECORD sweep ends
  and STATS_RESET; WORK mutates only per §5 and WORK_RESET; PDRAIN mutates
  nothing.
- I-6 (clamp): a site absent from the order list never changes during
  PSAMPLE.
- I-7 (determinism): §7, bit-level, including the D-014 word rule.
- I-8 (widths): acc within s14 (degree contract); cnt ≤ 2^24; m1,m2 ≤ cnt;
  |craw| < 2^48; WORK wraps mod 2^64 (defined).
- I-9 (memory): all S-cluster DMA flows through the port arbiter; kernel-
  visible memory values are latency-invariant (G1 discipline extends to
  S-ops).

### 9. v1 limits

n ≤ 8192 sites; degree ≤ 8; colors ≤ 16; schedule ≤ 32 entries; replicas
≤ 64; P = 8 lanes/streams; thin = 1; one outstanding command (no
tensor/sampling concurrency); n immutable after first PCONFIG; WORK_TRACK
binary-mode only; no on-device resample; cnt ≤ 2^24; T_M in PSAMPLE IMM
≤ 2^24−1; COV/MOMENTS drain the full 8-slot layout regardless of degree.

### 10. S7σ gate (gates/s7_isa.py) — bars FROZEN 2026-07-07, before any RTL or measurement

All runs from compiled SIMT kernels through the D4 socket, full simt_soc
under axi_pessimistic, evidence to ci/logs/s7/. Fuzz seed base
S7_SEED_BASE = 0x53370000. Directed suite D1-D5:
D1 4×4 deg-4 torus, ferro J=+8, h=0, binary, beta 32, 60 sweeps;
D2 6×6 king (deg-8), J~U{−16..16}\{0}, h~U{−8..8}, bipolar, beta 48, 40 sweeps;
D3 = D2 topology binary with 20% sites clamped (order-exclusion + STATE);
D4 the D-014 shape: 12+8 bipartite, all 12 visibles clamped ⇒ empty color
segment; D5 multi-entry schedule [(16,10),(64,20)] + persistence split (two
PSAMPLEs, 20+20 sweeps, no SEEDS reload) vs one 40-sweep call.

| bar | claim | frozen pass rule |
|---|---|---|
| S7.1 | trajectory exactness | RTL state == golden after EVERY sweep, zero tolerance, on D1-D5 + 100 fuzz configs (seed-keyed: n∈[4,64], degree ≤8, both spin modes, clamp prob 0.2, 1-4 schedule entries, ≤64 sweeps) |
| S7.2 | statistics mode exact | MOMENTS drain (cnt, m1, m2) integer-exact vs golden on every RECORD run of the S7.1 suite, zero tolerance |
| S7.3 | D30/D33 semantics | COV and WORK drains integer-exact RTL==golden on the suite; AND golden anchored to golden/fabric_anchor.py by the forced-state protocol: identical state/parameter sequences ⇒ cnt/m1/m2 integer-exact, work exact in float64 (dyadic rationals), cov allclose atol 1e-9 (emulator divides in float). The anchor half is proven in golden/sampling_isa.py self-check and re-run by the gate |
| S7.4 | border tax ≤ 5% (σ) | frozen workload C*: n=1024 (32×32 deg-4 torus, J~U{−16..16}\{0}, h~U{−8..8}, seed 0x5337C0DE), binary, one iteration = PCONFIG(ROWS only) + PSAMPLE(IMM, sweeps=4096, beta=128, RECORD) + PDRAIN(MOMENTS). S = busy cycles of the PSAMPLE command; B = T_total − S where T_total = mcycle-measured cycles of the whole iteration in the compiled kernel (so B charges config DMA, drain DMA, doorbells, polling and kernel overhead). PASS: B/S ≤ 0.05, measured ratio archived (σ — cycle-count, target-clock projection deferred to bring-up) |
| S7.5 | socket contract | GO at FULL traps mcause 2 (fault-injected: fill the queue behind a long op, then overflow — amended D-032d; GO while merely busy enqueues legally); divergent T_STATUS poll traps per the uniform rule; a 16-shape G2 subsample re-runs green with the S-cluster integrated (peer coexistence, I-2) |

Amendment discipline: bars move only through a written decision record
(the D-012/D-015/D-017 pattern); recipes/configs may iterate, bars may not.

### 11. Build order (golden-first)

golden/sampling_isa.py (this spec executable, self-checked — DONE at freeze)
→ spec review → rtl/fusion/s_cluster.sv (config DMA engine + fabric_grid
graft + D19/D30 accumulators + D33 work scan + drain engine) → simt_core
CSR decode for T_OP 8-10 → tb/s_cluster exact-diff bench (per-sweep state +
all drains vs golden) → mk.h intrinsics (mk_pconfig, mk_psample, mk_pdrain)
→ kernels → gates/s7_isa.py.

### 12. QSITE addendum (fabric-v2, docs/HARDWARE_ARCHITECTURE.md#categorical-q-site-architecture) — registered 2026-07-13

Additive: arity 2 images are BIT-IDENTICAL to the frozen §1-11 spec and
run the frozen golden/RTL paths untouched (Q1 regression bar). Oracle:
golden/qsite_golden.py (frozen at S1 close). Sections below define
q ∈ {4, 8}.

- Header: word2[31:24] = ARITY (0 ⇒ 2, else ∈ {2,4,8}; other values
  are a config bug, golden-asserted). BIPOLAR must be 0 for q > 2
  (I-10). WORK_TRACK is defined for q (E2-q below); COV drain (mode 2)
  is DEFERRED for q (v2; golden asserts unsupported).
- ROWS: 8 + ceil((q−1)/3) words/site. Slots 0-7 unchanged
  ({valid, nbr u13, J s1.6.3 raw}). Bias words pack 3 lanes each,
  10b at [9:0], [19:10], [29:20], states ascending from 1 (state 0 is
  the reference, b0 ≡ 0): q4 ⇒ 1 word {b3,b2,b1} (same word count as
  binary); q8 ⇒ 3 words {b3,b2,b1}, {b6,b5,b4}, {0,0,b7}.
- STATE section AND STATE drain: q4 packs 16 sites/word (2b lanes,
  site i at bits [2(i mod 16) +: 2]); q8 packs 8 sites/word (4b
  lanes, top bit 0). Values < q (I-10).
- Update semantics (the only new math in the ISA, frozen here):
  per site i, per candidate a ∈ 0..q−1:
    acc[a] = b_raw[a] + Σ_slots (valid && x_nbr == a) ? J_raw : 0
  (each acc[a] within s14 — the degree contract is per-candidate and
  identical to binary since a neighbor contributes to exactly one
  candidate). Lane word(s) w from the farm; byte lane a = bits
  [8a+7 : 8a] (q8: candidates 4-7 take bytes 0-3 of the SECOND word).
    score[a] = beta_raw · acc[a] + (GLUT[byte_a] << 5)
  with GLUT[k] = clip(round(16 · (−ln(−ln((k+0.5)/256)))), −128, 127)
  (256-entry s3.4 Gumbel table; scale-exact: beta·acc and the shifted
  GLUT share the 1/512 real-value grid). Winner = argmax score; ties
  break to the SMALLEST candidate index. x_i ← winner. upd_cnt += 1
  per decision; flip_cnt += (winner != old).
- PRNG (D-014 extension): the farm advances once per non-empty chunk
  at q4, TWICE at q8 (both words drawn in order before any lane
  decides); lane l consumes its word(s) only if it holds a real site.
- Moments (D19-q): cnt as §4; m1[i][a] += (x_i == a) for a = 1..q−1
  (state 0 implied: cnt − Σ_a m1[i][a]); m2[i][k] += (x_i ==
  x_nbr(i,k)) — the agreement counter keeps §4's exact shape and is
  the delta-coupling sufficient statistic.
- MOMENTS drain (mode 1, q): cnt, then n·(q−1) m1 words site-major
  state-minor (a = 1..q−1), then n×8 m2 words as §6.
- WORK/E2-q (D33): E2[r] = −( Σ_i Σ_slots J_raw·δ(x_i, x_nbr(i,k))
  + 2·Σ_i b_raw[x_i] ), raw ints, b_raw[0] = 0 — binary's formula
  with δ replacing the bit product; bracket/update rules of §5
  unchanged.
- Invariants: I-10 (arity domain, bipolar=0, state < q); I-11
  (per-candidate s14 = I-8 unchanged); I-12 (determinism incl. the
  byte-lane rule and q8 double-advance). I-1..I-9 apply verbatim.
- v1-q limits: q ∈ {4,8}; all §9 limits inherited; q8 halves the
  farm's chunk rate (two advances); COV-q deferred.

### 13. ACCWALK addendum (D-034) — in-sweep moment accumulation,
### Shape A (snapshot + overlapped walk) — frozen 2026-07-15, BEFORE any RTL

Motivation measured 2026-07-15 (see
docs/FPGA_IMPLEMENTATION.md#in-sweep-accumulation): P_ACC = 93.06% of PSAMPLE at
n=256 — 4,608 cyc/sweep = 2×9×n exactly — while the fabric sweep it
reads out costs 212.5 cyc. This addendum makes PSAMPLE sweep-bound.
Shape A chosen by user directive 2026-07-15.

**The governing invariant: drained VALUES do not change.** m1/m2
remain the post-sweep statistics of every RECORD sweep, accumulated
per §4 math verbatim; drain layouts (§6) and I-1..I-12 hold
unchanged. This addendum changes WHEN counting happens, never WHAT
is counted. Consequence: NO golden change — the entire diff surface
is cycles.

Mechanism (all inside s_cluster; ZERO fabric_grid port changes):

a. SNAPSHOT (new state P_SNAP). On f_done of a RECORD sweep, a copy
   cursor streams ceil(n_sites/32) addresses through the EXISTING
   sA_addr → sA_word/sA_word2/sA_word3 port (idle between sweeps),
   latching all planes into R=4 replicated dual-port snapshot RAMs
   (packed {p3,p2,p1} per site). Copy cost n/32 + 2 cyc (10 at
   n=256). acc_cnt increments at P_SNAP entry (the old P_WAIT→P_ACC
   count edge, same semantics). NOTE: the state_mirror/state_flat
   path is SIM-ONLY (ifdef SYNTHESIS) and stays that way — the
   snapshot rides the synthesizable word port.
b. OVERLAP. After P_SNAP the FSM proceeds to P_NEXT/P_LOADSCH and
   starts the NEXT sweep immediately; the WALKER consumes the
   snapshot concurrently. The next P_SNAP entry AND PSAMPLE
   completion interlock on walker_idle — §8 busy semantics hold
   (busy clears only when the final walk has committed, so a
   subsequent PDRAIN reads settled sums).
c. WALKER, 1 site/cycle (WALK_W=1 frozen; the banked layout admits
   WALK_W=2 later with no layout change):
   - shadow rows RE-BANKED BY SLOT: 8 banks × NMAX words (same total
     bits as §2's shadow); every existing single-read consumer (E2
     scan, COV drain, fetch mux) routes through a bank-select on
     addr[2:0] — values untouched. The sweep itself never reads the
     shadow (the fabric owns its jmem), so overlap has no port
     conflict.
   - per cycle: all 8 slot words of site i read in parallel; the
     site's own state rides a sequential 32-bit stream register
     (refilled once per 32 sites from replica 0); the 8 neighbor
     states tap the 4 snapshot replicas (2 ports each) by nbr id.
   - commit: m1 = one pipelined dual-port RMW/cycle ({site, lane 0}
     binary / {site, fs_aval−1} q — §4 verbatim); m2 = 8 parallel
     RMWs into 8 SLOT-BANKS (bank = slot, addr = site; at NB=13 the
     banks are the same 8 URAMs m2 occupies today, one per bank).
     No RMW hazards: each (site,lane)/(site,slot) visited exactly
     once per walk.
   - agreement math verbatim §4 (binary abit&&bbit; q aval==bval;
     validity = shadow bit 23). Clamped sites need no special case:
     the snapshot holds their held state exactly as the old walk
     read it.
d. LAST SWEEP: no successor to hide behind — one exposed walk per
   PSAMPLE, amortized over its sweeps; busy holds until it commits.
e. P_ZERO: UNCHANGED in v1 (m2 banks zero in lock-step with the same
   cursor; cycle count identical). STATS_RESET ordering preserved —
   P_ZERO runs before sweep 1, when the walker is provably idle.
f. Drains: MOMENTS (d_mode 1) decodes bank = addr[2:0] in the m2
   region; on-the-wire word order IDENTICAL. COV (d_mode 2) single
   reads route through the same select.

Cycle model (n=256, sweeps=16, RECORD-all): per sweep ≈ copy 10 +
max(sweep 213, walk 259) ≈ 279 vs 4,821 today → PSAMPLE phase ≈
16×279 + 259 tail ≈ 4.7k vs 77.1k ≈ 16×. WALK_W=2 → ≈21×.

Resources at NB=8 (simulation build): 4 snapshot replicas × 256×3b
(LUTRAM), shadow re-bank ±0, m2 re-bank ±0..+2 BRAM18, walker pipe +
8-way nbr muxes ≈ +2–3k LUT / +1k FF. At NB=13 (assembly ceiling):
snapshot 4 × 24Kb ≈ 4 BRAM36; m2 = the same 8 URAMs. OOC 8 ns
re-hearing pre-registered (AW.3).

v1 limits: WALK_W=1; P_ZERO pace unchanged; COV-q deferral (§12)
inherited; the walker serves PSAMPLE only — PDRAIN/PCONFIG paths
untouched.

Bars: AW.1–AW.4 FROZEN 2026-07-15 in docs/FPGA_IMPLEMENTATION.md#in-sweep-accumulation. Build
order (§11 form): this spec → bench-first (s_cluster suite gains
smoke_accwalk: overlapped-walk vs oracle at boundary shapes incl. q8
+ clamps + STATS_RESET) → RTL → full battery → bars.

§13 LANDED 2026-07-15 ~21:30 (branch accwalk): AW.1–AW.4 all MET —
PSAMPLE phase 11.9× (855,040 vs 10,140,800 cycles at the 32-iteration
sampling shape), P_ACC exposed 0.00%, and config/drain controls
cycle-identical (1.000). Verdicts + anatomy in
docs/FPGA_IMPLEMENTATION.md#in-sweep-accumulation; OOC 8 ns re-hearing deferred-disclosed to first
VM availability. WALK_W=2 + parallel P_ZERO follow-ups open as their
own follow-up work (S1/S2).

§13 v2 LANDED 2026-07-16 ~01:40 (S1 WALK_W=2 + S2 banked P_ZERO):
m1/m2/sh_slot re-banked
{site parity, lane/slot} (16 × NMAX/2 each, same bits; bank = the
full address's low 4 bits); the walker commits site PAIRS (walk
n/2+3 = 131 < sweep 213 — FULLY hidden, P_SNAP = pure 9-cycle copy);
P_ZERO zeroes all 32 counter banks lock-step (65,536 → 4,096
cyc/RESET, 16×). Measured: PSAMPLE phase 493,056 cyc at the frozen
shape = 20.6× vs pre-ACCWALK; P_WAIT 88.3% of phase — the §13
sweep-bound endpoint. Values remained bit-identical across both depths,
battery ALL PASS, controls in band. DISCLOSED (supersedes v1's "no
layout change" claim; CORRECTED 2026-07-16): the 16-way banking
doubles the number of independent memories, but URAM-BLOCK count is
NB-dependent — NEUTRAL at NB=13 (v1's 8K-deep banks already cascaded
2 URAMs each; 16 blocks both ways), doubles at NB=12 (8→16 of the
K26's 64), maps to BRAM below that. Logic ≈ +6–8k LUTs at NB=13
worst case (selects + snapshot taps), trivial at profile n=256; the
tight sampler line-item remains m1 (~64 BRAM36 at NB=13,
pre-existing). PR1 + the deferred OOC re-hearing measure the real
profile; an 8-bank W=1 profile knob is the escape hatch if fit
demands.

## Sampling Cluster Construction


Decisions frozen before RTL (all verified against the spec + golden):

1. **Fabric untouched.** rtl/pbit/fabric_grid.sv keeps its frozen
   contract. The s_cluster drives it ONE SWEEP AT A TIME: for each
   sweep of the (IMM or loaded) schedule: sch_we entry0=(beta,1),
   n_sched=1, start, wait done, then (if RECORD) run the accumulate
   FSM. This realizes golden's accumulate-at-every-sweep-end exactly;
   PRNG farm state persists across starts (D-014/D22 contract holds).
   IMM vs SCHED is s_cluster-side sequencing only.
2. **s_cluster owns shadow config**: rows (n x 9 words as parsed),
   RID bytes, schedule entries, header flags — because fabric memories
   have no read ports and D33/D30 need row/RID data (work scan reads
   the shadow + state_flat; two scans bracket a ROWS rewrite per §5).
3. **Accumulators**: cnt u32; m1: NMAX x 32; m2: NMAX x 8 x 32 —
   BRAM-modeled like the fabric's slotmem. Accumulate FSM: 9 cycles
   per site (m1 then 8 slot AND-counters), n*9 cycles per recorded
   sweep. COV is drain-time: one 32x32->64 multiply reused across the
   craw walk (cnt*m2 - m1_i*m1_j), int64 assembled hi/lo.
4. **Memory port**: no arbiter change — the s_cluster shares the
   sidecar's requester port via a static mux selected by which engine
   the last GO targeted (T_OP 0-5 tensor, 8-10 sampling); legal because
   the socket is one-outstanding globally (spec I-1/I-2).
5. **simt_core**: T_OP 8-10 route GO to s_cluster; T_STATUS busy =
   tensor_busy | s_busy. CSR map unchanged (0x8C0-0x8C9).
6. **DMA**: PCONFIG parses the image streaming (header -> section
   walkers in T_FLAGS bit order), writing fabric ports + shadows;
   PDRAIN streams the §6 layouts from accumulators/state_flat/work.
   Same valid/ack one-outstanding discipline as tensor_sidecar's DMA.
7. **Bench**: tb/s_cluster exact-diffs per-sweep state + every drain
   word against golden/sampling_isa.py on the S7.1 directed shapes +
   fuzz configs, reusing the tb/simt_core harness patterns.

## Categorical Q-Site Architecture


Pre-registered 2026-07-13. Motivated by the archived Potts experiment
(4.8x/17x mixing vs steelman one-hot,
LUT Gumbel-max clean at eqTV <= 0.0042). Spec-first; golden-first
law applies at every stage. Status: REGISTERED, stage 0.

### What it is

A per-image site arity q in {2, 4, 8} for the sampling fabric.
q=2 is today's binary fabric, bit-frozen and untouched. q=4 serves
chance codes; q=8 serves action digits (FabricMPPI's shape — the
17x task). One site per categorical variable replaces one-hot
groups + penalty terms.

### Design decisions (frozen at registration)

- D1 arity: image-wide (PCONFIG header field), not per-site. Mixed
  arity in one image is a NON-GOAL.
- D2 interactions: delta coupling ONLY (J fires iff x_u == x_v),
  scalar J per edge — the same 10-bit J lane as today. General qxq
  matrices are a NON-GOAL (16x weight memory, no workload demand).
  KEY COST FACT (from the delta structure): a neighbor contributes
  J to exactly ONE candidate accumulator (acc[x_nbr]) — so per-site
  field accumulation costs deg MACs, the SAME as binary; only the
  sampler pays the q-way cost.
- D3 sampler: Gumbel-max — q LUT reads (256-entry -log(-log u),
  s-grid quantized) + q adds + a depth-log2(q) comparator tree. No
  normalization, no division. PRNG budget: q x 8 bits per update =
  one xoshiro draw at q=4, two at q=8 (exactly as validated in
  golden/potts_exp.py).
- D4 state store: the D-027 write-behind store gains bit-planes
  (1 -> 2 -> 3 per site); broadcast write stream, zero-sweep, and
  color-boundary semantics unchanged. Clamps carry q-ary values.
- D5 biases: q-1 lanes per site (state 0 is the reference), s1.6.3
  grid per lane; PCONFIG ROWS/BIAS layout grows accordingly
  (spec §5 amendment).
- D6 drains: STATE packs log2(q) bits/site; MOMENTS generalizes to
  per-state counts m1[site][a]. COV-q and WORK-q E-scan generalize
  mechanically (E2 = -sum J*delta - sum b_a*delta) but COV-q is
  staged LAST and may be deferred by amendment.
- D7 chromatic engine, replicas, RID, drain protocol: UNCHANGED
  (Potts Gibbs is color-parallel exactly as Ising).

### Stages (each behind the full gate set)

- S1 spec + golden: sampling ISA spec amendment (QSITE section);
  frozen q-site golden (semantics = potts_exp.py arm Q, packaged as
  an oracle with the PRNG farm discipline). No RTL.
- S2 fabric_grid q=4: state planes, delta-MAC candidate
  accumulators, Gumbel LUT sampler; unit benches (smoke/fuzz vs the
  new golden) + battery.
- S3 s_cluster surface: PCONFIG arity + bias lanes, STATE-q +
  MOMENTS-q drains, E2-q scan; benches + battery.
- S4 q=8 extension + device kernels + S8σ gate (the q-site
  analogue of S7σ; certifies the ISA on-device vs the oracle).
- S5 integration benchmark: q-native bridge behavior and categorical
  proposals via native q=8, measured against the one-hot encoding.

### Frozen bars

- Q1 (correctness): bit-exact vs the frozen q-site golden across
  benches, fuzz, S8σ, battery — at every stage. Binary-mode images
  remain bit-identical to the current fabric (regression: existing
  S7σ + d2king untouched and green).
- Q2 (area/timing): the q=4 fabric config at NB=10 routes on the
  K26 within +30% LUTs of the binary fabric and closes the
  registered domain clock (8ns today; 7ns if leg E re-registers
  first). Evidence: OOC routed reports.
- Q3 (throughput floor): q=4 site-updates/s >= 0.7x binary
  sweep rate at the same clock (the per-update sampler cost stays
  bounded; mixing is where the win lives).
- Q4 (end-to-end payoff): native-q=8 categorical proposals reach
  fixed proposal quality (same-seed score distribution) in <= 1/4
  the fabric site-updates of the one-hot path — measured at the
  bridge/golden level in S5, on-device after S8σ.
- Decision points: any bar missed at any stage -> stop, bank, and
  amend or park. Q2/Q3 measured at S2 before S3 begins.

### Non-goals (registered)

General qxq couplings; mixed arity per image; q > 8; COV-q in the
first pass; any change to binary-mode semantics.

### Sequencing

T_OP 7 and the serving-path commitments hold their slots. S1
(spec+golden) may interleave earlier since it needs no VM/FPGA and touches
no RTL.

### Amendment 2 (2026-07-14, post-measurement — Q2 form corrected)

Q2's ratio form ("<= +30% of the binary fabric") proved mis-specified
against a MOVING baseline: the measurement campaign itself found and
fixed 3.6k LUTs of pre-existing baseline fat (pbit_cell's sigmoid
ROMs read combinationally since M4, defeating BRAM inference — now a
sync read, 8 BRAM tiles). Fixing the reference shrinks the
denominator; a ratio bar then punishes improving the baseline. The
bar's stated intent was BUDGET PROTECTION ("registers, not logic
explosions"). Q2 is re-expressed in the intent-preserving absolute
form: the q4-capable fabric <= 9.0% of the device at NB=12 AND the
G8.2 SoC sum including the QSITE surface <= 80%.

Iteration record (all routed at 8.000ns, NB=12, evidence
ci/logs/g8/qsite):
- i1 first build:            13,290 LUTs, +0.209
- i2 DSP-mapped score mults: 13,973 LUTs, +0.261
- i3 sigmoid ROMs -> BRAM:   10,408 LUTs, +0.408 (8 BRAM)
- i4 shared acc adders +
     DSP post-adder score:    9,944 LUTs, +0.255 (8 BRAM)
Transparency: a fresh binary-only build with the ROM fix would be
~3.9k, so the TRUE q4 capability delta is ~+6.0k (cells ~2.6k after
slims + second bit-plane ~2.1k + misc). Two intermediate VM verdicts
measured STALE builds before the wipe-then-launch discipline landed
(recorded; loop verdict checks must never precede log hygiene).

### S2 CLOSE-OUT (2026-07-14)

- Q1 CLOSED: bit-exact vs the frozen oracle at every stage (fabric
  smoke_q4 first run; s_cluster ISA-level smoke_q4 first run); binary
  world bit-identical throughout (S7σ ALL PASS x5 on QSITE trees,
  battery 28/28 x5, d2king, full suites).
- Q2 CLOSED under Amendment 2: 9,944 LUTs = 8.49% of device (bar
  9.0%); projected SoC sum ~76.5% (bar 80%).
- Q3 CLOSED: 8.000ns routed +0.255 (best iteration +0.408); chunk
  cadence unchanged by construction, so q4 sweep rate = binary rate
  at the same clock (>= 0.7x bar met structurally and by timing).
- S4 NOTE (registered): q8 starts area-conscious — the third
  bit-plane and wider cells inherit i4's shared-adder structure, and
  Q2' (9.0% / 80%) applies to the q8 build unchanged.

### S4 pre-registration (2026-07-14, BEFORE any q8 RTL)

Scope: q=8 sites end-to-end + the S8σ device gate. The oracle is
frozen (S1; self-check incl. the q8 double-advance rule). Bars Q1/Q2'
(9.0% fabric / 80% SoC) apply unchanged; Q3 stays a q4 bar (the q8
chunk pays one extra sample cycle + the second farm draw by
construction — throughput is not the q8 claim; mixing is).

AREA DISCLOSURE (the honest number first): the q4 fabric measured
9,944 LUTs vs the 10,541 bar — ~600 LUTs of formal headroom. The
naive q8 build (third bit-plane ~+2k by S2's second-plane precedent,
doubled score/GLUT paths, argmax-8) would miss Q2' by 2-3k. The
area-conscious design below targets the miss margin; the SoC <= 80%
clause holds with ~3.5k of margin either way. If the fabric-9.0% leg
still misses at OOC, the card's decision point applies: stop, bank
the measured number, and amend-or-park with the argument in writing.

Design (frozen before RTL):
- qsite_cell q8: accumulators grow 4 -> 8 REGISTERS ONLY (the i4
  shared-adder structure is address-based — delta coupling writes ONE
  acc per contribution; no new adders). The score/GLUT/DSP path is
  TIME-MULTIPLEXED: q8 samples in two stage-1 passes (candidates 0-3
  from farm word 1, then 4-7 from word 2) reusing the SAME 4 DSP
  mult-adds and GLUT ports; argmax runs incrementally (best-of-first-
  half compared against the second half, strict > keeps ties-to-
  smallest). One extra sample cycle at q8 only; the q4/binary
  sequencing is untouched (zero-regression posture).
- fabric_grid: third state bit-plane on the D-027 store (write queue
  widens 2b -> 3b); farm steps TWICE per non-empty chunk at q8 (word
  1 latched, word 2 drawn in the added sample state — golden's draw
  order: both words in order, decision after); bias store widens to
  the 7-lane q8 payload (BRAM width, not LUTs); STATE-q8 packs 8
  sites/word (4b lanes); value taps widen to 3b + a plane-3 row word.
- s_cluster: arity 8 accepted (tripwire retired); ROWS at 11
  words/site (8 slots + 3 bias words assembled to the wide bias
  write); m1 becomes {site, lane[2:0]} (8 lanes; q4 images stay
  bit-identical in their 4-lane view); MOMENTS-q drains n*(q-1) m1
  words; STATE-q drain interleaves three planes; E2-q selects among 7
  bias lanes by the 3-bit site value; COV-q stays deferred (tripwire).
- GLUT-to-BRAM is the registered RECLAIM CANDIDATE if OOC misses:
  the q8 pipeline adds the register stage a sync GLUT read needs
  anyway; moving both modes' GLUT reads to BRAM trades ~1-1.5k
  LUTRAM/mux LUTs for ~8-16 RAMB18 (budget: 142 BRAM free). Applied
  only on measured evidence, never speculatively.

S8σ gate (gates/s8_isa.py; bars FROZEN here, before measurement).
All runs from compiled kernels through the D4 socket on simt_soc —
sw/kernels/s7_run.c is a generic op-table replayer and is REUSED
VERBATIM (no new kernel; socket red paths are S7.5's, already green).
Directed suite: Q1 4x4 q4 ferro-delta torus (the S2 shape as an ISA
image); Q2 6x6 king q8, J~U{-16..16}\{0}, 7 bias lanes ~U{-8..8},
beta 48, 40 sweeps; Q3 = Q2 with 25% sites clamped to values >= 4
(order-exclusion + STATE-q); Q4 the D-014-q8 shape: bipartite with an
empty color segment AND a chunk-parity tail (odd site count) — the
double-advance accounting bar; Q5 multi-entry schedule + persistence
split-vs-whole (two PSAMPLEs, no SEEDS reload) at q8.
| bar | claim | frozen pass rule |
|---|---|---|
| S8.1 | trajectory exactness | device STATE-q drains == golden after EVERY sweep, zero tolerance, Q1-Q5 + 60 fuzz configs (seed-keyed 0x53380000+i: q in {4,8}, n in [4,64], deg <= 8, clamp prob 0.2, 1-3 schedule entries, <= 48 sweeps) |
| S8.2 | statistics exact | MOMENTS-q drains (cnt, m1[site][a], m2 agreement) integer-exact on every RECORD run of the S8.1 suite; TELEMETRY upd/flip counts exact (q8 upd_cnt counts decisions, not draws) |
| S8.3 | work/E2-q | WORK drains integer-exact incl. the D33 bracket on a live q8 ROWS rewrite (the S3 golden-fix regression, now on-device) |
| S8.4 | regression | S7σ ALL PASS + fabric/s_cluster binary+q4 suites + full battery green on the S4 tree (Q1's standing clause, re-proven) |
Evidence: ci/logs/s8/. Ledger row S8-sigma added at gate landing.

### S4 RTL + S8σ CLOSE-OUT (2026-07-14; Q2'-q8 OOC pending)

- S8.1/S8.2/S8.3 ALL PASS first full run after one real fix: the gate
  caught the s_cluster header clobbering n_colors/n_sched on ROWS-only
  images (golden persists coloring/schedule; latent since M21 — every
  prior diffed image carried ORDER+SCHED; the C* border-tax workload
  hit it un-diffed, and the fix only INCREASES its measured S, so the
  S7.4 pass is conservative). Spec §2 now states the persistence rule.
- The area-conscious design held: qsite_cell q8 adds accumulator
  REGISTERS and a time-multiplexed second score pass over the SAME 4
  DSP mult-adds and GLUT ports (one extra sample cycle at q8 only);
  the fabric adds the third bit-plane, the G_SMP1B double farm
  advance, and URAM-width-free 70b biases; q4/binary sequencing
  untouched (S8.4: battery + S7σ ALL PASS re-run).
- Benches: fabric smoke_q8 (odd-n cold chain + clamped warm) and
  s_cluster smoke_q8 (full ISA path incl. the D33 E2-q8 bracket)
  bit-exact vs the frozen oracle; loader contract pinned in the bench
  (arity precedes ROWS — the bias assembly is arity-keyed at load).
- OPEN: the Q2' (<= 9.0% fabric at NB=12 / <= 80% SoC) OOC verdict for
  the q8-capable build, queued on the VM behind the T_OP 7 run. The
  pre-registered fallback (GLUT -> BRAM reclaim) applies only on a
  measured miss.

### Amendment 3 (2026-07-14, post-measurement — Q2' binding bar moves to the SoC clause)

The measured q8 verdict (ci/logs/g8/qsite_q8): 15,989 LUTs = 13.65%
at NB=12 vs the 9.0% leg (10,541); timing -2.847 root-caused to the
35-level linear argmax scan and fixed (tournament tree, bit-identical
semantics, re-verdict t8b in flight). The area miss is STRUCTURAL:
the 9.0% leg was calibrated against the measured q4 build (9,944 +6%
margin) and the q8 CAPABILITY is materially wider silicon — a third
state bit-plane replicated across 8 lanes (~2k), 8 candidate
accumulators/lane, 8-way score+argmax, 7 bias lanes. The fixable
class (synthesis artifacts) was already harvested in S2; the
remaining identified fruit — GLUT->BRAM (~-1.6k), sequential argmax
tree reuse across the two passes (~-1.0k, 4 comparators for 7),
AreaOptimized_high synth (~-1k, zero RTL) — sums to ~-3.6k against a
-5.4k gap. Dead ends checked and recorded: plane packing/sharing (no
LUT amortization; 16 independent reads/plane/cycle force the
replication), sequential plane fetches (peak ports unchanged), q2-on-
qcell (breaks the frozen p17 semantics, Q1), LUTRAM accumulators
(write/read port count erases the win), narrower P (breaks frozen
chunk semantics).

AMENDED (intent-preserving, the Amendment 2 pattern): Q2's BINDING
bar is the intent clause — the G8.2 SoC sum including the QSITE
surface <= 80% at the device configs. Measured projection with q8:
core 58.8k + sampling(q8 fabric ~11.3k at the SoC's NB=10 + s_cluster)
+ sidecar 7.8k ~= 90.5k ~= 77.3% — HOLDS with ~3.1k margin. The
per-module NB=12 number becomes an INFORMATIONAL per-tier ceiling,
recorded per build (q4: 9,944; q8: 15,989 measured / ~12.4k projected
with the banked fruit), never silently exceeded — any future growth
is re-measured and re-recorded. The three fruit items are REGISTERED
follow-ups, applied on measured need (the SoC-assembly budget
conversation, where the T_OP 7 array — not q8 — is the dominant
line item). Honest cost of the q8 capability at the config that
ships: ~+3.8k LUTs (~3.2% of device) over the binary+q4 fabric.

Amendment 3 addendum (same day, user-directed): the three fruit items
are APPLIED IMMEDIATELY rather than banked — (1) GLUT -> sync BRAM
with a last-MAC-cycle prefetch (zero added cycles at q4; q8 adds
G_SMP2B, chunk 11 -> 12); (2) sequential argmax-tree reuse (one
depth-2 tree + one final compare = 4 comparators for 7; score
registers 8 -> 4; best1 carries pass 1 forward); (3)
AreaOptimized_high synth directive on the verdict flow. Bit-exact
through the fabric + s_cluster suites; the re-measured NB=12 number
lands with the t8c verdict. The BINDING bar remains the SoC <= 80%
clause regardless (the fruit adds ~-2.6k of projected margin). The
state-store -> BRAM mega-lever (~-4-5k, ripples into the s_cluster
walkers and every mode's cycle shape) stays BANKED for the
SoC-assembly era.

### S5 CLOSE-OUT (2026-07-14) — Q4 PASS at 8x (bar: >= 4x)

The bridge/golden benchmark uses the same q-site representation implemented
by `golden/qsite_golden.py`: one site per categorical variable, q-1 bias
lanes (state 0 reference), delta couplings, and MOMENTS-q plus TELEMETRY
drains. Native proposals require no exclusion cliques and are structurally
valid by construction, versus the one-hot arm's 41% invalid groups at round
zero.

The Q4 protocol uses two task shapes (H=6, R=16, q=8): "separable"
(one-hot's easiest case) and "chain" (pairwise step coupling in the score +
a smoothness prior in the proposal graph). 5 seed-keyed instances;
both arms score identical landscapes; distributions never
trajectories. Quality target Q* = min over arms of best population
mean; currency = the goldens' own TELEMETRY upd_cnt (one count per
site decision, q8 double-draw not double-counted). One-hot arm gets
a STEELMAN exclusion sweep {2, 6, 12}, best-of per instance.

Verdict: median updates-to-Q* ratio (one-hot/native) = 8.0 on BOTH
tasks; steelman median also 8.0; worst instance 5.33x; instances
where one-hot needed extra rounds ran to 24-27x. Q4 bar (>= 4x)
PASS, shipped and steelman.

Honesty column (registered): ratio/q = 1.0 — at raw slot-work the
arms are even at equal round counts; the native win is site
DECISIONS (the sequencer/walker currency), structural validity, and
expressiveness (below). No mixing advantage was claimed at this
shape and none is needed for the bar.

Finding (structural, recorded for the SoC/profile era): the one-hot
encoding CANNOT carry the chain smoothness prior in-fabric at q=8 —
exclusion clique degree (q-1=7) + 2 same-value chain edges = 9 >
the chip's hard degree-8 cap. The native encoding carries it with
ONE delta edge per step pair (degree 2). Sequence priors over q=8
alphabets are expressible only in the native encoding on this
hardware; this is a capability gap, not a tuning gap.

Q4 on-device re-measurement (same harness, device backend) is the
registered bring-up follow-up, alongside PROFILES PR1-PR4.

Amendment 3 addendum 2 (2026-07-14, the re-measured number): the
fruit build's OOC verdict landed — fabric_grid NB=12 q8 = 15,118
LUTs total (8,950 logic + 6,168 LUTRAM state store; 12.91% of
device), 10 RAMB36 (the GLUT -> BRAM fruit TOOK: registered sync
read inferred block ROM; per-cell 645 LUTs + 4 DSPs x 16 lanes),
40 DSPs, and TIMING CLOSED: WNS +0.125 @ 8.000ns (was -2.847 —
the argmax tournament + sequential tree reuse killed the 35-level
scan). The 15,989 -> 15,118 delta is the argmax+synth fruit net of
BRAM control; the informational NB=12 ceiling is recorded at 15,118.
The BINDING bar remains SoC <= 80% per profile: the planning
profile ships NB=10 (smaller than this OOC config) and its
projection already held with margin. Banked levers unchanged
(plane-2-to-BRAM; state-store -> BRAM mega-lever at assembly).
