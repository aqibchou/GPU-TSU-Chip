# GPU–TSU FPGA Implementation

This document consolidates the implemented FPGA architecture, Kria K26
resource and timing requirements, SG0 bridge, personality builds, and the
hardware optimizations that made the integrated design fit and close timing.

## Contents

- [FPGA Personalities](#fpga-personalities)
- [Timing and Resource Requirements](#timing-and-resource-requirements)
- [SG0 FPGA Bridge](#sg0-fpga-bridge)
- [Tensor Pipeline Optimization](#tensor-pipeline-optimization)
- [Fast-Path Optimizations](#fast-path-optimizations)
- [In-Sweep Accumulation](#in-sweep-accumulation)


## FPGA Personalities


Pre-registered 2026-07-14 and built for full-SoC assembly.

### The idea

Mk-I ships as N bitstreams ("profiles"), one per workload, instead of
one union SoC. The Kria K26 full-reconfigures in ~seconds via the SOM
FPGA manager; serving, sampling, and training are session-level modes, so a
seconds-class swap is free at the product level. The
looming assembly-era budget fight (G8.2's 80% clause vs the union of
engines) DISSOLVES: the budget only has to close PER WORKLOAD, not
for the union — strictly more powerful than any LUT diet.

### The measured arithmetic that motivates it (2026-07-14 numbers)

Union projection: core 58.8k + sampling(q8) ~22.6k + sidecar 7.8k +
verify array ~22k (post-iteration-1) + glue >> 80% — over, even
before glue. Per profile:
- PROFILE S (serving, SV2/SV3): core + sidecar + tensor_array
  (T_OP 0-7). NO sampling domain (SV3 never issues a sampling op).
  ~89k + glue ~= 76-78%. FITS.
- PROFILE P (sampling): core + sidecar + q8-capable sampling domain
  (s_cluster + fabric). NO tensor_array. ~89k + glue ~= 76%. FITS.
- PROFILE T (on-chip training): identical to P.
- Future: the LATTICE card's FIELD co-processor, if L2 passes, is
  simply another profile; the concept generalizes.

### Why this is architecturally clean here

- STATE ACROSS SWAPS IS ALREADY AN ISA FEATURE: the sampling domain's
  full state (params, persistent chains, accumulators, work) round-
  trips through DRAM via PDRAIN/PCONFIG — the drains ARE the
  checkpoint mechanism. D22's persistent-chain story survives
  reconfiguration by construction; nothing new is invented.
- The D4 socket is the stable seam: profiles differ only in WHICH
  engines answer the socket; the core, memory face, runtime, and
  kernel ABI are identical bits in every profile.

### Frozen bars (move only by written amendment)

- PR1 (per-profile fit): every SHIPPED profile closes post-route at
  its domain clocks with SoC sum <= 80% on xck26, and its gate set is
  green ON THAT BITSTREAM (S: G1-G5/G6/G7 + SV2 legs; P/T: S7σ and
  S8σ). The ledger records gate -> profile.
- PR2 (swap correctness): a drain -> swap -> reload -> continue
  round trip is BIT-EXACT vs an unswapped golden trajectory (an
  S7σ-style equality run spanning a reconfig; farm state re-seeded
  via SEEDS per §7 — determinism contract §7/I-7 holds across the
  swap).
- PR3 (swap cost, σ then measured): bitstream-swap wall time archived
  (σ at sim/spec time; measured at bring-up). No threshold — the
  claim is "session-level", and the number is printed.
  **σ WRITTEN 2026-07-17**: the K26 full bitstream is ~19.3 MB; raw
  PCAP programming at the Zynq US+'s ~150–250 MB/s ⇒ ~80–150 ms; the
  Kria userland path (xmutil/fpgautil: firmware move + device-tree
  overlay + driver rebind) historically lands ~1–3 s end-to-end.
  State drain+reload around the swap is DRAM-resident and µs–ms
  (PR2-measured mechanics), i.e. negligible beside the load. σ:
  **swap ≈ 1–3 s wall, so personalities are SESSION-GRANULAR (T by
  night / P by day / per-workload), never per-request** — consistent
  with how PROFILES was pitched. Bring-up measures xmutil wall time
  and re-runs the PR2 procedure on silicon; if the userland path
  dominates, raw fpga_manager loads (~150 ms class) are the known
  fallback, and DFX partial reconfig stays out of scope (union
  personality already covers the fast-switch need).
- PR4 (op-absence contract, spec-first): GO targeting an engine
  absent from the running profile traps deterministically (the
  mcause-2 family), and the profile is kernel-discoverable (a
  read-only profile-id field; ISA amendment WRITTEN BEFORE any RTL).

### Non-goals (registered)

- DFX/partial reconfiguration in v1 — registered as the v2 refinement
  with its caveat named: the reconfigurable partition must span the
  COLUMN resources of the larger occupant (>= 1024 DSPs for the array
  AND >= 48 URAMs for the fabric), a floorplanning problem full-
  bitstream swap simply does not have.
- More than the S/P(T) profiles in v1; per-profile ISA forks beyond
  the op-absence contract; any change to frozen engine contracts.

### Sequencing

ASSEMBLY OPENED 2026-07-16. **PR4 spec amendment: WRITTEN (tensor_spec
§1c / D-036, spec-first per the bar)** — T_PROFILE @0x8CA read-only,
op-absence traps mcause 2 at GO, presence masks frozen (union builds
= id 0, today's certified behavior unchanged). VM restarted + OOC re-hearings RUNNING
(s_cluster @8/@7ns, q8 fabric @7ns). **PROFILE RTL + PER-PROFILE
BATTERIES: DONE 2026-07-16 ~13:30 (main @ 0a2b9cd)** — one
parameterized simt_soc (PROFILE generic, generate-guarded sampler),
T_PROFILE + absence trap live per §1c, and the gate matrix EXECUTED
at sim level: UNION bit-exact (incl. PSTAT via the new scope), S =
PR4 + G15 green with NO sampler built, P = PR4 + S7σ + S8σ bit-exact.
PR4's red path is a permanent gate
(gates/pr4_absence.py). **PR2 GATE GREEN 2026-07-16 ~14:45
(main @ 0689468)**: drain → SWAP (union → build_p2, binaries differ
by hash) → reload → continue is BIT-EXACT — 138/138 checkpoints (109
pre + 29 post swap) vs the unswapped golden. The drain contract is
now explicit and machine-checked: the kernel's exit write-back puts
every live cursor in DRAM (noise xoshiro → d[27..30], NEG chains →
d[60..75]); the runtime's reload obligations are scrub the pool
mailbox, patch the farm SEEDS with the host-recomputed cursor (PL
state, deterministic advance), set the resume window. Permanent gate
gates/pr2_swap.py; evidence ci/logs/pr2/pr2_swap.json. **PR1 EXECUTED
2026-07-18 (first full-SoC hearings ever): BAR FAILED AS MEASURED —
union 99.05% LUT, routed, WNS −12.288; S 82.06% LUT, routed, WNS
−10.956 (just over the ≤80% bar); P 100.57% LUT at synth, PLACER
FAILED (over capacity; placer seed luck separated it from union).
ROOT CAUSE localized: both worst paths are the core's reconvergence
STACK (stk_pend/stk_top → stk_rcv), 42–54 logic levels, ~19–20ns
data path — a 3-D FF-array + mux forest; barrel_sched alone is 40k
cells (24% of the chip). The core is the one block that never had
an OOC hearing (Verilator is timing-blind) — PR1 existed to catch
exactly this. Cell census: u_core 105k cells (u_sched 40k),
u_sampler 52.5k, u_tensor 8.5k; DSPs 4.6%, BRAM 30%, URAM 37% —
LUTs are the sole binding resource. DECISION DATA BANKED: C4
concurrency has zero headroom in the union until the core diet
lands; S3 fabric scaling is per-profile-only. NEXT: the CORE-DIET
sitting (same playbook as s_cluster ×2: RAM-ify the per-warp
reconvergence stacks — push/pop is inherently 1R1W — and pipeline
stack updates across the barrel rotation, which grants up to 8
ticks of architectural slack per warp by construction; values
bit-exact, battery verifies), then PR1 re-hearing, then bitstreams.
Evidence ci/logs/profiles/pr1/. VM stopped (evidence pulled first).**
**PR1 RE-HEARING (dieted core, 2026-07-18): FIT BAR MET
EVERYWHERE** — union 51.92% LUT (was 99.05), S 30.70% (was 82.06),
P 50.78% (was unplaceable at 100.57); ≤80% clears by ~30 points on
the union, so C4 concurrency and S3 fabric scaling are UN-PARKED
with real headroom (~28 LUT points). Timing at 8ns, FINAL —
**ALL THREE PROFILES MET: union +0.060 (Explore+phys_opt closure),
S +0.153, P +0.045. PR1 IS FULLY GREEN: every personality fits
≤80% AND closes 8ns, routed.** Closure evidence
ci/logs/profiles/pr1/pr1c_p{0,2}_route*.rpt. The closure directives
(place/route Explore + phys_opt AggressiveExplore) become the
default flow for bitstream builds. Infra note: four spot
preemptions in ~36h forced the runs onto mk-ondemand (on-demand
n1-std-8 cloned from the worker disk) — bitstream builds live
there. PR3 σ written (above); REMAINING TO SG0: the bitstream
sitting (bd wrapper around the certified sg0_bridge + PS, device
tree carveout, MK_TRANSPORT=uio) → per-profile bitstreams → board
day. Interaction with QSITE Amendment 3: the SoC <= 80%
binding clause is evaluated PER PROFILE from that sitting onward
(the amendment argument to write there, with routed numbers).

### Build definitions + gate matrix (frozen 2026-07-16, assembly sitting 1)

One parameterized `simt_soc` (PROFILE generic per tensor_spec §1c),
NOT per-profile forks: generate-guards instantiate only the profile's
engines; absent engines are not built and their T_OPs trap at GO per
the frozen masks. Identical core, memory face, runtime, kernel ABI in
every profile (the D4-socket seam).

| profile | id | engines built | T_OPs live | gate set (PR1, on THAT build) |
|---|---|---|---|---|
| UNION (dev/sim) | 0 | all | all | the full battery (today's, unchanged) |
| S (serving) | 1 | core + sidecar + tensor_array | 0–7 | G1–G5, G6, G7, SV2 legs, PR4 red path |
| P (sampling) | 2 | core + sidecar + s_cluster+fabric (q8) | 0–6, 8–10 | S7σ, S8σ, PR4 red path |
| T (training) | 3 | = P | = P | P's set |

Device configs ride ooc.tcl's frozen generics: s_cluster NB=10
(n ≤ 1024 — the C* workload size), fabric NB=12 under evaluation;
NB=13 stays the sim/architectural limit (D-022/D-023). RTL scope for
the next sitting: the PROFILE generic + generate guards in simt_soc,
the §1c id/mask/comparator in the T-CSR block, tie-offs for absent
engine ports — then per-profile batteries, then PR1 routed fit on
the VM (whose numbers decide S3 fabric scaling and the C4
concurrency posture).

## Timing and Resource Requirements


Out-of-context synthesis + implementation timing for every critical
module, on the real Kria device model (`xck26-sfvc784-2LV-c`, Vivado
2024.2, free tier), run batch-mode on the DP-2 Linux host. The timing
report is the gate; the M15+ Yosys proxies were the leading indicator.

### Frozen parameters

- Target core clock: **100 MHz** (the documented design assumption —
  mem_spec §"base latency" derives from it).
- OOC constraint: **125 MHz** (1.25× target, period 8.000 ns) on every
  clock pin. Margin absorbs OOC-vs-in-
  context estimation error.
- Mode: `synth_design -mode out_of_context` + `opt_design` +
  `place_design` + `route_design`; the POST-ROUTE WNS is the number.
- Part: xck26-sfvc784-2LV-c, default speed grade as shipped free-tier.

### Module list (each gated independently)

| module | top | clock |
|---|---|---|
| SIMT core (8×8) | simt_core | clk |
| Barrel scheduler | barrel_sched | clk |
| SIMT register file | simt_regfile | clk |
| Tensor sidecar (D4 socket) | tensor_sidecar | clk |
| Memory-port arbiter | port_arbiter | clk |
| D-cache (M16) | dcache | clk |
| P-bit fabric grid | fabric_grid | clk |
| P-bit PRNG farm | prng_farm | clk |

(The s_cluster joins this table when M21's RTL lands — same bar.)

### Bars (FROZEN before any run)

- **G8.1 timing**: post-route WNS ≥ 0 ns at the 8.000 ns constraint for
  EVERY module in the table. A miss is a FAIL for that module; fixes
  are RTL work re-verified by the full bench battery, never a bar move.
- **G8.2 utilization sanity**: post-route LUT, FF, BRAM, DSP counts
  archived per module; the SUM across the table must fit xck26
  (117,120 LUT / 234,240 FF / 144 BRAM36 / 1,248 DSP) with ≥ 20%
  headroom on every resource class (the SoC integrates all of them
  plus glue). Informational per-module; the headroom check is the bar.
- **G8.3 methodology checks**: no critical warnings from
  `check_timing` (unconstrained paths, no_clock) in any run.

Evidence: `ci/logs/g8/<module>_route.rpt` (+ util reports) pulled from
the host; `gates/g8_ooc.py` parses and verdicts. σ-scope: this is the
OOC proxy gate — in-context SoC timing on the real board is the
Bring-up Campaign's problem (H9), not M20's.

Amendment discipline: bars move only via a D-entry (R3).

## SG0 FPGA Bridge


STATUS: registered decisions + bar CANDIDATES. This is the spec
skeleton for the last unexercised engineering layer (PS–PL
integration + host transport). Bars freeze at this spec's own
sitting (R3) before any bridge RTL is written.

### Shape

Vivado block design = Zynq US+ PS + simt_soc. **Zero PL pins**: the
SoC needs no physical I/O — clock and reset from the PS, commands
and memory over AXI. Pin/constraint work is therefore trivial; the
whole integration is two AXI surfaces and a device-tree file.

### Decisions (enumerated for the freeze sitting)

1. **Command surface**: everything soc_harness.cpp does over the
   sim socket (RESET / RUN n / DONE & ASSERT peeks / PSTAT / SCHED)
   becomes an AXI-Lite register file on PS M_AXI_HPM0 via a thin
   bridge module. One register map, documented here at freeze.
2. **Memory**: the SoC's memory face masters S_AXI_HP into the PS
   DDR controller — device DRAM becomes a reserved-memory carveout
   of the 4 GB LPDDR4 (proposal: 2 GB carveout; device-tree
   reserved-memory node). This is the same physics PR2's drain
   contract already models: PL state dies at reconfig, the carveout
   persists.
3. **Host transport**: mkcuda.Runtime grows a second transport
   (MK_TRANSPORT=uio): _cmd → mmap'd AXI-Lite registers; read/write
   → mmap'd carveout. THE PYTHON API DOES NOT CHANGE — every gate
   runs on silicon unmodified. This is the bring-up thesis: the
   gates ARE the bring-up suite.
4. **Clock/reset**: PL0 = 125 MHz from the PS (the certified 8 ns);
   reset from the PS reset controller. No PL-side PLL work at SG0.
5. **Port width**: SG0 ships the current 32-bit face (known ~0.4
   GB/s ceiling, disclosed — SG0 is bring-up, not serving
   performance; the SV2 512-bit face over aggregated HP ports is
   its own later leg).
6. **Boot/swap**: per-profile full bitstreams via xmutil loadapp;
   PR3 σ (profiles_card) measured here.

### Port-contract findings (2026-07-18, from the SoC face — these
### harden decisions 1-3 above)

- **imem is BRAM, not DRAM**: the core's imem contract is
  always-ready with rdata the NEXT cycle — incompatible with
  variable-latency AXI. On silicon imem = a dedicated preloaded
  BRAM (128 KB proposal; kernel images are KB-scale), loaded
  through an AXI-Lite window at launch. THE CORE IS UNTOUCHED — the
  1-cycle contract holds exactly as in sim.
- **dmem is the AXI master**: the dmem face already speaks
  req/ack with variable latency (the SoC was built latency-tolerant
  here), so the adapter is a small FSM: one outstanding beat,
  req→AR/AW+W, ack on R/B — correctness identical, bandwidth the
  known 32-bit-face ceiling (disclosed; SV2's wide face is the
  later leg).
- **The control surface shrinks**: on silicon the clock free-runs,
  and DONE/ASSERT/LOG/PARAMS all live in the DRAM carveout the host
  mmaps directly — so the AXI-Lite register file is just
  {SOC_RESET, RUN_EN, MCYCLE snapshot, STATUS, IMEM load window}.
  "RUN n" cycle-stepping is a sim-only concept; silicon runs free
  and the host polls DONE words in memory, exactly like
  launch_progress already does.
- Bridge module plan: rtl/soc/sg0_bridge.sv = AXI-Lite slave
  (control regs + imem write window) + imem BRAM + dmem→AXI4
  master FSM; cocotb TB against an AXI memory model (no
  soc_harness collision — battery-independent).

### As-built (2026-07-18): rtl/sg0_bridge.sv — unit-certified

Register map (AXI-Lite, word regs): 0x00 CTRL {bit0 run — gates
soc_rst_n; "reset" = run 0→1}; 0x04 STATUS {magic 0x05D0, version,
imem size/8K}; 0x08 MCYCLE_LO (read latches HI); 0x0C MCYCLE_HI;
0x10 IMEM_ADDR (word index, readback supported); 0x14 IMEM_DATA
(write → imem[IMEM_ADDR++]). dmem: 8-deep FIFO absorbing the credit
face's pulses → one-outstanding AXI4 FSM (AR/R, AW+W/B), in-order
acks, CARVE_BASE added to every address. Bench
tb/sg0_bridge/test_sg0_bridge.py: directed (magic, run/MCYCLE
gating, imem window incl. the 1-cycle read contract, 8-credit
burst, ordered reads, byte strobes) + randomized soak vs a python
mirror over a latency-randomizing AXI slave model. SMOKE PASS.
Remaining to SG0 hardware: the block-design wrapper (PS + bridge +
simt_soc), device tree, MK_TRANSPORT=uio in mkcuda, bitstreams.

### Bar candidates (to freeze at the sitting)

- SG0.1 boot + echo: bitstream loads, AXI-Lite register echo
  round-trips from Python.
- SG0.2 memory: PL-side read/write of the carveout verified against
  host mmap (pattern + boundary tests).
- SG0.3 chip ISA/profile bar: S7σ and S8σ run unchanged on silicon
  and are bit-exact vs their goldens. Follow with the PR4 probe per
  profile (absence trap fires on silicon), PR2 procedure across a
  real xmutil swap.
- SG0.4 PR3 number printed (swap wall time), plus first
  silicon-vs-sim cycle-count comparison (informational, not a bar —
  D-026).

### Explicitly out of scope at SG0

SV2 wide face + tensor_array, FLASHDEC, C4 concurrency, any
performance claims beyond "the certified behavior reproduces on
silicon." SG0 proves the machine is real; the speed story stays
sim-certified until its own on-device re-measurements (registered
list rides in the ledger).

### Bitstream flow as-built (2026-07-18, ci/ooc/sg0_bitstream.tcl)

Scripted bd: zynq_ultra_ps_e (HPM0_FPD + HP0_FPD @32b, LPD masters
OFF — a default-on LPD port was iteration-2's dangling clock;
pl_clk0 @125MHz) + proc_sys_reset + two smartconnects + sg0_top as
a module reference (PLAIN VERILOG top — Vivado's module-ref refuses
SV tops, filemgmt 56-195, iteration 1) with PROFILE pushed through;
Performance_Explore strategy (the PR1-certified closure recipe).
**FIRST BITSTREAM: union sg0_p0.bit, WNS +0.135 @8ns, 53.40% LUT
(PS integration included) — three flow iterations to first light,
both failures config-class.** Packaging proven the same day:
bootgen → .bit.bin, dtc → sg0.dtbo, shell.json, tarball
(sg0-p0-app.tgz; artifacts in ci/logs/sg0/). Runner lesson: the
wipe-first runner deletes prior artifacts — PULL BITS BEFORE
RELAUNCHING (the laptop copy saved the union .bit when the S/P
launch wiped sg0_out); packaging lives in ~/sg0/pkg, outside the
runner's blast radius.

## Tensor Pipeline Optimization


Successor to the "TA 4-stage restructure" line item; scoped after the
D-030 verdicts (ci/logs/g8/tensor). Fit is closed (sidecar 7,773
LUTs, array 5,490); this card is timing only.

### Frozen bars (moves only by written amendment)

- B1: tensor_sidecar routed WNS >= 0 at 8.000 ns on xck26-sfvc784-2LV-c
  (OOC, AggressiveExplore, the D-030 flow unchanged).
- B2: tensor_array routed WNS >= 0 at 4.000 ns, same flow (SV2.3's
  250 MHz serving bar, registered in ci/ooc/ooc.tcl).
- B3: bit-exactness unchanged — G15 tensor gate PASS, S7σ ALL PASS,
  battery green. Golden results identical (op-level contract; cycle
  shape is not contract, D-026).
- B4: area stays sane — each top within 2x of its D-030 LUT count
  (pipelining adds registers, not logic explosions).

### Allowed moves

Pipeline registers and stage restructuring inside tensor_sidecar COMP
and tensor_array's MAC (valid/flag chains may lengthen); DSP mapping
attributes; retiming hints. NOT allowed: protocol changes on the
memory face, tile layout changes, result changes of any kind.

### Baselines (banked)

- sidecar @ 8ns: WNS -3.523 (COMP single-cycle cone: addr multiplies
  -> LUTRAM read -> rotate -> MAC -> 3-way 32b add).
- array @ 4ns: WNS -4.830 (x fetch + guard feeding the W4
  4-mult+3-add tree in one stage).

### Design

- Sidecar (leg 1): 3-stage COMP pipeline, II=1 — S0 registers the
  address multiplies (cursor runs ahead; c_done + drain empties the
  pipe into ST_C), S1 fetches/rotates A, B, and the C RMW window into
  registers, S2 multiplies (DSP-hinted), accumulates accv, and
  commits the C window on writeback beats. ~2 bubble cycles per COMP
  session; beat count otherwise unchanged.
- Array (leg 2): x-fetch stage (register x_eff and the weight beat
  before the multiply cone) + split the W4 product tree across two
  stages; downstream valid/last/addr chain lengthens to match.

### Amendment 1 (2026-07-13, pre-measurement of the change)

B2's "same flow" clause amended for the array leg only: synth
directive becomes PerformanceOptimized WITHOUT -retiming (route flow
unchanged: AggressiveExplore + post-route phys_opt). Evidence: tp4
and tp5 route to the identical 3.609ns DSP-internal path
(PREADD_DATA -> multiplier -> cascade ALU_OUT, M/P registers empty)
with one AND two explicit product register stages — -retiming
disassembles whatever pipeline the RTL states, and AreaOptimized_high
prefers the cascade packing that creates the 2-DSP chain. The RTL's
6-stage structure is the textbook M/P-absorption pattern; the flow
must be allowed to keep it. Sidecar leg verdict (+1.222, old flow)
stands unchanged.

### CLOSE-OUT (2026-07-13): CARD GREEN

- B1 CLOSED: tensor_sidecar +1.222 @ 8.000 (3-stage COMP pipeline;
  from -3.523). 8,114 LUTs (B4 ok).
- B2 CLOSED: tensor_array +0.010 @ 4.000 (from -4.830; seven
  iterations). 2.7k-LUT class (B4 ok). Winning sequence: 5-stage
  pure-product split -> row_base strength reduction (control multiply
  was the hidden #2 path) -> 6-stage double-registered products ->
  flow amendment (retiming disassembled stated pipelines) -> wsel_q:
  ALL operand selection at the fetch stage, S1 = pure reg x reg
  multiply. The decisive fix was the module's own line-77 lesson
  applied one stage earlier.
- B3 CLOSED throughout: G15 PASS at every iteration, S7σ ALL PASS,
  battery 28/28 on the final tree; array smoke/fuzz bit-exact x7.
- Evidence: ci/logs/g8/tensor_pipe/ (tp1/tp7 logs + final rpts).
- Lesson banked: when successive structural fixes leave the critical
  path BYTE-IDENTICAL, the tool is reversing them — read the path
  PRIMITIVES (PREADD_DATA = an operand-side subtract; C_DATA = an
  absorbed downstream add) before adding more registers.

## Fast-Path Optimizations


The connective-tissue program: the engines are closed (fit + timing);
remaining general-purpose speed lives between them. Four legs +
one banked freebie, independent, committed separately, each behind
the full gate set (relevant σ-gates + battery, bit-exact op results).

### Legs and frozen bars

- A (D-032a) barrel slot compaction: the scheduler skips harts parked
  on engine/memory waits and rotates only the ready set.
  BAR: all program-level goldens bit-identical (S7σ, G15, battery);
  measured scalar-phase cycles on the S7 kernel suite drop >= 2x when
  <= 8 harts are active. No ISA change; hart-visible semantics
  preserved (a hart never observes its own slot skipped while ready).
- B (D-032b) shared engine scratchpad: an on-chip SRAM window in the
  SoC address map, DMA-reachable by both engines and the core.
  BAR: goldens unchanged (the window is just memory to software);
  engine->engine handoff (drain -> GEMV input) round-trip measured
  >= 4x faster than the DDR-model path at SoC sim level.
- C (D-032c) credit-based memory face: in-order multi-outstanding
  (credit counter, no tags) on the one-outstanding valid/ack face;
  slave side + arbiter first, masters converted one at a time
  (drain streamer, LD_X/LD_A/LD_B, PCONFIG walker).
  BAR: word order preserved exactly (goldens bit-identical); DMA
  phase cycles on converted paths improve >= 2x at sim memory
  latency >= 4 cycles. Protocol default remains depth 1; depth is a
  parameter.
- D (D-032d) T-op command queue: depth-2..4 queue in the T-CSR block;
  ISA amendment to I-1/I-2 (one outstanding -> queue with FULL
  status bit); kernels may enqueue while busy.
  BAR: spec amendment written BEFORE RTL; golden updated first;
  S7σ/G15 green; serving-path op-chain latency (GEMV->LUT->GEMV)
  drops the inter-op core round-trip (measured at SoC sim).
- E (banked) 7ns re-registration of the 8ns engine domain once B2
  closes: OOC evidence first (both engines routed >= 0 at 7.000),
  then the D-entry re-registers the domain constraint.

### Order

A -> B -> C (infra, then per-master) -> D (spec-first) -> E (after
the D-031 array leg closes; VM currently occupied by it).

### Amendment 1 (2026-07-13, pre-measurement)

Order changes to B -> C -> A -> D. Reading barrel_sched.sv before
implementation showed leg A is an invariant-catalog amendment, not a
scheduler tweak: INV-2 (stage tags = pure function of phase) is the
never-skip property itself, and INV-4's hazard freedom is structural
(W > DEPTH) only BECAUSE rotation never skips. Compaction requires a
per-warp in-pipe interlock as the new INV-4 enforcement and retires
INV-2 in favor of tags-travel-with-slots. Legs B/C have no doctrine
surface and go first. Bars unchanged.

### Leg C design note (pre-implementation, 2026-07-13)

The current face is LEVEL-valid with pulse-ack, and masters deassert
valid one cycle after consuming ack — the arbiter's own NOTE records
the duplicate-beat race this creates for any same-cycle re-grant.
Full-rate streaming therefore needs an EDGE-request migration, not an
ack tweak: req is a one-cycle pulse per beat; a master may hold up to
CRED (2-4) beats in flight; responses return strictly in order
(rd_valid pulse + data, or implicit write completion credit); credits
replenish on completion. Slave/arbiter/masters must migrate together
per port — staged as its own sitting: arbiter + harness slave first
behind a CRED=1 compatibility mode proven identical to today's face,
then the drain streamer, then LD_X/LD_A/LD_B, then PCONFIG.
Measurement rides the existing harness LAT command (G1 knob,
soc_harness stdin protocol) — no new infrastructure needed.

### Amendment 2 (2026-07-14, pre-RTL): leg A catalog amendment written

Leg A's invariant-catalog amendment is now WRITTEN in
docs/HARDWARE_ARCHITECTURE.md#simt-core ("Invariant-catalog amendment (D-032a)"): INV-2
retired for INV-2' tags-travel-with-slots; INV-4 kept and enforced by
a per-warp in-pipe interlock (ready = !busy && !in_pipe); INV-1 a
corollary; INV-10 restated over the ready set (round-robin, no
pass-over). simt_core audit: the only phase consumer is the
scheduler's own issue decision, so the RTL surface is barrel_sched.sv
+ its assertions + tb/barrel_sched, in one commit, behind the full
battery. Bars unchanged.

### Leg A feasibility math (2026-07-14, pre-measurement — read before
### building; found while writing the catalog amendment)

Two facts the ≥2× scalar-phase bar depends on, worked out on paper:

1. **The interlock period bound.** With the in-pipe interlock, a lone
   ready warp's issue period is set by where in_pipe clears: at
   WB-EXIT the period is DEPTH=5 (8/5 = 1.6× ceiling at W=8 — BELOW
   the bar); at WB-ENTRY it is 4 (2.0× — knife-edge), and that
   variant needs the regfile-write-vs-D-read timing argument (WB
   writes commit at the posedge before the follower's D read) written
   into the amendment before use. Neither clears the bar with margin
   on pure-ALU stretches.
2. **The waiter problem.** Compaction only pays if the ready set
   actually shrinks — but done harts and mk_t_wait/T_STATUS pollers
   SPIN (loads park them only ~LAT cycles); at LAT=1 the ready set is
   ~W and compaction wins ~nothing on the S7 shapes. The measured win
   likely requires waiters to PARK: candidate = engine-wait parking
   (a T_STATUS read issued while the engine is busy parks the warp's
   busy bit until the engine goes idle — value semantics unchanged,
   the poll loop collapses into one stalled read), and/or a done-hart
   halt. Both are spec-surface changes beyond the catalog amendment
   and take their own pre-registered note.

Consequence: the leg-A sitting STARTS with a measurement pilot —
ready-set occupancy histogram + tid0 issue duty over the S7 scalar
phase (harness-side counters, no RTL) — and picks mechanisms from the
measured mix. The bar does not move; if compaction+parking cannot
reach 2× on the frozen workload, that is a banked miss with mechanism,
not a tuned bar.

### Leg D pre-registration (2026-07-15 — the spec-first sitting;
### RTL is the NEXT sitting)

SPEC AMENDMENT WRITTEN (tensor_spec §1b, the socket's home; sampling
spec §1 + I-1 amended by reference; CSR table updated): a
depth-QDEPTH command queue (parameter, frozen default 2) above both
engines. GO snapshots the eight staging CSRs into a queue record —
legal while busy, trap mcause 2 at FULL (the boundary moves from
busy to full, same fault code); T_STATUS bit0 BUSY = engine busy OR
queue nonempty (!busy semantics preserved VERBATIM — the completion/
visibility barrier every existing kernel polls), bit1 FULL (only
ever set while bit0 is set, so nonzero-polling stays correct);
dispatch feeds each engine from the QUEUE-HEAD record, never the
live staging registers (staging semantics now explicit: rewrite
freely after GO retires); strict program order across all T_OPs; I-2
execution serialization untouched — the queue pipelines ISSUE, it
adds no concurrency. GOLDEN IMPACT: none on values by construction
(goldens execute commands instantly in program order; the ISS
carries no socket model) — that disclosure is the golden-first step.
Red-path boundary flips; the s7_gowb kernel + S7.5 gate expectation
update ride the RTL commit.

Frozen sub-bars (the card's leg-D bar made concrete):
- D.1 (chain bar — the differential form, the stage-C2 lesson):
  serving chain alternating GEMV(64x1x64)/GELU(64), chain lengths
  n=3 vs n=1, gaps := [wall(3) - wall(1)] - [dTBUSY(3) - dTBUSY(1)]
  (boot, pre-first-op setup, and tail completion-detect cancel
  exactly; TBUSY removes the added ops' busy). BAR: gaps <= 8 cycles
  total (<= 4 per op pair) at LAT in {1, 4}. BASELINE measured
  2026-07-15 (instrument validity; ci/logs/fastpath/d_baseline.json;
  kernel sw/kernels/fastpath_d_chain.c): 1,560 cycles @ LAT=1
  (780/pair) and 1,886 @ LAT=4 (943/pair) — the statistic separates
  from the bar by >100x. Gap composition deliberately not decomposed
  (poll cadence x barrel duty suspected; the queue removes it
  wholesale either way; decompose only if the post-RTL residual
  exceeds the bar).
- D.2 (values): G15 + S7σ + the full battery, bit-identical.
- D.3 (red paths, directed): GO at FULL traps mcause 2; GO while
  busy-below-FULL does NOT trap and the enqueued command executes
  bit-exactly.

RTL sketch for the next sitting (bounded): queue storage + head
dispatch mux + per-dispatch t_go pulse in the simt_core T-CSR block;
the trap condition at simt_core's GO-write site moves from t_busy to
q_full; core-level directed tb + the S7.5 flip; then the battery and
D.1.

#### D.1 instrument note (2026-07-15, first-contact — bar unchanged)

Two measurement corrections, both caught by the gate's own controls:
(1) the 3-vs-1 chain form carries +-16..45 cycles of rotation-phase
quantization per boundary (the detect/stamp alignment walks with
warp0's 1/8 slot; a 2-boundary differential even produced a NEGATIVE
gap once) — the chain lengthened to 21-vs-1 so 20 boundaries amortize
the jitter to +-1.5; (2) the gate's first divisor used n/2 "pairs"
instead of the correct n-1 BOUNDARIES, doubling every per-boundary
number — flagged immediately by the polled control reading 2.06x
against its banked baseline (real value 1.03x). The frozen <= 4
cycles/boundary bound never moved.

### LEG D: VERDICT (2026-07-15) — RTL landed, bar met

Queue implemented per §1b exactly (QDEPTH=2 records in the simt_core
T-CSR block; staging ts_* CSRs snapshotted at GO; dispatch loads the
engine ports from the HEAD RECORD with go_pend covering the
dispatch-to-busy window so T_STATUS never blinks; trap moved to
FULL). D.1: queued inter-op gap 2.3 cyc/boundary @ LAT=1 and 1.9 @
LAT=4 (bar <= 4) vs the banked polled baseline 780/943 — the
serving-path core round-trip is GONE (~350-500x per transition).
CTRL: the polled discipline re-measured 1.03x baseline both LATs
(the queue leaves the old path's timing shape untouched). D.3b:
queued-vs-polled outputs byte-identical. D.3a (GO at FULL traps,
enqueue-while-busy doesn't) rides the flipped s7_gowb in S7.5.
mk_t_full()/mk_t_post() added to mk.h. Evidence
ci/logs/fastpath/d_bar.json.

BATTERY VERDICT (2026-07-15): ALL PASS — lint, 22/22 units, G15,
S7σ (including the FLIPPED S7.5 red path: GO at FULL trapped, the
two legal enqueues did not), S8σ, and
the D bar re-confirmed. One worktree-data fallback rode the re-run
(golden/text8.py — datasets/ is main-tree, the mkcuda _MAIN split).
LEG D IS CLOSED. FASTPATH standing: B closed, C closed, A banked,
D closed; E stays banked-conditional (fabric 7ns re-registration —
the OOC precondition never met). The connective-tissue program is
COMPLETE; next is full-SoC assembly (PROFILES).

### Leg A measurement pilot registration (2026-07-15 — written BEFORE
### the pilot runs; a mechanism-selection measurement, NOT a bar)

Per the feasibility note, the leg-A sitting opens by measuring, not
building. Instrumentation is sim-only and logic-free: verilator
public-read annotations on three EXISTING signals (scheduler busy
vector, rotation phase, per-warp PCs) sampled by the soc harness
behind new stdin commands (SCHEDCLR / SCHEDCLASS / SCHED; sampling is
off unless armed, so certs pay nothing). Per cycle the sampler
accumulates: ready-set-size histogram, per-warp ready/issue counts,
the warp0-ready x ready-count joint, and ready-warp composition by PC
CLASS — classes are address ranges: the crt0 done-park self-jumps and
the engine-poll (csrr T_STATUS) loops, both located from the kernel
disassembly; everything else = worker.

Frozen workloads, LAT in {1, 4}: (a) fastpath_a_scalar.c — tid0 runs
a pure integer loop, every other hart returns immediately and
done-spins (the worst-case serial-update shape);
(b) the C* S7.4 shape (mixed ISA); (c) fastpath_gemm (engine-wait
shape). Report per workload: measured warp0 issue duty; predicted
duty under {compaction only; + engine-wait parking; + done-halt;
both}, each capped by the two interlock variants (issue period 5 =
in_pipe clears at WB-exit; period 4 = WB-entry); the implied
scalar-phase speedup against the leg's frozen >= 2x bar.

Decision rule: mechanisms whose predicted speedup on workload (a)
clears 2x with margin get spec'd next (each its own pre-registered
note: the catalog amendment already covers compaction; parking/halt
are new spec surface). If none clear, the leg banks per the card. The
bar does not move either way.

### LEG A: BANKED (2026-07-15 — pilot verdict, mechanism in full)

Pilot ran as registered (evidence ci/logs/fastpath/a_pilot.json);
none of the buildable mechanisms clears the bar with margin:

1. COMPACTION ALONE IS WORTH 0.96x (i.e., nothing). Measured
   ready-set mean 7.7-7.8 of 8 on every workload: done harts spin in
   crt0's park (6.9-7.0 of the 7 non-w0 warps, all workloads) and are
   always ready, so there is no ready-set to compact.
2. ENGINE-WAIT PARKING IS WORTH NOTHING: measured poll fraction 0.00
   everywhere. mk_t_wait is a ONE-SHOT busy check, not a barrier —
   non-issuing harts race past it while the engine is idle, store
   DONE, and park; the only hart that ever polls T_STATUS during an
   op is the issuer itself. (Semantic note banked: this is fine for
   correctness — the host waits on DONE flags — but "the pollers"
   leg-A imagined do not exist.)
3. DONE-HALT (+compaction) frees ~7 warps, but the win is then capped
   by the NO-FORWARDING INTERLOCK FLOOR: a lone warp's issue period
   is 4 (in_pipe clears at WB-entry; the WB-exit variant's period 5
   caps at 1.6x = FAIL). Period 4 on a pure-ALU scalar phase is
   EXACTLY 2.0x — the bar with zero margin (whole-kernel baselines
   show 2.48x only because boot/park overheads dilute the old
   denominator). A sigma measurement of ~2.00x vs a >= 2.0 bar is a
   coin flip; building two new spec surfaces (done-halt + the
   interlock rewrite + the catalog amendment) to chase it is what
   this program does not do.
4. THE REAL UNLOCK IS OUT OF SCOPE: exceeding 2x needs per-register
   hazard tracking (scoreboarding — issue independent instructions
   at period < 4), a materially bigger core change than the
   registered "no ISA change, semantics preserved" leg. REGISTERED to
   the ISA-v2/assembly wishlist beside tensor-fabric concurrency;
   done-halt itself is noted there as a cheap freebie with
   independent value (power/trace hygiene) if the wishlist opens.

The catalog amendment (D-032a in simt_spec) stays written and unused
— correct as spec'd, waiting on a mechanism that can pay for it. The
sampler taps + SCHED protocol remain (armed-only, zero cost idle) as
permanent observability. Leg order resolves to: C (CLOSED) -> A
(BANKED) -> D (spec-first, next) -> E (conditional).

### Leg C stage C2 pre-registration (2026-07-15, sidecar loaders —
### written BEFORE RTL and BEFORE any measurement)

Scope: tensor_sidecar's face flips to v2 edge/credit (CRED=4) and its
masters convert in one commit (they are one migration step): LD_A,
LD_B, LD_C0 (streaming loads — issue cursor and response cursor
decouple; placement keys off the RESPONSE index, legal because
completions are strictly in order), ST_C (streaming writes — compose
from the async bank read at issue, the idx+1 preload trick retires),
and the LUT stream (converts at ONE-OUTSTANDING on the edge face: its
load->store data dependency is serial and it is not on any measured
path — disclosed, not hidden). Phase transitions barrier on a full
credit pool (~LAT cycles each, trivial against thousands of beats),
so busy still means memory-visible. port_arbiter's B port becomes the
second v2 edge port (same skid+bypass shape as C; the legacy b_infl
adapter retires); A (core) is then the last v1 port.

Bar C2 (the leg-C >= 2x clause applied to this path, frozen now):
- measured: whole-op cycles of the frozen LOAD-DOMINATED shape
  GEMM8 M=64 K=64 N=1 acc=0 (A loads = 1024 beats >> COMP ~0.6k
  cycles; DMA dominates by construction — disclosed as whole-op
  because kernel mcycle stamps cannot cut inside the engine)
  improve >= 2.0x at LAT in {4, 8} vs current main (post-C1 = the
  pre-C2 tensor path).
- controls, all within [0.8, 1.25]: (a) the compute-dominated shape
  M=N=K=64 (COMP ~32k cycles dominates; a big move means the
  measurement, not the conversion, changed); (b) the C* drain and
  config phases from the C1 gate re-run vs the same baseline
  (already-converted and never-converted paths must both sit still).
- values: G15 G2 lattice bit-exact (the battery half), plus the C
  result image sha old-vs-new per LAT inside the gate.

#### Bar C2 instrument amendment (2026-07-15, post-first-measurement —
#### DISCLOSED; the >= 2.0x bar itself does not move)

The first run FAILED as registered (1.72x @ LAT4) and the failure was
the INSTRUMENT's, banked with mechanism: both registered shapes had
arithmetic errors. (1) The "load-dominated" N=1 shape ignores the
COMP loop's lane padding — iterations = M x K x ceil(N/8), so N=1
pays the full 4,096-cycle compute of N=8 and the op is ~71% COMP at
LAT=4; per-phase RTL instrumentation (scratch, reverted) measured the
LOADS themselves at 1.5 cyc/beat new vs ~7 old = ~4.6x — the
conversion performs as designed and the whole-op fence dilutes it
below any 2x for ANY shape (max whole-op ratio at this L/C is ~1.9x
at LAT=4). (2) The "compute-dominated" M=N=K=64 control ignores
ST_C's word beats (M x N = 4,096 — 57% of the old op is DMA), so the
control legitimately moved 1.64x. The C* cross-stage controls sat at
1.00x, proving the harness/instrument sane.

AMENDED MEASURED STATISTIC — differential DMA isolation, no RTL
observability changes, both arms measured identically: at fixed
M=K=64, LAT in {4, 8}:

    dDMA = op(N=8) - op(N=1)

COMP cancels exactly (both shapes share ceil(N/8)=1: 4,096+3
iterations each — frozen loop structure), LD_A cancels (same M x K),
doorbell/poll overhead cancels; the difference is PURE converted-path
DMA (LD_B: 112 beats + ST_C: 448 beats — a read stream and the write
stream). C2.a' = dDMA(base)/dDMA(new) >= 2.0 at LAT {4, 8}. C2.b
(values) and C2.c (C* controls in [0.8, 1.25]) unchanged; the N=1
whole-op ratio is recorded informationally as the original mistake's
tombstone. Same kernel (the two fenced slots become N=1 and N=8).

VERDICT (2026-07-15, full battery re-run ALL PASS): dDMA 3,360 ->
840 cycles = 4.00x @ LAT=4 and 5,568 -> 1,400 = 3.98x @ LAT=8 (bar
>= 2.0; 2.92x at LAT=1 info); C* controls 1.00x both; values
identical; new differential = 560 beats at exactly the trace's 1.5
cyc/beat (old: 6.0). N=1 whole-op tombstone: 1.72x/2.02x. Evidence
ci/logs/fastpath/c2_bar.json. Stage C2 LANDED; the PCONFIG walker
(stage C3) is the last leg-C conversion.

### Leg C stage C3 pre-registration (2026-07-15, PCONFIG walker — the
### last conversion; written BEFORE RTL and BEFORE measurement)

Scope: the PCONFIG walker converts to native credit issue and the
s_cluster lv shim RETIRES (no legacy users remain; the whole SoC
memory system is then v2 except the core's port A). Design: the
walker's per-beat CONSUME logic (fabric/shadow writes, cursor
advances, section-exit decisions) stays exactly as-is but fires on
completion pulses; a separate ISSUE engine streams addresses
credit-gated WITHIN a section. Sections stream internally and
barrier at boundaries because SECTION LENGTHS ARE DATA-DEPENDENT
(n_sites/arity/n_sched arrive in header beats; the ORD index count
arrives in its own section) — every length is latched strictly
before its section dispatches, and the last consume of a section
implies zero outstanding beats, so the existing exit transitions
are already the barrier (~1 round trip per section, ~8 sections,
negligible vs the ROWS bulk). dma_left (the image guard) decrements
per COMPLETION; dma_addr advances per ISSUE. The D33 E2 bracket
scans (header-exit and ROWS-exit) run with zero DMA outstanding by
the same barrier.

Bar C3 (the leg-C >= 2x clause, this path):
- measured: the C* CONFIG phase (t1-t0: the ROWS-only rewrite,
  9,232 beats — the never-converted control of stages C1/C2 becomes
  the measured path) improves >= 2.0x at LAT {4, 8} vs the branch
  fork point (post-C2 main).
- controls in [0.8, 1.25]: the C* drain phase AND the stage-C2
  differential dDMA (both already-converted paths must sit still).
- values: drain sha + gemm C sha identical per LAT; the s_cluster
  suite's bit-exact PCONFIG diffs ride the battery, with the unit
  bench's config phase additionally exercised at service latency 4
  (credit-stall coverage inside section streaming).

VERDICT (2026-07-15, full battery ALL PASS): config phase 56,080 ->
14,616 cycles = 3.84x @ LAT=4 and 93,088 -> 23,952 = 3.89x @ LAT=8
(bar >= 2.0; 2.75x @ LAT=1 info); drain and dDMA controls 1.00x;
values identical. Evidence ci/logs/fastpath/c3_bar.json.

### LEG C: CLOSED (2026-07-15)

All three conversions landed behind full batteries with measured
bars: C1 drain 4.54x/6.22x, C2 loaders dDMA 4.00x/3.98x (one
instrument amendment, banked + disclosed), C3 config 3.84x/3.89x
(all at LAT 4/8 vs >= 2.0). Word order and values bit-identical
throughout (G15/S7σ/S8σ + per-gate shas). The SoC memory system is
v2 edge/credit end to end except the core's port A (out of leg C's
scope — its conversion belongs to the assembly-era conversation, and
leg A's parking design may reshape core memory behavior first). The
one-outstanding valid/ack contract survives only there; D-017's race
class is structurally extinct everywhere else.

The PDRAIN streamer is the first migrated master: s_cluster's face is
the v2 edge/credit contract (CRED=4) on a dedicated third arbiter
port (skid FIFO + empty-bypass; A=core and B=tensor keep v1 level
untouched; the engine mux died with the level contract). The drain
issues one beat per cycle credits permitting — never waiting for
completions — and busy holds until the credit pool refills, so !busy
still means memory-visible. The PCONFIG walker is UNMIGRATED: its FSM
text rides a one-outstanding level->edge shim (its own conversion is
a later commit). Unit evidence: new tb/port_arbiter (contract bench:
per-port order/payload/routing, credit bounds, D-017 duplicates;
smoke+fuzz), s_cluster suite with every drain mode diffed bit-exact
vs the frozen oracle at lat=1 AND lat=4 (credit exhaustion), region
poisoned between passes. VERDICT (2026-07-15, full battery under the fixed
instrument — lint, 22/22 units, G15 incl. real G1, S7σ, S8σ, then
gates/fastpath_c.py vs the C0 baseline worktree): C1.a drain-phase
ratio 4.54x @ LAT=4 and 6.22x @ LAT=8 (bar >= 2.0; 2.62x even at
LAT=1, informational); C1.b drain images bit-identical per LAT;
C1.c config control 1.00x (the unconverted walker did not move —
the instrument is sane). ALL PASS; evidence
ci/logs/fastpath/c1_bar.json. Stage C1 is LANDED; next conversions
are the sidecar loaders, then the PCONFIG walker, same battery each.

### Leg B implementation note (2026-07-13)

Landed as the arbiter-funnel window: SCRATCH_BASE = 0xF000_0000, 8KB
byte-enable BRAM, 2-cycle on-chip service, reset-free RAM block
(D-028 lesson). One decoder serves all masters. Software-invisible
except latency (flat memory to goldens/ISS).

## In-Sweep Accumulation


### Verdicts (2026-07-15 ~21:30, RTL @ accwalk branch)

- **AW.1 MET**: full battery ALL PASS on the ACCWALK RTL — lint,
  units (s_cluster suite grew smoke_accwalk; 8 tests + fuzz×12 ALL
  bit-exact vs the oracle), G15, S7σ, S8σ, and the leg-D bar. Bench
  overlap tripwire:
  8 RECORD sweeps at n=64 in 908 busy cycles vs the 9,216-cycle
  walk-alone bound the serial P_ACC could never beat.
- **AW.2 MET with margin**: PSAMPLE phase 10,140,800 → 855,040 cycles
  at the identical 32-iter shape = **11.9×** (bar ≥10×); 417.5
  cyc/sweep incl. P_ZERO (bar ≤480); **P_ACC-exposed share 93.06% →
  0.00%** (bar ≤10%) — the walk is fully hidden. Per-sweep anatomy:
  P_WAIT 212.5 (the fabric, cycle-identical to pre), P_SNAP 57.8
  (copy + the predicted walker-drain stall), P_NEXT 17.2 (tail gate).
- **AW.3 MET (cycle-perfect controls)**: config phase 321,801 →
  321,801 and drain phase 19,220 → 19,220 — ratio 1.000 (band
  [0.9,1.1]); the slot-bank conversion is provably cycle-neutral for
  every untouched path. The OOC 8 ns re-hearing leg is DEFERRED WITH
  DISCLOSURE to first VM availability (both spot VMs TERMINATED;
  assembly opens with the VM restart — the pre-registered re-hearing
  stands and PR1 measures the true fit regardless).
- **AW.4 (informational)**: the mixed sampling workload fell from 0.885M
  to **0.573M cyc/iter = 1.55×** on top of v6-solo (**3.1× cumulative
  vs v5**); sampler idle at this shape rose 75.0% → 96.3% (sampling is
  now ~free; the residue is the core-side scalar border). Sampling-bound
  proxy:
  the S7σ battery itself fell 280.1 s → 70.2 s = **4.0×**, mid-range
  of the 4–6× C* expectation; the C* mixed-ISA iteration re-measures
  at the next fastpath-instrument run.

Follow-ups S1 (WALK_W=2) + S2 (banked P_ZERO): **LANDED 2026-07-16
~01:40** — see the "Committed follow-ups — LANDED" section below for
the verdicts (20.6× phase, sweep-bound, P_ZERO 16×, URAM disclosure).

## (original card below — spec-frozen state, kept as the record)
## ACCWALK — in-sweep moment accumulation (SPEC FROZEN — Shape A)

Status: **OPENED 2026-07-15** by the §V6.5 decision rule firing (P_ACC ≥
80% of PSAMPLE); **SEQUENCED BEFORE ASSEMBLY by user directive the same
day** (verbatim intent: *"i also really want in sweep accumulation for
the sampling … sampling fast is the entire selling point of this chip
and must happen"*); **Shape A CHOSEN by user directive the same day**
("lets go with shape a") and the **spec FROZEN as
`docs/HARDWARE_ARCHITECTURE.md#sampling-isa` §13 (amendment D-034)** — snapshot via the
existing sA_word port (the state_mirror path turned out SIM-ONLY),
1-site/cycle walker, slot-banked shadow + m2, all interlocks and the
cycle model written before any RTL. **Bars AW.1–AW.4 below are FROZEN
(2026-07-15).** Timebox and de-scope tripwire in §Sequencing. Next:
the RTL+bench sitting (verification queued behind the 12k v6 re-cert
that holds the box until ~19:45 today).

### The measurement that opened the card (2026-07-15, PSTAT tap)

Per-cycle FSM histogram of `s_cluster` across a full 32-iteration sampling
run (tap: `st` public_flat_rd + armed-only harness histogram; proven
cycle-neutral and bit-exact with or without instrumentation):

| where the PSAMPLE cycles go | cycles | share |
|---|---|---|
| P_ACC (moment read-out walk) | 9,437,184 | **93.06%** |
| P_WAIT (the actual fabric sweep) | 435,200 | 4.29% |
| P_ZERO (STATS_RESET zero walk) | 262,144 | 2.59% |
| P_LOADSCH/P_START/P_NEXT/P_ENTRY | 6,272 | 0.06% |

- P_ACC = **4,608 cyc/sweep exactly** = 2 cyc × 9 sub-steps × 256
  sites (the D-028 fetch/commit pair over 1+8 slots per site).
- P_WAIT = 212.5 cyc/sweep: the fabric performs an entire 256-site
  sweep **~21.7× faster than the walk that reads its moments out**.
- Every RECORD-flagged sweep pays the walk; this benchmark flags every sweep.
  At n=1024 (C* shapes) the walk is 18,432 cyc/sweep — same ratio.
- Whole-run context (smoke depth): sampler idle 75.0% (core-side
  scalar border — a different lever), PSAMPLE 24.1%, config 0.77%,
  drain 0.05% (v6's 128→4 batching already crushed drains).

### The invariant (load-bearing; the whole verification strategy)

**Drained values are architectural and MUST NOT change**: MOMENTS
(m1/m2), STATE, WORK, TELEMETRY bit-identical for every image, q,
schedule, and clamp configuration. ACCWALK changes WHEN counting
happens, never WHAT is counted: the statistics remain those of the
**post-sweep state**, per RECORD sweep. Consequence: **no golden
change at all** — the golden already defines the values; the entire
diff surface is cycles. This makes AW.1 a zero-tolerance bit-exact
bar over the existing σ machinery, the cheapest-to-verify kind of
RTL change this project does.

### Design direction (endpoint first, two shapes; spec sitting picks)

**Endpoint**: PSAMPLE becomes **sweep-bound** — the walk's cost is
hidden or eliminated, so a RECORD sweep costs ≈ P_WAIT (~213 cyc at
n=256), not P_WAIT + 4,608.

**Shape A — snapshot + overlapped wide shadow-walk (CHOSEN 2026-07-15;
spec = sampling_isa_spec §13 / D-034, which supersedes the sketch
below where they differ).** At `f_done`, latch the fabric's site state (all bit-planes;
~768–1024 FF — the walk's read source, frozen in time) into a snapshot
bank and **start the next sweep immediately**. A widened accumulator
walks the SNAPSHOT into **banked m1/m2** (K sites/cycle, bank = site
mod K) while the next sweep runs. K=16 ⇒ 288 cyc walk vs 213 cyc sweep
(≈75 cyc exposed); K=32 ⇒ 144 cyc, fully hidden. The LAST sweep of a
PSAMPLE has no successor to hide behind — one exposed wide walk per
PSAMPLE, amortized over its sweeps. Why primary: the snapshot IS the
post-sweep state, so AW.1 holds **by construction**; the seam stays
inside s_cluster (fabric only needs its state parallel-readable at
f_done, which the current walk's fetch path already implies).
Expected PSAMPLE phase: 16×(213)+~300 ≈ **3.7k cyc vs 77.1k today ≈
21× phase win** at the registered sampling shape (K=32).

**Shape B — true in-sweep (later-endpoint edge accumulation).** m1[i]
accumulates when site i commits its update; edge (i,nbr) accumulates at
the LATER-updating endpoint (schedule/color order known), where both
post-states exist — mathematically identical to the post-sweep pair
count IF every site updates exactly once per sweep. Walk disappears
entirely (last sweep too). Why fallback: the once-per-sweep premise
must be proven against clamps/schedule forms (clamped sites never
"update" — their edges need an ownership convention), and the write
bandwidth lands mid-sweep at the fabric's update rate — the m-counter
ownership seam the original registration flagged. Choose only if the
spec sitting finds Shape A's exposed-walk residue matters (it doesn't,
at current shapes).

**Known hard points — RESOLVED at the 2026-07-15 spec sitting**
(dispositions in sampling_isa_spec §13: shadow → slot-banked at ±0
BRAM, not replicated; snapshot source → the existing sA_word/2/3 port,
n/32+2-cycle copy, because state_mirror is sim-only; drain → bank
select on addr[2:0], wire order identical; q8 → 3 planes packed per
snapshot word; P_ZERO → untouched in v1. Kept for the record):
1. Shadow ROWS read bandwidth — the walk reads (validity, nbr) per
   slot; K-wide needs K slot-words/cycle: repack slots-per-word or
   replicate the shadow BRAM (est. +2–8 BRAM36; K26 has headroom, but
   PR1's fit bar wants this number BEFORE assembly freezes margins).
2. Banked m1/m2 vs the drain contract — PDRAIN streams 1 beat/cyc in
   address order; the drain cursor reads bank (addr mod K); order
   preserved, but the credit-face timing must not regress (AW.3
   control covers it).
3. q8: 3 bit-plane snapshot, 7-lane m1 indexing (m1[{site, val}]) —
   same banking, wider lanes; S8σ fuzz is the net.
4. P_ZERO rides along: banked zeroing cuts the 2,048-cycle reset walk
   ~K× for free.
5. Timing: s_cluster currently meets 8 ns; the wide walker adds
   mux/adder trees — OOC re-hearing pre-registered as part of AW.3.
   **RE-HEARING CLOSED 2026-07-16 (main @ 8bb508b): 8ns MET, WNS
   +0.026 at NB=10, routed** — but only after TWO synthesis-contract
   fixes the hearing itself surfaced: (a) the v2 2-D-array banking
   was not a RAM template (Synth 8-7186) and dissolved into 191k FFs
   + 77k LUTs — restructured to per-bank g_bank generate memories
   (banks now RAMB18s; 23.6k LUT / 41 BRAM / 24 URAM total); (b) the
   COV drain's BRAM→bank-mux→64-bit-MAC→CARRY8 single-cycle path
   missed by 0.414 — m1[neighbor] now stages through d_mnb (third
   fetch phase; D-026, values bit-exact by full battery). Evidence
   ci/logs/ooc/rehear/fix_p8.000_*. The 7ns question is CLOSED as
   NO for this silicon generation: post-fix s_cluster needs 7.50ns
   (WNS −0.496) and leg-E q8 fabric needs 7.33ns (WNS −0.329) — the
   ~1.14× clock lever stays parked behind S3's 2–4×.

### Bars (FROZEN 2026-07-15, at the spec sitting, before any RTL — D-034)

- **AW.1 (values, zero tolerance)**: full S7σ + S8σ batteries and a
  192-iteration smoke, every drained word bit-identical to pre-ACCWALK.
  Plus the existing s_cluster bench suite bit-exact vs the oracle.
- **AW.2 (the win)**: PSAMPLE phase cycles on the registered shape (n=256,
  RECORD-all, sweeps=16) reduced **≥ 10×** (measured by PSTAT: phase
  cyc/sweep ≤ 480 vs 4,821 today); P_ACC-exposed share of PSAMPLE
  ≤ 10% (from 93.06%).
- **AW.3 (no collateral)**: config phase and drain phase cycle counts
  in [0.9, 1.1] of pre-ACCWALK (untouched paths sit still); s_cluster
  OOC WNS ≥ 0 at 8 ns; BRAM/LUT delta disclosed vs the frozen estimate.
- **AW.4 (end-to-end, informational — no threshold)**: mixed-workload
  cycles/iteration re-measured (expect ~1.25–1.3× on top of v6-solo;
  sampler idle 75%
  caps it — that residue belongs to the scalar-duty levers, not this
  card); C* mixed-ISA iteration re-measured (expect ~4–6×: post-
  FASTPATH the C* border is ~89% PSAMPLE compute).

### Committed follow-ups — LANDED 2026-07-16 ~01:40 (§13 v2, D-034 add.2)

**WALK_W=2 + banked P_ZERO both landed in one commit re-run through
the same battery, bars unmoved and re-met**: PSAMPLE phase 493,056
cyc at the frozen shape = **20.6×** vs pre-ACCWALK (projection ~21×
hit); P_WAIT = 88.3% of phase — the sweep-bound endpoint; P_ZERO
65,536 → 4,096 cyc/RESET (**16×**, better than the promised 8-wide);
walk fully hidden (P_SNAP = pure copy, zero stall); results bit-exact at both
depths. Honest correction: W=2 was NOT a parameter flip —
it re-banks m1/m2/sh_slot 16-way ({site parity, lane/slot}).
Resource math corrected 2026-07-16: URAM-block count NEUTRAL at
NB=13 (v1 banks were 2-URAM cascades already), doubles only at
NB=12, BRAM-mapped below; +6–8k LUTs worst case; PR1/OOC measure,
and an 8-bank W=1 profile knob is the escape hatch.

### Sequencing — BEFORE assembly (user-directed), with a tripwire

Decision 2026-07-15: ACCWALK is the next sitting, **before full-SoC
assembly**. Rationale: (1) sampling throughput is the chip's thesis —
SG0's board demo should not ship a sampler spending 93% of its time on
bookkeeping when the fix is contained; (2) doing it BEFORE assembly
means PR1's fit and PR2's bit-exact bars measure the REAL sampler once
— landing it after assembly forces a second pass through routed-fit/
swap verification; (3) the verification net is at peak readiness
(σ-batteries hot, v6 reference fresh, PSTAT instrument built —
before/after is one flag); (4) the ISA surface does not move (PSAMPLE/
PDRAIN semantics identical), so nothing downstream re-opens.

**Timebox/tripwire**: 3 sittings (spec → RTL+bench → battery+bars).
If the spec sitting cannot hold AW.1 (bit-identical values) or the
pre-check says 8 ns / BRAM budget breaks, **bank the card with the
mechanism and fall back to assembly on the original plan** — the walk
works today; nothing is broken while deferred. Assembly's §5.2 queue
(PR4 spec → builds → PR1/PR2/PR3, VM restart) is unchanged, just
shifted behind this card.

Deliberately out of scope here: the 75% core-side scalar border (warp-
granular pool / scoreboarding — ISA-v2 wishlist), tensor–fabric
concurrency (assembly's §3.6.6 conversation), any change to what
MOMENTS mean.
