# GPU–TSU Software and Validation

This document consolidates the device runtime, CUDA-shaped programming model,
serving engine, GPT-2 validation workload, quantization study, and the
golden-first RTL verification methodology.

## Contents

- [Kernel Runtime and ABI](#kernel-runtime-and-abi)
- [CUDA-Shaped Programming Model](#cuda-shaped-programming-model)
- [Serving Engine](#serving-engine)
- [GPT-2 Integer Pipeline](#gpt-2-integer-pipeline)
- [GPT-2 Device Sequencer Notes](#gpt-2-device-sequencer-notes)
- [Quantization Study](#quantization-study)
- [Testbench Methodology](#testbench-methodology)


## Kernel Runtime and ABI


The "tiny-CUDA" contract: how compiled C kernels run on the simt_core.
Everything downstream programs against THIS document — M18's G-gates,
M21's sampling opcodes (`pconfig`/`psample`/`pdrain` wrap into this same
launch path), and M23's Llama ops. The bring-up campaign flips
`--backend sim` to `--backend hw` with zero kernel changes.

### 1. Execution model

- One **launch** = all W×L = 64 hardware threads enter `kernel_main(tid)`
  simultaneously at reset; `tid = mhartid = warp*8 + lane` (M14 spec).
- Work > 64 items uses the **grid-stride idiom** with warp-uniform
  bounds (§4). Multi-launch batching is a host loop.
- A thread finishes by returning from `kernel_main`; crt0 then writes its
  DONE mailbox and parks on a warp-uniform self-loop. The host polls the
  64 mailboxes; the launch completes when all read 1.

### 2. Memory map (frozen; RAM = 2 MiB flat, byte-addressed from 0)

| region | base | size | notes |
|---|---|---|---|
| code + rodata | 0x0000_0000 | 128 KiB | crt0 at 0; RESET_PC = 0 |
| param block | 0x0002_0000 | 256 B | §3; written by host before launch |
| DONE mailboxes | 0x0002_0100 | 256 B | word per tid; host-cleared pre-launch |
| heap / buffers | 0x0003_0000 | ~1.6 MiB | host-managed allocator, 64 B aligned |
| stacks | 0x0020_0000 (top) | 64 × 16 KiB | sp(tid) = TOP − tid·16 KiB, grows down |

### 3. Param block layout (offset from 0x0002_0000)

| offset | field |
|---|---|
| 0x00 | u32 n_args |
| 0x04 | u32 grid_n (total work items, kernel-interpreted) |
| 0x08.. | u32 args[14] — buffer addresses or scalar values, in order |

C side: `mk_arg(i)`, `mk_grid_n()`, `mk_tid()` accessors in `mk.h`.

### 4. Divergence rules for kernel authors (v1 hardware limits, M14 spec)

1. **Backward branches must be warp-uniform.** Loop bounds may not
   depend on lane-varying values. The grid-stride idiom is:
   `for (u32 i = tid; i < grid_n_padded; i += 64) { if (i < n) {body} }`
   where `grid_n_padded` is n rounded up to a multiple of 64 by the HOST
   (so every thread iterates the same count; the body guard is forward
   divergence, which is fully supported).
2. Forward if/else divergence: unrestricted (max-PC stack, depth 8).
3. Indirect calls / lane-divergent function pointers: forbidden (traps).
4. Compiled at `-O1 -march=rv32i_zicsr -mabi=ilp32` — the kernel build
   wrapper enforces flags; -O2 is opt-in per kernel after inspection
   (compilers may rotate guarded loops into lane-divergent backedges).

### 5. Host protocol (persistent sim process)

`tb/simt_soc/soc_harness.cpp` — Verilated simt_core + 2 MiB RAM behind
the spec §1 (imem 1-cycle) and mem_spec §1 (dmem valid/ack) contracts —
speaks a line protocol on stdin/stdout (one reply line per command):

| command | reply |
|---|---|
| `LOAD <hexaddr> <hexbytes>` | `OK` |
| `PEEK <hexaddr> <len>` | `<hexbytes>` |
| `RESET` | `OK` (asserts rst_n low 2 cycles; RAM persists) |
| `RUN <cycles>` | `RAN <cycles> <commits>` |
| `COUNTERS` | `CYCLES <n> COMMITS <n> WBCOMMITS <n> MEMCOMMITS <n> DBEATS <n>` |
| `QUIT` | (exits) |

The process persists across launches: model build cost paid once,
RESET+LOAD between kernels. Counter definitions: DBEATS = dmem ack beats
(the coalescer-efficiency numerator for G-gates).

### 6. Host API (`host/run_kernel.py`)

    rt = Runtime(backend="sim")           # spawns/attaches the harness
    k  = rt.compile("sw/kernels/vec_add.c")
    a  = rt.buffer(np.int32 array); b = rt.buffer(...)
    c  = rt.buffer(nbytes=...)
    stats = rt.launch(k, grid_n=N, args=[a, b, c, N])
    rt.read(c) -> np.ndarray                        # download + diff host-side

`--backend hw` constructs the same Runtime against the (stubbed) UART
bridge — NotImplementedError until the Bring-up Campaign, same interface
by construction.

### 7. Phase-6σ DoD (gates/m17_runtime.py, frozen)

1. Toolchain: every kernel in sw/kernels compiles with the enforced
   flags; the ELF→image path is byte-exact (loader self-check).
2. Execution: each gated kernel runs on the persistent sim backend and
   its outputs diff EXACTLY vs the NumPy reference (integer kernels: no
   tolerance).
3. Batching: ≥2 kernels launched through ONE harness process (persistence
   proven — RESET isolation verified by re-running kernel 1 after 2).
4. Divergence coverage: at least one gated kernel exercises forward
   divergence; at least one uses the grid-stride idiom with grid_n > 64.
5. Counters: RUN reports cycles+commits; DBEATS < lane-serial baseline
   on the coalescing-friendly kernel (uniform loads) — the M16 win
   measured through the real stack.
Positive completion everywhere; evidence ci/logs/m17/.

## CUDA-Shaped Programming Model


Every CUDA concept, mapped to our stack with honest status:
**✓ works today** · **△ works with v1 limits** · **◇ designed, lands at
the milestone noted** · **✗ out of scope for this architecture**.

### Programming model

| CUDA | mk equivalent | status |
|---|---|---|
| `__global__` kernel | `void kernel_main(u32 tid)` in plain C | ✓ |
| `threadIdx/blockIdx/blockDim` | `mk_tid()` (global), `mk_warp_id()`, `mk_lane_id()` | ✓ |
| grid > device threads | `MK_GRID_STRIDE(i, n)` — host pads grid_n | ✓ |
| kernel arguments | param block: `mk_arg(i)`, buffers as addresses | ✓ |
| warp size 32 | warp size 8 (L=8 lanes) | ✓ |
| multi-block concurrency | 8 warps × 8 lanes = 64 resident threads | ✓ |
| dynamic parallelism | — | ✗ (host loops launches) |

### Divergence + synchronization

| CUDA | mk equivalent | status |
|---|---|---|
| if/else divergence | max-PC reconvergence stack, depth 8 | ✓ (D-016-hardened) |
| loops with lane-varying trip counts | **backward divergence traps** — restructure with MK_GRID_STRIDE + guards | △ v1 limit; general PDOM is a v2 study |
| `__syncwarp()` | implicit — lanes are lockstep every instruction | ✓ (free) |
| `__syncthreads()` (inter-warp) | — within a launch; split into 2 launches | △ (◇ M18 can add a barrier CSR if G-gates demand) |
| atomics | — (RV32I has no A extension) | ✗ v1 (◇ A-ext is a v2 ISA decision) |
| warp shuffle/vote | — | ◇ candidate M18 sidecar ops |
| function pointers / virtual dispatch | divergent JALR traps | △ uniform-only |

### Memory model

| CUDA | mk equivalent | status |
|---|---|---|
| `cudaMalloc/cudaFree` | `rt.alloc()` / `rt.free_all()` (bump arena) | ✓ |
| `cudaMemcpy` H2D/D2H | `rt.to_device/from_device/write/read` | ✓ |
| `cudaMemset` | `rt.memset(buf)` | ✓ |
| global memory | 2 MiB flat RAM (map: runtime_spec §2) | ✓ |
| coalesced access | M16 coalescer: warp-uniform words dedupe 8:1 (measured: gate rule 5) | ✓ |
| shared memory / scratchpad | — | ◇ M18 (tensor sidecar brings SRAM; sampling scratchpad at M21) |
| constant memory | param block (256 B) | ✓ |
| unified/managed memory | everything is one address space already | ✓ (trivially) |
| L1/L2 caches | dcache verified (M16); SoC integration behind the same valid/ack face | ◇ M18 G-gates |

### Toolchain + host API

| CUDA | mk equivalent | status |
|---|---|---|
| nvcc | riscv-none-elf-gcc 15.2 via `rt.compile()`, content-hash cached | ✓ |
| `-O3` | **-O0 default = SIMT-safe by construction** (unoptimized backedges are unconditional jumps → warp-uniform); -O1+ opt-in after disasm inspection — observed twice turning legal C into divergent backward branches | △ by design |
| NVRTC (runtime compilation) | `rt.compile()` is already runtime compilation | ✓ |
| cuModuleLoad | LOAD protocol into the persistent process | ✓ |
| streams / async | single stream, synchronous launches | ◇ (harness protocol supports interleave later) |
| CUDA context | `Runtime()` = one persistent Verilated SoC process | ✓ |
| `--backend hw` | same API, UART bridge stub → Bring-up Campaign flips it | ◇ SG0 |

### Debug + profiling (where we beat toy stacks)

| CUDA | mk equivalent | status |
|---|---|---|
| printf from device | `mk_log(v)`: per-thread 32-slot rings, `rt.device_log()` | ✓ |
| `assert()` in kernel | `mk_assert(cond, code)` → host raises `KernelAssert(tid, code)` | ✓ |
| illegal-op diagnosis | **crt0 trap handler**: mcause+mepc+mtval per thread reported to host with decoded v1-limit hints | ✓ |
| nvprof / Nsight | per-launch `LaunchStats`: cycles, commits, IPC, memory beats | ✓ |
| cuda-memcheck | golden-diff mode: `kernel_lockstep` bench runs ANY compiled kernel image instruction-lockstep vs the golden SIMT ISS | ✓ (unique: full-machine formal-adjacent replay) |
| compute-sanitizer race check | single-owner memory unit; cross-warp races possible as on real GPUs | △ same as CUDA |

### Kernels proven today (gate battery + showcase)

vec_add · saxpy (software mul via transform-proof `mk_mulu`) · relu_abs
(nested divergence) · reduce_partial (64-way partials) · bcast_scale
(coalescing showcase, 12% fewer memory beats/elem than vec_add) — all
EXACT vs NumPy through one persistent process. M15's INT8 matmul runs at
the bench layer; its C-kernel port is the M18 G-gate warm-up.

### What this stack uniquely enables downstream

- **M21**: `pconfig/psample/pdrain` become kernel-callable ops in this
  same ABI — the world's first *sampling instructions* behind a CUDA-
  shaped API (Artifact #3's claim).
- **M23**: RoPE/RMSNorm/SwiGLU/GQA as mk kernels vs llama.cpp dumps.
- **Bring-up**: `--backend hw` swaps the process for silicon; every
  kernel, test, and profile above re-runs unchanged.

## Serving Engine


Why: G5 measured the v1 sidecar at 5.12 MACs/cycle sustained; Llama-1B
W8 decode needs ~1.2 GMAC/token, so the v1 engine tops out ~0.4 tok/s
against the registered >= 8 tok/s KV260 target — and the SoC's narrow
port (32b, one outstanding) caps ANY engine at ~0.4 GB/s, 30x under
the board's ~12-15 GB/s DDR4. SV2 adds, ADDITIVELY (v1 sidecar,
s_cluster, core contracts untouched):

1. a WIDE streaming read face (512b/beat, burst) — the sim model of
   the 4x AXI-HP aggregate;
2. `tensor_array` (TA), a weight-STREAMING GEMV/batched-GEMM engine
   in the reserved D4 growth op space:
   - T_OP 6 GEMV_STREAM: y[M] (int32) = W[M x K] (int8 or packed
     int4) . x[K] (int8); weights stream through the wide face and
     are used once; x is resident in an internal SRAM.
   - T_OP 7 GEMM_VERIFY (v2.1, spec'd now, built after T_OP 6):
     Y[M x B] = W[M x K] . X[K x B], B <= 8 — the speculative-decode
     verify shape: one weight beat feeds B MAC lanes, which is the
     ONLY way past the port ceiling in decode;
3. W4 weight format (frozen here; model-side quality bars live with
   M23): packed little-endian nibbles, zero-point 8 (w = nib - 8,
   range [-8, +7]), raw int32 accumulation in-engine; SCALING IS
   MODEL-SIDE (per-row scales applied at requant, exactly like the
   v1 gemm8 -> requant contract).

### The port-width law (pinned so nobody sizes the array wrong)

Decode is matvec: every weight byte is used ONCE. MACs/cycle is
therefore min(array width, port bytes/cycle) in W8. Array capacity
beyond the port pays ONLY under batch (T_OP 7). AND the face itself
must out-run the DDR budget or IT becomes the wall: at 250 MHz,
256b/cycle is 8 GB/s < the frozen 12 GB/s budget — undersized.

Config changelog (recipes iterate, bars do not — R3):
- v0 (spec creation): BEAT = 256b. Caught BEFORE any gate ran: at
  250 MHz the face caps at 8 GB/s and the SV2.4 W8 projection lands
  ~6.4 tok/s < the frozen 8 — the port, not the DDR, would bind.
- v1 (pre-registered 2026-07-09, before gates): BEAT = 512b
  (16 GB/s at 250 MHz >= the 12 GB/s budget; the 4x AXI-HP aggregate
  honestly delivers it). PE row = 64 (W8) / 128 (W4), B_MAX = 8.
  K/rows pad to 64 bytes (W8) / 128 nibbles (W4). Bars unchanged.

### Frozen interfaces

- CSR map (v1 socket, widths already carried): T_OP 6/7; T_A weight
  base; T_B x base (X base for T_OP 7); T_C y base; T_M packs
  {k[11:0], m[11:0]} (rows m <= 4095, cols k <= 4095; both nonzero);
  T_N[2:0] = B-1 for T_OP 7 (0 for T_OP 6); T_FLAGS bit0 = W4,
  bit1 = X_RESIDENT (skip x load; reuse the SRAM contents), others 0.
- Wide read face (TA <-> memory): request {addr[31:0] aligned to
  64B, beats[15:0]}; then a stream of 512b beats, one per cycle when
  the server has them (valid; engine always ready). One outstanding
  burst; bursts back-to-back. Reads only.
- Narrow face: TA also owns a v1-style 32b port for x loads and y
  writes (small traffic), muxed into the existing port_arbiter as a
  third requester (engines never concurrent: one outstanding command
  globally per the socket contract).
- K alignment: K padded to 64 (W8) / 128 (W4) by the caller with
  zeros (one beat); W rows padded likewise (the golden defines it).

### Bars (FROZEN — move only via D-entry)

| bar | claim | frozen pass rule |
|---|---|---|
| SV2.1 | exactness | TA output bit-exact vs the golden (W8: golden/tensor.py gemm8 with N=1; W4: golden/quant4.py) on the directed suite (M,K in {1, 31, 32, 33, 512, 2048} crosses, W8+W4, x-resident on/off) + 200 fuzz shapes, zero tolerance, exact-diff bench on the house pattern (smoke/fuzz) |
| SV2.2 | streaming efficiency | On M=2048, K=2048 GEMV from the wide face: total cycles <= 1.10 x ceil(K/(BEAT/8))*M (W8) resp. ceil(K/(BEAT/4))*M (W4) at the registered BEAT — the engine keeps the port >= 90% saturated end-to-end, measured by the bench |
| SV2.3 | it synthesizes fast | TA passes OOC on xck26 at 4.000 ns (250 MHz — STRICTER than the G8 8 ns; a new-module addition, not a bar move) with WNS >= 0 and >= 1 DSP48 per 2 MACs of array capacity (proves the INT8 packing mapped) |
| SV2.4 | honest projection | gates/sv2_project.py computes tok/s = min(BW/bytes-per-token, MAC-rate/MACs-per-token) from MEASURED bench cycles at frozen budgets (12 GB/s effective DDR, TA at 250 MHz): Llama-1B W8 >= 8 tok/s AND W4 >= 16 tok/s, all inputs printed in the report (σ-projection, labeled) |
| SV2.5 | batched verify (v2.1) | T_OP 7 bit-exact vs gemm8 for B in 2..8 + the same efficiency law with B lanes; frozen NOW, gated when built — speculative-decode orchestration kernels and acceptance logic are M23-side work |

Evidence: ci/logs/sv2/. Build order (Golden-First): this spec ->
golden/quant4.py (+self-check) -> rtl/tensor/tensor_array.sv ->
tb/tensor_array (exact-diff, smoke/fuzz) -> OOC run -> simt_soc
integration (third arbiter requester + CSR routing) -> gates.

### T_OP 7 implementation freeze (2026-07-14, BEFORE RTL — golden-first)

The v2.1 surface the golden and RTL share, frozen here:

- B = T_N[2:0] + 1 everywhere in the engine — T_OP 6 IS B=1 (the CSR
  map already says T_N=0 for op 6). One datapath, no op decode.
- X image (golden/quant4.py::x_image defines it): COLUMN-MAJOR, each
  column zero-padded to KPAD = ceil(K/kstep)*kstep bytes (kstep 64 W8
  / 128 W4 — x elements are bytes in both modes); column b at byte
  offset b*KPAD. Capacity (v1): B*KPAD <= 4096 bytes (the x SRAM);
  larger K rides the M23 split-K tiling exactly as T_OP 6.
- Y: ROW-MAJOR M x B int32 at T_C (gemm8's C layout with N=B).
- Structure: the weight beat, W operand select and K-guard are SHARED
  across B accumulate slices; per-slice x windows read via per-column
  registered cursors (kcol[b] = b*KPAD + kx — every window stays
  64B-aligned, so each slice keeps the D-031 read/product/reduce
  shape). Product registers stay DSP M/P-absorbed per slice.
- Area/timing risk disclosed at freeze: up to 8x reduce trees and
  8*128 DSP products. The fit+timing claim is a VM OOC verdict at
  4.000 ns (SV2.3 flow), yosys proxy as the leading indicator; if the
  LUT delta threatens the G8.2 20% headroom, the structure iterates
  (DSP-cascade reduce absorption) — bars do not move.
- The Y-drain law (disclosed BEFORE any efficiency measurement): Y
  rides the one-outstanding narrow port at ~2 cycles/word, so a row
  costs 2B drain cycles against ceil(K/kstep) streaming cycles —
  sustained port saturation requires beats/row >= 2B. With the v1
  4KB x SRAM (B*KPAD <= 4096) the full-rate frontier is therefore
  K-dependent (e.g. W8 B=4 @ K=1024 saturates; B=8 cannot at any
  feasible K). SV2.5's efficiency clause is measured AT feasible
  shapes: W8 B=4 M=2048 K=1024 and W4 B=2 M=2048 K=2048, bound
  1.10x ideal beats, recorded in ci/logs/sv2/eff.json; correctness
  (bit-exactness) is gated at ALL B in 1..8. Wider Y (or a larger
  x SRAM) is the registered v2 lever if serving numbers ever need
  full-rate B=8; the M23 verify shape (k=4 drafts -> B=5, split-K
  768) sits inside the feasible frontier.

### σ-scope

All cycle numbers are Verilator observables; the DDR budget and the
250 MHz TA clock are frozen model inputs, not measurements — every
projection prints them (R7). The multi-clock SoC (TA at 250 MHz, core
slower) is a bring-up-campaign CDC design pinned at the D4 socket
boundary; in sim everything runs one clock and the projection scales.

### SV3 — the 3B-class MIXED-PRECISION tier (pre-registered 2026-07-09, before any measurement; D-024)

Target (pinned per D-024, replacing the earlier Llama-1B tier):
**VibeThinker-3B** — WeiboAI, Qwen2.5-Coder-3B base, Qwen2ForCausalLM,
3.09B params, 36 layers, d=2048, 16 Q / 2 KV heads (hd 128), MLP
11,008, vocab 151,936 (tied embeddings), MIT license, GGUF available
(the llama.cpp dump path holds). Qwen2-family op additions to the M23
golden list: attention QKV biases; tied embeddings. The 11,008-wide
MLP and 151,936-row lm_head ride the T_M split-K/split-M tiling. Format: MIXED PRECISION, frozen as a RULE (not tuned post-hoc):
embeddings/lm_head and every weight tensor of the FIRST and LAST
transformer blocks are W8; all other weight tensors are W4. The
engine already supports this with ZERO RTL change — precision is a
per-tensor T_FLAGS bit on each GEMV call. Scales stay model-side.
Both throughput legs are gated: WITHOUT speculative decoding (raw)
and WITH it (T_OP 7 batched verify + a draft).

Speculative protocol (frozen): GREEDY target decoding in v1 —
acceptance is token equality, exactness is trivial to verify and the
device output must equal the non-speculative output token-for-token.
(Distribution-correct SAMPLED speculation is a later claim needing
its own golden for the rejection-correction math.) Draft: a <= 300M
W4 model OR the self-speculative variant (first-N-layers early exit
of the target — zero extra weight memory); the gate reports which.
k = 4 drafted tokens per verify. Acceptance is measured on the
FROZEN eval slice only.

| bar | claim | frozen pass rule |
|---|---|---|
| SV3.1 | capacity | Total device image (MP weights + both KV caches at ctx 2048 int8 + draft weights if any + runtime) <= 3.0 GB — >= 1 GB headroom on the 4 GB SOM, computed at model-prep, itemized in the report |
| SV3.2 | raw throughput (σ) | SV2.4 machinery on the MP byte count: >= 5 tok/s at the frozen budgets (12 GB/s, TA 250 MHz), all inputs printed |
| SV3.3 | speculative throughput (σ) | On the frozen eval slice, greedy, k=4: effective tok/s >= 2.0x the SV3.2 measured raw AND >= 10 tok/s; device speculative output token-identical to non-speculative |
| SV3.4 | quality | MP model perplexity <= 1.06x the FP reference on the frozen eval slice (the [Quantization Study](#quantization-study) methodology extended to W4/MP; per-tensor rule above, no post-hoc reassignment) |

Sequencing: gates AFTER the M23 op-correctness suite (on
VibeThinker-3B vs llama.cpp dumps) and T_OP 7 are green. A miss on SV3.3's acceptance-dependent half is a
D-entry conversation, not a silent re-run on friendlier text.

## GPT-2 Integer Pipeline


Artifact #2's milestone: GPT-2 small (124M) generating tokens on the
simt_soc, every device operation INTEGER-ONLY (RV32I + the M18 sidecar),
bit-exact against a golden integer chain that is itself validated against
FP32 within a frozen perplexity delta.

### 1. Memory map amendment (runtime_spec §2 revision, documented)

SoC RAM grows 2 MiB → 256 MiB for M19+ (sim memory is host RAM; the
KV260 has 4 GB). Low map unchanged (code/params/mailboxes/log/stacks);
heap becomes 0x0020_0000 .. 0x0FFF_FFFF (above the per-hart
stacks, which stay at their M17 addresses). Weights are host-loaded once
per session; the M19 runs skip Verilator --savable snapshots because the
whole 20-token job costs ~1 min of sim wall, so snapshots are unnecessary.

### 2. Quantization scheme (OUR Q8_0 flavor, frozen as-built — D-018)

The scheme below is what G7 froze after construction-phase iteration
(bars untouched; the original max-abs/per-tensor draft measured +743%
and was redesigned; the D-018 rationale is summarized in the quantization
study below).

- Weights: per-output-row symmetric int8 AFTER the equalization fold
  (below). Stored row-major int8 + per-row requant pair.
- Residual stream: int32 at a GLOBAL 2^-16 scale. Adds are exact; LN is
  scale-invariant; GPT-2's 650x per-layer residual growth costs nothing.
- Activation interfaces are per-channel equalized: x is stored at
  per-channel scales u_c and every consuming GEMM absorbs u_c into its
  weights BEFORE quantization (W~_cr = W_cr * u_c) — kernels unchanged.
- Interface widths: q/k/v int8 (scores need int8 x int8); LN outputs,
  attention output, GELU output at int15, realized on the int8 sidecar
  as TWO GEMMs (hi7 = x>>7, lo7 = x&127; acc combined 64-bit in the
  kernel). Attention weights u14 via the same hi/lo split.
- Scale calibration (FROZEN protocol): 4x1024-token text8 TRAIN-region
  windows, per-channel 99.9th pct (gelu: max) floored at max/64,
  aggregated across windows by max. int8 interfaces: alpha=0.75
  channel/tensor split, margin 1. int15 interfaces: alpha=0.5,
  margin 16 (clipping was their ENTIRE quantization cost; alpha<1
  compresses the weight-row spread the fold induces).
- Requantization: integer-only gemmlowp style —
  y = sat( (acc * M + (1<<(sh-1))) >> sh ), (M, sh) offline per row;
  M < 2^15, 1 <= sh <= 30. Wide products use the kernel's software
  64-bit multiply.
- Biases: int32 at acc scale (input unit folded into the weights, so
  acc unit = srow/128 for int15 inputs).

### 3. Integer op set (all bit-frozen in golden/gpt2_int.py)

| op | device realization (as frozen, D-018) |
|---|---|
| GEMM | M18 sidecar, 64x64x64 tiles, FLAGS.acc chaining; int15 inputs as hi7/lo7 dual GEMM, 64-bit combine |
| LayerNorm | exact int32 deltas: mean = (sum*21845)>>24; s0 = max(0, msb(max abs d)-26); vs = sum((d>>s0)^2) in 64-bit; 1/sqrt = RSQRT LUT seed + ONE Newton step against the full input (range-reduced to <2^31 by even shift); n = (d*r)>>(k+s0-16); y = sat15((n*gq + 1<<19)>>20 + bq) |
| GELU | delta form at q8 input: y = relu(g) + interp(GELUD LUT)>>4, GELUD = (gelu-relu)(x/32)*512 int8, linear interp on the 3 fractional bits |
| Softmax | int16 scores in the x/32 domain (clip only AFTER max-subtract); EXP LUT on z>>1 plus one multiply (exp(-1/32)*2^15 = 33406) for the odd half-step; weights u14 = (e<<14)/sum |
| Attention | QK^T via sidecar (K cached transposed int8), per-head score (M,sh); PV as the u14 hi/lo dual GEMM |
| Sampling | greedy argmax over q8 logits (v1; temperature sampling is a fabric job later — noted for M21 fusion) |

### 4. Gates (frozen)

- **G7 (golden quality)**: Q8_0 golden-chain perplexity on the frozen
  eval slice within **5% relative** of the FP32 numpy forward, both
  computed by us; FP32 forward itself sanity-checked (finite, ppl in
  [15, 60] on the slice — catches broken loads, not a leaderboard claim).
  Eval slice: 2048 tokens, frozen at first materialization (sha pinned).
- **G6 (device generation)**: 20 greedy tokens from a frozen prompt on
  the simt_soc == the golden integer chain's 20 tokens EXACTLY (token
  IDs identical; logits of the final step within 0 — bit-exact is the
  bar because both sides are integer-deterministic). Runs nightly.
- Both under LAT >= 1 with G1's invariance already proven at M18.

### 5. Build order

golden/gpt2_load.py (safetensors+BPE, no torch) → fp32 forward validated
→ golden/gpt2_int.py (bit-frozen integer chain, G7) → device kernels
(tiled GEMM driver, ln/softmax/attn, sequencer loop) → gates/g6_gpt2.py.

## GPT-2 Device Sequencer Notes


### Execution model
- One kernel launch runs the WHOLE generation (prefill + 20 greedy
  steps) on all 64 harts; host reads the token trail from the log ring.
- Data-parallel sections split rows by global hart id (64-way);
  sense-reversing full barrier between phases (bar words in low RAM;
  polls are warp-uniform loops — legal under the v1 divergence limits).
- Socket sections are hart-0-gated (single active lane can never fire a
  divergence trap: JALR traps need >=2 差 targets, backward-branch traps
  need divergent taken/not-taken inside the active mask).
- KV cache int8 (k and v as produced per token position); incremental
  attention is bit-identical to golden's full-context vectorization
  because every op is causal and per-token/per-row.

### LUT tables
EXP_T / GELUD_T / RSQ_T are loaded into device RAM as constant blobs so
all 64 harts can look up concurrently (the sidecar LUT engine is a
single resource). Byte-identical to the sidecar ROMs — G3 ties ROM ==
generator; the loader serializes from the same generator.

### 64-bit arithmetic
RV32I has no 64-bit ops; sw/rt/mk64.h hand-rolls u64/i64 as hi/lo pairs
(16-bit-limb multiply; no libgcc dependency, -O0-friendly). Used by:
requant (acc*M up to 2^62), LN sums, Newton rsqrt (v*r^2 <= 2^57).

### GEMM driver
int15 activations feed the sidecar as hi7/lo7 dual GEMM (acc combined
in 64-bit before requant). K-strips of 64 chained with FLAGS.acc; the
sequencer reads the C tile once per (hi|lo) pass. Weights stay int8.

### Memory map (heap, host-serialized by host/gpt2_device.py)
A descriptor block at HEAP_BASE points at: wte int8 rows + emb (M,sh);
wpe int32; per-layer {ln gq/bq, c_attn Wq/bacc/Ms/shs, per-head sM/ssh,
oM/osh, c_proj ..., ln2, c_fc ..., gM/gsh, mproj ...}; lnf; head Wq +
hM/hsh; LUT blobs; scratch (resid i32[768], h15 hi/lo rows, C tiles,
score/e rows, KV cache [12][2][ctx][768] int8, logits q8, bar words,
token trail).

### -O0 doctrine, extended (found via ISS lockstep, 2026-07-07)
The v1 reconvergence contract (max-PC, EXACT pc==rcv match) requires
divergence joins to be fall-through-reachable by every path. -O1 jump
threading routed the not-taken side AROUND the join pc: lane 0 pended
forever (warp mask 0xFE), the warp-granular barrier hid its absence
(warp-mates write the shared flag), and the run "completed" with
garbage. Sequencer kernels are -O0 ONLY; a pc>=rcv reconvergence
amendment (ISS+RTL together) is the M20+ path if -O0 speed binds.
Other hard-won rules: mk_mask() for -(pred)&x (GCC folds it into a
__mulsi3 call); per-hart stacks 16KiB (deep -O0 frames overflowed the
old 2KiB into the neighbor hart); the Newton step in rsqrt must use
the pre-shifted 64-bit-safe form (r*diff needs ~70 bits raw).

## Quantization Study


How GPT-2 small went from +743% perplexity to **+2.44%** as pure
integer math (int8 weights, device-realizable ops only), written down
so the method — not just the result — is reusable. Companion to
[GPT-2 Integer Pipeline](#gpt-2-integer-pipeline) §2/§3 (the frozen scheme)
and the retained D-018 audit narrative. Everything here happened BEFORE the G7 gate
was declared; the bar (≤5% relative ppl vs FP32) never moved.

---

### 0. Ground rules that made this work

1. **Freeze the measurement before touching the design.** The eval
   slice (2048 text8-test tokens, sha `386e62ab532d2ab2`), the FP32
   reference (44.63 ppl, our own numpy forward), and the bar (≤5%
   relative) were fixed first. Every experiment below reports against
   the same number. No experiment was allowed to change the ruler.
2. **One metric, fast to evaluate.** Full-slice integer ppl costs ~12s
   (exact int8 GEMMs via float64 BLAS — |acc| < 2^53 so f64 matmul is
   EXACT integer arithmetic, then cast back). A 12-second loop is what
   made ~30 experiments affordable in an afternoon.
3. **Bit-frozen golden.** The chain under study IS the device contract
   (G6 = device must match it bit-for-bit), so every op had to stay
   device-realizable: no op entered the chain unless RV32I + the M18
   sidecar could compute it exactly.
4. **Calibration hygiene.** Early runs calibrated on a prefix of the
   eval slice — leakage (favorable!). Barred; the frozen protocol is
   4×1024-token windows from the text8 TRAIN region, per-channel stats
   aggregated across windows by max.

### 1. The instrumentation toolkit (build these first)

Three instruments did all the diagnostic work. In order of increasing
resolution:

**(a) Lockstep layer diff.** Run the integer chain and the FP32 chain
side by side on the same tokens; at every stage boundary print
correlation and relative RMS of (dequantized int) vs fp32. This is a
*localizer*: the first stage where corr drops is where the bug/design
flaw lives. It found: the LNC fold constant 256× off (ln corr 0.716 →
0.9996 after fix), the L5 residual collapse at long context, the g8
saturation break (corr 0.968 while everything upstream was 0.996+).
Key detail: run it at REAL context length — several defects (attention
weight underflow, calibration under-coverage) are invisible at
6-token prompts and catastrophic at T=1024.

**(b) The hybrid ablation harness.** A parallel implementation of the
forward where each quantization interface can be individually switched
to exact fp32 (`exact={"ln"}`, `{"gelu"}`, `{"qkv"}`, `{"soft"}`,
`{"o8"}`, `{"emb"}`, `{"logit"}`). Measure ppl with each interface
ablated: the drop attributable to each interface = its error budget.
This converts "the model is bad" into "LN costs 5.4 ppl, GELU 3.0,
softmax 0.7, ...". THE cardinal rule: **the hybrid's non-exact
branches must mirror the real chain's current semantics exactly** —
a stale hybrid (e.g. still modeling int8 after the chain moved to
int15) produces confidently wrong budgets, which burned an hour.
Cross-check: hybrid-with-nothing-exact must reproduce the real chain's
ppl to within noise, and all-exact must reproduce FP32 exactly (44.63).
The difference real-chain − hybrid-none = the WEIGHT quantization cost
(the hybrid keeps fp weights), a budget line you can't get any other
way.

**(c) Sub-ablation splits.** When an interface's budget is large,
split the mechanism: for LN we ran four variants — full int (real),
fp-arithmetic + int grid+clip, int-arithmetic without the saturation
clip, fully exact. Result: `noclip` ≈ `fp` (46.93 vs 46.95) while
`arith` ≈ `int` (52.35 vs 52.32) → **the entire LN cost was the clip;
arithmetic and grid were free.** That single experiment redirected the
whole design (margins, not precision).

### 2. The chronology — every hypothesis, experiment, and number

| # | Observation | Hypothesis | Experiment | Result |
|---|---|---|---|---|
| 1 | Draft scheme babbles; ppl +743% (376.4) | — | lockstep diff | ln1 corr 0.716, int LN saturating |
| 2 | LN output 256× hot | fold constant wrong | re-derive: `n = d·r·2^(16−shr)` ⇒ LNC = 65536·16/√768, not /(16·√768) | ln corr → 0.9996; ppl 376 (still) |
| 3 | Real words, wrong long-context ppl | u7 attention weights (step 1/128) zero out diffuse 1024-token rows | u14 weights; device realizes PV as TWO int8 GEMMs (hi7=p>>7, lo7=p&127, acc=(hi<<7)+lo — bit-identical to one u14 GEMM) | 376 → **176** |
| 4 | Ablation round 1 | which interface dominates? | hybrid ablation | exact-LN alone: 171 → **48.6**. Everything else combined: the rest |
| 5 | LN interfaces dominate | GPT-2's outlier channels live in LN outputs; per-tensor scales starve them | **per-channel equalization folds**: store x at per-channel scales u_c; consuming GEMM absorbs u_c into its weights BEFORE per-row quantization (W̃_cr = W_cr·u_c). Kernels unchanged; the fold is free at runtime. Same trick for v (per-channel out-scales in the qkv requant) and gelu-out | broke embedding tie (separate int8 tables for emb vs lnf-folded head) |
| 6 | Fold version WORSE (619, "the the the") | a sub-interface regressed | lockstep diff | g8 (gelu-out requant) corr 0.968, relrms 0.25 — everything else 0.996+ |
| 7 | GELU activations are sparse spikes | 99.5-percentile per-channel scale saturates exactly the rare big values that carry signal | per-channel MAX for gelu scales (dead-channel floor max/64) | 619 → **71.4**, greedy nearly right |
| 8 | +60%, need ≤5% | calibration percentile choice? | sweep pcv ∈ {99.5, 99.9, 100} | 99.9 best: **66.7** — calibration is no longer the lever |
| 9 | Ablation round 2 | re-budget | gelu −11, ln −7.6, o8 −4.8, weights ~−8 (hybrid 58.7 vs real 66.7) | |
| 10 | GELU LUT output grid is 1/32 absolute — coarser than most channels' own scales | delta form: gelu−relu ∈ [−0.17, 0] fits int8 at q9 (·512) | **GELUD table** (sidecar op 5; same hardware op, new ROM; G3 ties table==generator); `y = relu(g) + GELUD[g>>3]` — the ±4 passthrough special-case disappears at the domain edges | 66.7 → 64.2 |
| 11 | Weights cost ~8 | the α=1 fold pushes the FULL 100× channel spread into weight rows; per-row int8 rounds small-u_c columns away | **SmoothQuant-style α split**: activation scales = s_c^α·S^(1−α); sweep α | α=0.75: **55.0** (+23%) |
| 12 | rsqrt table 0.4% ≈ the grid step | LN arithmetic precision? | Newton step after the LUT seed. FIRST ATTEMPT WRONG: refining against the truncated mantissa converges to the truncation error (errors unchanged!). Fix: Newton against the FULL v | rsqrt 0.4% → 1e-4… **ppl unchanged** — arithmetic was never the binding error |
| 13 | β sweep paradox: grid margins help at one calib, hurt at another | int8 can't afford margins (resolution), but clipping persists | **int15 activations** where the consumer is a weight GEMM (LN-out, attn-out, gelu-out): dual int8 GEMM (hi7/lo7), 64-bit combine in the kernel; weights stay int8 = the Q8_0 claim. Partial accs int32-safe to K=4096 | ppl EXACTLY unchanged (53.21) — **clipping, not resolution, binds** |
| 14 | Which part of LN clips? | sub-ablation (c): int/noclip/arith/fp | **noclip 46.93 ≈ fp 46.95; arith 52.35 ≈ int 52.32** — pure clip | |
| 15 | Why did global β make it worse then? | β also shrank int8 q/k/v resolution | **margin split by width**: int15 interfaces get margin 16 (2 of their 7 spare bits — free), int8 interfaces margin 1; and int15 folds go α=1 (full coverage) → later re-tuned | margin 16: **49.3** (+10.4%), saturates by 32 |
| 16 | GELU input clips at x=64 (q9 sat16) under calibrated fc max 62.6 | move the input domain | q8 input (clip 128), 3-bit linear interp on the GELUD index (two loads, one mult, one shift — kernel arithmetic only) | 49.35 — marginal, but greedy now matches FP32 exactly |
| 17 | Scores at x/16 grid = ±3% weight noise | halve the exponent grid without touching the frozen EXP table | x/32 scores; EXP_T[z>>1] plus one multiply by exp(−1/32)·2^15 = 33406 for the odd bit; int16 scores clipped only AFTER max-subtract | (bundled with 16) |
| 18 | Ablation round 3 (faithful hybrid!) | re-budget at the new operating point | **every activation interface now ≤0.35 ppl; the remaining 4.3 = int8 weights** (hybrid 45.06 vs real 49.35) | |
| 19 | Weight-row spread again — but now int15 margins cover the activation side for free | α on the int15 folds trades ONLY weight-row spread | sweep α15 ∈ {0.75, 0.5, 0.25} | α15=0.5: **45.73 = +2.44% — UNDER THE BAR** |
| 20 | Freeze | bake defaults into code (no env knobs), freeze calibration protocol as module functions, write gates/g7_gpt2.py | G7 PASS, +2.47% first run, +2.44% after the p14 clamp re-baseline | |

### 3. The final frozen scheme (what all of that bought)

- **Weights**: per-output-row symmetric int8, quantized AFTER the
  equalization fold (consumer absorbs activation channel scales).
- **Residual stream**: int32 at a single global 2^-16 scale. Adds are
  exact; LN is scale-invariant; GPT-2's 650× per-layer residual growth
  (calibrated range 4.7 → 3045.7) costs literally nothing. This
  dissolves the famous outlier problem at the residual level.
- **Interface widths**: q/k/v int8 (scores need int8×int8); LN-out,
  attn-out, GELU-out int15 via dual-GEMM; attention weights u14
  (clamped to 16383 so hi7 fits int8); scores int16 in the x/32
  domain, clipped only after max-subtraction.
- **Scales**: per-channel 99.9-pct (GELU: max), floored at max/64,
  4×1024 train-window calibration aggregated by max; int8 interfaces
  α=0.75 margin 1; int15 interfaces α=0.5 margin 16.
- **LN**: exact int32 deltas, 64-bit sums, s0-guard only inside the
  sum of squares, RSQRT LUT seed + one Newton step against the full
  input in the 64-bit-safe pre-shifted form.
- **GELU**: `relu(g_q8) + interp(GELUD)`, delta table at q9.
- **Softmax**: EXP LUT on z>>1 + odd half-step multiply, u14 weights.
- **Head**: tied wte re-quantized separately for embedding (raw) and
  head (lnf-folded); logits q8; first-max argmax.

### 4. Design rules extracted (the transferable part)

1. **Localize, budget, then design.** Lockstep diff finds WHERE,
   ablation finds HOW MUCH, sub-ablation finds WHY. Never redesign on
   a hunch when a 12-second experiment can assign the blame.
2. **Distinguish resolution errors from clipping errors early** — the
   remedies are opposite (finer grids vs wider margins), and the
   noclip/arith split is a two-run experiment.
3. **Spend integer headroom where it's free.** int32 residual and
   int15 interfaces cost only sim-time; margins on wide interfaces
   cost nothing. Only genuinely constrained widths (GEMM operands)
   deserve α-style compromises.
4. **Folds beat runtime ops.** Every per-channel scale in the scheme
   is absorbed offline into weights or requant constants; device
   kernels never multiply by a scale vector.
5. **Watch for asymmetric/sparse distributions** (GELU): percentile
   scales are wrong exactly when the tail IS the signal.
6. **Keep the tie-break/rounding conventions frozen and boring**
   (floor divisions, round-half-up requant, first-max argmax) — G6
   holds the device to them bit-for-bit.
7. **Calibration coverage is a real failure mode**: scales must come
   from disjoint data, multiple windows, aggregated pessimistically;
   sweeping a margin knob upward and watching ppl IMPROVE is the
   smell of under-calibrated ranges (or leakage in reverse).
8. **Ablation harnesses rot.** Re-derive the hybrid from the current
   chain semantics after every scheme change, and re-verify its two
   fixed points (none-exact == real, all-exact == fp32).

### 5. Reproduction

```
.venv/bin/python golden/gpt2_int.py     # self-test: rsqrt, greedy vs fp32, ppl
.venv/bin/python gates/g7_gpt2.py       # the frozen gate (nightly)
```
Knobs (env, for study only — frozen defaults in code): MK_PCV, MK_ALPHA
(int8 α), MK_A15 (int15 α), MK_B15 (int15 margin), MK_NGEN.

## Testbench Methodology


Every RTL module in this project gets verified the same way. This is the
copy-paste skeleton; `tb/fifo`, `tb/mul_pipe`, `tb/uart` are the reference
implementations (cocotb 2.0.1 + Verilator 5.048).

### The law (R1, absolute)

No RTL merges until:
1. its golden model exists in `golden/` (pure Python, self-check in `__main__`),
2. a cocotb bench diffs RTL against golden (`tb/<module>/`),
3. the bench is picked up by nightly (automatic: `ci/run_units.sh` discovers
   every `tb/*/Makefile` exposing `smoke` and `fuzz`).

### Files per module

```
golden/<name>.py          # the oracle: model class + __main__ self-check
rtl/common/<name>.sv      # must pass ci/lint.sh: verilator -Wall, ZERO warnings
tb/<name>/Makefile        # 4 lines: DUT, TEST_MODULE, FUZZ_N, include ../common.mk
tb/<name>/test_<name>.py  # the bench: one shared run() + smoke/fuzz wrappers
```

### The per-vector protocol (memorize this — it is not optional)

```python
await drive_edge(dut)       # falling edge: the ONLY place inputs are written
dut.x.value = ...
await sample_edge(dut)      # rising edge + ReadOnly: NBA-settled outputs
exp = model.step(inputs)    # golden steps ONCE per clock, same inputs
check(int(dut.y.value) == exp["y"], ...)
```

Why: with cocotb-on-Verilator, deposits made right after a `RisingEdge`
trigger do NOT reliably land before the next edge, and reads made right at
the edge race NBA updates. Driving on the falling edge gives half a cycle of
setup; sampling in `ReadOnly` after the rising edge reads settled values.
M1's first FIFO run failed on exactly this — vector 1, write dropped. Use
`mkutil.drive_edge` / `mkutil.sample_edge`, never raw edge waits.

Compare style: **per-cycle** for pipelines and Moore outputs (fifo, mul_pipe);
**per-transaction** for handshaked/serial units (uart) — but make the golden
decoder *strict* (exact framing, per-bit stability) so a transaction compare
still pins down the timing. Don't-care outputs (empty FIFO's rd_data, invalid
pipe's p) are `None` in the model and skipped by the bench.

### Latency convention for pipelines

An input captured at edge N of an S-flop pipeline is observable at the sample
point of edge N+S-1 (the capture edge counts as flop 1). In model terms: a
step() that appends-then-pops needs a backlog of S-1 entries. M1's mul_pipe
golden had this off by one — the second bug the pattern caught (in the model,
not the RTL; the diff is symmetric and doesn't care which side is wrong).

### Seeds (ci/seeds.yaml)

- `smoke`: MK_SEED=0xC0FFEE, small N — bit-reproducible forever.
- `fuzz`: MK_SEED from `MK_FUZZ_SEED` (nightly passes epoch seconds), full N.
- NEVER name anything `RANDOM_SEED` — cocotb consumes that env var itself
  (deprecated) and dies on an empty value.
- All randomness inside a test comes from `mkutil.Rng(seed)`. No exceptions —
  a failure must replay exactly from its seed.

### Failure artifacts (the red path — verified by fault injection at M1)

On a fuzz mismatch the bench raises with vector index, seed, and a one-line
repro (`MK_SEED=0x2a make -C tb/fifo fuzz`); common.mk then rebuilds with
`VERILATOR_TRACE=1` (compile-time flag — a clean is forced) and reruns the
same seed, leaving `dump.vcd` in the tb dir. Note: cocotb 2.0's knob is
`VERILATOR_TRACE=1`, *not* `WAVES`. Green runs never keep waves (R9).

### Adding module N+1 (the whole ceremony)

1. Write `golden/<name>.py`; run `python golden/<name>.py` until the
   self-check passes.
2. Write the RTL; run `ci/lint.sh` until zero warnings.
3. Copy a reference tb dir; edit the 4-line Makefile; write run() against the
   golden model using drive_edge/sample_edge; corner vectors first, then
   phased-bias random (hammer both extremes of every flag).
4. `make -C tb/<name> smoke && make -C tb/<name> fuzz` green.
5. Done — nightly discovers it automatically. Commit RTL + golden + tb
   together, never separately.

### The trace-diff sub-pattern (program-driven DUTs — locked at M2)

CPU-class modules don't fit the stimulus/response shape above; they get the
lockstep variant instead (`tb/rv32i_core` is the reference):

- The bench is a **pure-Verilator C++ harness** (no cocotb — ~MHz, so the
  1M-instruction nightly bar costs seconds, not hours). It loads a memory
  image, runs to the tohost store, and emits the canonical commit log.
- Three producers, one format (defined in golden/iss.py): RTL harness, the
  Python ISS, and spike (parsed by golden/spikelog.py). The gate script
  (gates/p1_lockstep.py) diffs them pairwise; the bar is ZERO diffs.
- Programs come from riscv-tests (their env/p link script) and rvfuzz
  (sw/rv/link.ld — tohost below text so huge programs can't collide).
- `smoke`/`fuzz` make targets exist so ci/run_units.sh picks it up like any
  bench; the full suite runs as its own nightly stage (`p1`).
- Spike quirks encoded in golden/spikelog.py: commit lines carry a privilege
  digit (disasm lines don't); loads log `xN val mem addr`, stores log
  `mem addr val`; the 0x1000 bootrom commits are dropped until entry pc.

### Sizing

FUZZ_N is a *transaction* count, tuned so each module's fuzz stays under
~30 s in nightly: fifo/mul_pipe 10k vectors (~0.25 s), uart 400 frames.
Throughput observed on this machine: ~0.4–0.75 Mns/s sim rate for these
module sizes.
