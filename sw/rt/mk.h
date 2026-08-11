/* mk.h — the tiny-CUDA kernel-side API (M17, runtime_spec §3/§4).
 *
 * Identity        mk_tid() mk_warp_id() mk_lane_id()   (cf. threadIdx)
 * Work            MK_GRID_STRIDE(i, n)                 (grid-stride loop)
 * Math (RV32I!)   mk_mul mk_mulu mk_divu mk_remu mk_min mk_max mk_clamp
 * Debug           mk_log(v)          per-thread ring the host dumps
 *                 mk_assert(c,code)  host raises KernelAssert(tid, code)
 *
 * Rules of the road (v1 hardware, docs/HARDWARE_ARCHITECTURE.md#simt-core §5):
 *   - backward branches must be warp-uniform (use MK_GRID_STRIDE; the
 *     host pads grid_n to a multiple of 64 -> uniform trip counts)
 *   - forward if/else divergence fully supported (stack depth 8)
 *   - no atomics (RV32I), no inter-warp barrier: cross-warp reduction =
 *     per-thread partials + host (or a second launch)
 *   - lanes of a warp are implicitly synchronized every instruction
 *     (__syncwarp is free and implicit; there is nothing to call)
 */
#ifndef MK_H
#define MK_H
typedef unsigned int u32;
typedef int i32;
typedef unsigned char u8;

static inline __attribute__((always_inline)) u32 mk_tid(void) {
    u32 t;
    __asm__ volatile("csrr %0, mhartid" : "=r"(t));
    return t;
}
static inline __attribute__((always_inline)) u32 mk_warp_id(void) { return mk_tid() >> 3; }
static inline __attribute__((always_inline)) u32 mk_lane_id(void) { return mk_tid() & 7u; }

#define MK_PARAMS   ((volatile u32 *)0x00020000u)
#define mk_n_args() (MK_PARAMS[0])
#define mk_grid_n() (MK_PARAMS[1])
#define mk_arg(i)   (MK_PARAMS[2 + (i)])
#define MK_NTHREADS 64u

/* warp-uniform grid-stride loop: grid_n is HOST-PADDED to a multiple of
   64, so every thread runs the same trip count (uniform backedge);
   the i < (n) guard is forward divergence (supported). */
#define MK_GRID_STRIDE(i, n)                                        \
    for (u32 i = mk_tid(); i < mk_grid_n(); i += MK_NTHREADS)       \
        if (i < (n))

/* ---------------- math (no M extension in v1) ---------------- */
static inline __attribute__((always_inline)) u32 mk_mask(u32 pred) {
    /* 0/1 -> 0/0xFFFFFFFF. The asm barrier stops GCC from folding
       -(pred) & x into pred * x (= a libgcc __mulsi3 call). */
    u32 p = pred;
    __asm__ volatile("" : "+r"(p));
    return 0u - p;
}

static inline __attribute__((always_inline)) u32 mk_mulu(u32 a, u32 b) {
    /* The asm barrier stops GCC from strength-reducing this into a
       while(b)-style EARLY-EXIT loop: with lane-varying b that becomes a
       divergent BACKWARD branch = v1 trap (learned the hard way; see
       runtime_spec §4). Bound stays uniform: exactly 32 trips. */
    u32 r = 0, bb = b;
    for (u32 t = 0; t < 32; t++) {
        if (bb & 1u) r += a;
        bb >>= 1;
        a <<= 1;
        __asm__ volatile("" : "+r"(bb), "+r"(a), "+r"(r));
    }
    return r;
}
static inline __attribute__((always_inline)) i32 mk_mul(i32 a, i32 b) { return (i32)mk_mulu((u32)a, (u32)b); }
static inline __attribute__((always_inline)) u32 mk_divu(u32 a, u32 b) {
    u32 q = 0, r = 0;
    u32 mb = mk_mask(b != 0u);
    for (i32 t = 31; t >= 0; t--) {              /* uniform 32 trips */
        u32 m;
        r = (r << 1) | ((a >> (u32)t) & 1u);
        q <<= 1;
        m = mk_mask(r >= b) & mb;                /* branchless: -O1   */
        r -= b & m;                              /* rotates lane-vary */
        q |= 1u & m;                             /* ing ifs backward  */
        __asm__ volatile("" : "+r"(q), "+r"(r));
    }
    return (q & mb) | ~mb;
}
static inline __attribute__((always_inline)) u32 mk_remu(u32 a, u32 b) {
    return b ? a - mk_mulu(mk_divu(a, b), b) : a;
}
static inline __attribute__((always_inline)) i32 mk_min(i32 a, i32 b) { return a < b ? a : b; }
static inline __attribute__((always_inline)) i32 mk_max(i32 a, i32 b) { return a > b ? a : b; }
static inline __attribute__((always_inline)) i32 mk_clamp(i32 v, i32 lo, i32 hi) {
    return mk_min(mk_max(v, lo), hi);
}

/* ---------------- debug (host-visible, runtime_spec map) ------------ */
#define MK_LOG_BASE    0x00020200u
#define MK_LOG_SLOTS   32u
#define MK_ASSERT_BASE 0x00022200u

static inline __attribute__((always_inline)) void mk_log(u32 v) {
    volatile u32 *ring =
        (volatile u32 *)(MK_LOG_BASE + mk_tid() * 4u * (MK_LOG_SLOTS + 1u));
    u32 n = ring[0];
    if (n < MK_LOG_SLOTS) { ring[1 + n] = v; ring[0] = n + 1u; }
}
static inline __attribute__((always_inline)) void mk_assert(int cond, u32 code) {
    if (!cond) {
        volatile u32 *slot = (volatile u32 *)(MK_ASSERT_BASE + mk_tid() * 4u);
        if (*slot == 0u) *slot = code ? code : 1u;
    }
}
/* ---------------- D4 tensor socket (tensor_spec §1) ---------------- */
#define MK_T_GEMM8   0u
#define MK_T_GELU    1u
#define MK_T_EXP     2u
#define MK_T_SIGMOID 3u
#define MK_T_RSQRT   4u

#define _MK_CSRW(csr, v) \
    __asm__ volatile("csrw " #csr ", %0" :: "r"(v))
static inline __attribute__((always_inline)) u32 mk_t_busy(void) {
    u32 v;
    __asm__ volatile("csrr %0, 0x8C9" : "=r"(v));
    return v;
}
/* free-running cycle counter (warp-shared read; S7.4 timing) */
static inline __attribute__((always_inline)) u32 mk_mcycle(void) {
    u32 v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}
/* issue a sidecar command and wait (uniform poll). ONE thread should
   issue (e.g. if (mk_tid()==0)) with others parked on mk_t_wait(). */
static inline __attribute__((always_inline)) void mk_t_cmd(u32 op, u32 a, u32 b, u32 c,
                            u32 m, u32 n, u32 k, u32 flags) {
    _MK_CSRW(0x8C0, op);
    _MK_CSRW(0x8C1, a);
    _MK_CSRW(0x8C2, b);
    _MK_CSRW(0x8C3, c);
    _MK_CSRW(0x8C4, m);
    _MK_CSRW(0x8C5, n);
    _MK_CSRW(0x8C6, k);
    _MK_CSRW(0x8C7, flags);
    _MK_CSRW(0x8C8, 1u);
}
static inline __attribute__((always_inline)) void mk_t_wait(void) {
    while (mk_t_busy()) { }
}
/* D-032d (tensor_spec §1b): the command queue. T_STATUS bit1 = FULL.
   mk_t_post waits only for QUEUE CAPACITY, never for completion —
   chain ops back-to-back, then mk_t_wait() once at the end. */
static inline __attribute__((always_inline)) u32 mk_t_full(void) {
    return (mk_t_busy() >> 1) & 1u;
}
static inline __attribute__((always_inline)) void mk_t_post(u32 op, u32 a, u32 b, u32 c,
                            u32 m, u32 n, u32 k, u32 flags) {
    while (mk_t_full()) { }
    mk_t_cmd(op, a, b, c, m, n, k, flags);
}
static inline __attribute__((always_inline)) void mk_gemm8(u32 a, u32 b, u32 c, u32 m, u32 n, u32 k,
                            u32 acc) {
    mk_t_cmd(MK_T_GEMM8, a, b, c, m, n, k, acc);
    mk_t_wait();
}
static inline __attribute__((always_inline)) void mk_lut(u32 op, u32 src, u32 dst, u32 count) {
    mk_t_cmd(op, src, 0, dst, count, 0, 0, 0);
    mk_t_wait();
}

/* ---------------- sampling ISA (M21, docs/HARDWARE_ARCHITECTURE.md#sampling-isa) ------
   Same doorbell discipline as the tensor ops: issue from ONE thread
   (tid 0 by convention), everyone else parks on the uniform poll. */
#define MK_S_PCONFIG 8u
#define MK_S_PSAMPLE 9u
#define MK_S_PDRAIN  10u

static inline __attribute__((always_inline)) void
mk_pconfig(u32 image_addr, u32 image_words, u32 flags) {
    mk_t_cmd(MK_S_PCONFIG, image_addr, 0u, 0u, image_words, 0u, 0u, flags);
    mk_t_wait();
}
/* IMM form: sweeps>=1 at beta (u2.6). SCHED form: sweeps=0, beta=0. */
static inline __attribute__((always_inline)) void
mk_psample(u32 sweeps, u32 beta_raw, u32 flags) {
    mk_t_cmd(MK_S_PSAMPLE, 0u, 0u, 0u, sweeps, 0u, beta_raw, flags);
    mk_t_wait();
}
static inline __attribute__((always_inline)) void
mk_pdrain(u32 dest_addr, u32 mode) {
    mk_t_cmd(MK_S_PDRAIN, 0u, 0u, dest_addr, 0u, 0u, 0u, mode);
    mk_t_wait();
}

#endif
