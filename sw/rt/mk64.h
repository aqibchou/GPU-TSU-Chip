/* mk64.h — hand-rolled 64-bit integer helpers for RV32I kernels (M19).
   No libgcc: u64 as {lo, hi} pairs, 16-bit-limb multiplies. All helpers
   are branch-light and -O0-friendly; results are bit-frozen against
   golden/gpt2_int.py (numpy int64). */
#ifndef MK64_H
#define MK64_H
#include "mk.h"

typedef struct { u32 lo, hi; } u64p;

/* 16x16 -> 32 unsigned multiply, 16 uniform trips (RV32I has no MUL;
   raw * would emit a libgcc call). Same asm-barrier discipline as
   mk_mulu (runtime_spec §4). */
static inline __attribute__((always_inline)) u32 mk_mulu16(u32 a, u32 b) {
    u32 r = 0u, bb = b & 0xFFFFu;
    for (u32 t = 0u; t < 16u; t++) {
        r += a & mk_mask(bb & 1u);   /* branchless (-O1 layout!) */
        bb >>= 1;
        a <<= 1;
        __asm__ volatile("" : "+r"(bb), "+r"(a), "+r"(r));
    }
    return r;
}

static inline __attribute__((always_inline)) u64p mk64_from_i32(u32 v) {
    u64p r; r.lo = v; r.hi = (u32)((i32)v >> 31);   /* branchless */
    return r;
}

static inline __attribute__((always_inline)) u64p mk64_add(u64p a, u64p b) {
    u64p r; r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (u32)(r.lo < a.lo);   /* sltu, no branch */
    return r;
}

static inline __attribute__((always_inline)) u64p mk64_neg(u64p a) {
    u64p r; r.lo = ~a.lo + 1u;
    r.hi = ~a.hi + (u32)(r.lo == 0u);
    return r;
}

/* conditionally negate: m = 0 or 0xFFFFFFFF (branchless two's compl) */
static inline __attribute__((always_inline)) u64p mk64_cneg(u64p a, u32 m) {
    u64p r;
    r.lo = (a.lo ^ m) + (m & 1u);
    r.hi = (a.hi ^ m) + (u32)(r.lo < (a.lo ^ m));
    return r;
}

/* unsigned 32x32 -> 64, register-only asm shift-add: -O0 C spills every
   local to the stack and the ONE memory port becomes the bottleneck
   (measured 0.038 IPC); this loop touches no memory. 32 uniform trips,
   branchless body (mask = -(b&1)) — no divergence hazard. */
static __attribute__((noinline, section(".text.mk0")))
u64p mk64_mulu32(u32 a, u32 b) {
    /* noinline: at -O1 the register allocator packing this 6-operand
       asm into large inlined frames produced wrong products (the -O0
       standalone allocation was always correct); as a call it gets a
       clean register file every time */
    u32 lo, hi, cnt;
    u32 ahi = 0u;
    __asm__ volatile(
        "li   %0, 0\n\t"
        "li   %1, 0\n\t"
        "li   %2, 32\n\t"
        "1:\n\t"
        "andi t0, %4, 1\n\t"
        "neg  t0, t0\n\t"
        "and  t1, %3, t0\n\t"
        "and  t2, %5, t0\n\t"
        "add  %0, %0, t1\n\t"
        "sltu t0, %0, t1\n\t"
        "add  %1, %1, t2\n\t"
        "add  %1, %1, t0\n\t"
        "slli %5, %5, 1\n\t"
        "srli t0, %3, 31\n\t"
        "or   %5, %5, t0\n\t"
        "slli %3, %3, 1\n\t"
        "srli %4, %4, 1\n\t"
        "addi %2, %2, -1\n\t"
        "bnez %2, 1b\n\t"
        : "=&r"(lo), "=&r"(hi), "=&r"(cnt), "+r"(a), "+r"(b),
          "+r"(ahi)
        :: "t0", "t1", "t2");
    u64p r; r.lo = lo; r.hi = hi;
    return r;
}

/* signed 32x32 -> 64 (branchless sign handling: -O1 lays if/else
   sign paths out as BACKWARD branches inside loops = v1 trap) */
static inline __attribute__((always_inline)) u64p mk64_muls32(u32 a, u32 b) {
    u32 ma = (u32)((i32)a >> 31), mb = (u32)((i32)b >> 31);
    u32 aa = (a ^ ma) - ma, bb = (b ^ mb) - mb;
    return mk64_cneg(mk64_mulu32(aa, bb), ma ^ mb);
}

/* signed 64 * unsigned small (fits u32) -> 64 (used for acc*M) */
static inline __attribute__((always_inline)) u64p mk64_muls64_u32(u64p a, u32 m) {
    u32 ms = (u32)((i32)a.hi >> 31);
    u64p aa = mk64_cneg(a, ms);
    u64p lo = mk64_mulu32(aa.lo, m);
    lo.hi += mk_mulu16(aa.hi & 0xFFFFu, m)
             + (mk_mulu16(aa.hi >> 16, m) << 16);
    return mk64_cneg(lo, ms);
}

/* arithmetic shift right, 0 <= sh < 64 — BRANCHLESS: any lane-varying
   branch in shared-warp code is a jump-threading divergence hazard at
   -O1 (requant sh varies per row/lane). The (x<<1)<<(31-sh1) form is
   exact for sh1 in [0,31] and avoids the shift-by-32 UB. */
static inline __attribute__((always_inline)) u64p mk64_sra(u64p a, u32 sh) {
    u64p r;
    u32 sh1 = sh & 31u;
    u32 lo_s = (a.lo >> sh1) | ((a.hi << 1u) << (31u - sh1));
    u32 hi_s = (u32)((i32)a.hi >> sh1);
    u32 lo_b = (u32)((i32)a.hi >> sh1);
    u32 hi_b = (u32)((i32)a.hi >> 31);
    u32 m = mk_mask(sh >= 32u);
    r.lo = (lo_s & ~m) | (lo_b & m);
    r.hi = (hi_s & ~m) | (hi_b & m);
    return r;
}

static inline __attribute__((always_inline)) u64p mk64_sll(u64p a, u32 sh) {
    u64p r;
    u32 sh1 = sh & 31u;
    u32 hi_s = (a.hi << sh1) | ((a.lo >> 1u) >> (31u - sh1));
    u32 lo_s = a.lo << sh1;
    u32 hi_b = a.lo << sh1;
    u32 m = mk_mask(sh >= 32u);
    r.lo = lo_s & ~m;
    r.hi = (hi_s & ~m) | (hi_b & m);
    return r;
}

/* golden requant: (acc*M + (1<<(sh-1))) >> sh, result assumed i32 */
static inline __attribute__((always_inline)) u32 mk_requant64(u64p acc, u32 M, u32 sh) {
    u64p p = mk64_muls64_u32(acc, M);
    u64p rnd = mk64_sll(mk64_from_i32(1u), sh - 1u);
    return mk64_sra(mk64_add(p, rnd), sh).lo;
}

static inline __attribute__((always_inline)) u32 mk_satq(u32 v, i32 lim) {   /* clamp to [-lim-1,lim] */
    i32 x = (i32)v;
    u32 mh = mk_mask(x > lim);            /* predicate masks: no  */
    x = (i32)(((u32)x & ~mh) | ((u32)lim & mh));   /* arithmetic, no  */
    u32 ml = mk_mask(x < -lim - 1);            /* overflow         */
    x = (i32)(((u32)x & ~ml) | ((u32)(-lim - 1) & ml));
    return (u32)x;
}

static inline __attribute__((always_inline)) u32 mk_sat15(u32 v) { return mk_satq(v, 16383); }
static inline __attribute__((always_inline)) u32 mk_sat8(u32 v)  { return mk_satq(v, 127); }
static inline __attribute__((always_inline)) u32 mk_sat16(u32 v) { return mk_satq(v, 32767); }

/* count leading zeros (for bit_length) */
static inline __attribute__((always_inline)) u32 mk_clz(u32 v) {
    u32 n = 0u;
    if (v == 0u) return 32u;
    if (!(v & 0xFFFF0000u)) { n += 16u; v <<= 16; }
    if (!(v & 0xFF000000u)) { n += 8u; v <<= 8; }
    if (!(v & 0xF0000000u)) { n += 4u; v <<= 4; }
    if (!(v & 0xC0000000u)) { n += 2u; v <<= 2; }
    if (!(v & 0x80000000u)) { n += 1u; }
    return n;
}

static inline __attribute__((always_inline)) u32 mk64_bitlen(u64p a) {
    if (a.hi) return 64u - mk_clz(a.hi);
    return 32u - mk_clz(a.lo);
}
#endif
