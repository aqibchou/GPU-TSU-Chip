/* gpt2_seq.c — the M19 on-device GPT-2 sequencer (gpt2_spec §3,
   docs/SOFTWARE_AND_VALIDATION.md#gpt-2-device-sequencer-notes). One launch runs prefill + greedy
   generation; all 64 harts cooperate (rows split by tid, sense-
   reversing barriers between phases); hart 0 alone drives the D4
   socket (single active lane -> no divergence trap can fire).
   Every value produced here must equal golden/gpt2_int.py bit-for-bit
   (G6). args: desc_ptr. */
#include "mk.h"
#include "mk64.h"

#define NL 12u
#define DM 768u
#define HD 64u
#define NH 12u

/* descriptor word indices (host/gpt2_device.py serializes) */
#define D_MAGIC    0u
#define D_NPROMPT  1u
#define D_NGEN     2u
#define D_CTX      3u   /* padded ctx capacity (32) */
#define D_PROMPT   4u   /* ptr: u32 prompt ids */
#define D_WTE      5u   /* ptr: int8 [50257][768] */
#define D_EMBM     6u   /* ptr: u32 [50257] */
#define D_EMBSH    7u   /* ptr: u32 [50257] */
#define D_WPE      8u   /* ptr: i32 [ctx][768] */
#define D_LAYERS   9u   /* ptr: u32 layer table, 12 * L_WORDS */
#define D_LNFG     10u  /* ptr: i32 [768] gq */
#define D_LNFB     11u  /* ptr: i32 [768] bq */
#define D_HEADW    12u  /* ptr: int8 tiles [12kb][786nb][64][64] */
#define D_HM       13u  /* ptr: u32 [50304] */
#define D_HSH      14u  /* ptr: u32 [50304] */
#define D_NVOCAB   15u  /* 50257 */
#define D_NVPAD    16u  /* 50304 */
#define D_EXP      17u  /* ptr: u32 [256] EXP_T */
#define D_GELUD    18u  /* ptr: i32 [256] GELUD_T */
#define D_RSQ      19u  /* ptr: u32 [256] RSQ_T */
#define D_SCRATCH  20u  /* ptr: scratch block, kernel-defined layout */
#define D_TRAIL    21u  /* ptr: u32 [n_gen] output token ids */
#define D_DBG      22u  /* ptr: i32 debug dump buffer (0 = off) */

/* per-layer table word indices */
#define L_LN1G 0u
#define L_LN1B 1u
#define L_WQKV 2u   /* int8 tiles [12kb][36nb][64][64] */
#define L_BQKV 3u   /* i32 [2304] acc-unit bias */
#define L_MQKV 4u   /* u32 [2304] */
#define L_SHQKV 5u  /* u32 [2304] */
#define L_SM   6u   /* u32 [12] per-head score M */
#define L_SSH  7u   /* u32 [12] */
#define L_OM   8u   /* u32 [768] */
#define L_OSH  9u   /* u32 [768] */
#define L_WPRJ 10u  /* int8 tiles [12kb][12nb][64][64] */
#define L_BPRJ 11u
#define L_MPRJ 12u
#define L_SHPRJ 13u
#define L_LN2G 14u
#define L_LN2B 15u
#define L_WFC  16u  /* int8 tiles [12kb][48nb][64][64] */
#define L_BFC  17u
#define L_MFC  18u
#define L_SHFC 19u
#define L_GM   20u  /* u32 [3072] */
#define L_GSH  21u  /* u32 [3072] */
#define L_WMP  22u  /* int8 tiles [48kb][12nb][64][64] */
#define L_BMP  23u
#define L_MMP  24u
#define L_SHMP 25u
#define L_WORDS 26u

/* scratch layout (word offsets inside D_SCRATCH) */
#define S_BAR    0u                    /* u32 [64] barrier flags */
#define S_PHASE  64u                   /* u32 [64] per-hart phase */
#define S_X      128u                  /* i32 [768] residual */
#define S_H      (S_X + DM)            /* i32 [768] ln out (int15) */
#define S_AHI    (S_H + DM)            /* int8 [3072] hi7 row (mproj is
                                          3072 wide — 768-byte sizing
                                          overran into ALO/CHI and fed
                                          the mproj GEMM garbage A) */
#define S_ALO    (S_AHI + 768u)        /* int8 [3072] lo7 row */
#define S_CHI    (S_ALO + 768u)        /* i32 [50304] hi acc (head!) */
#define S_CLO    (S_CHI + 50304u)      /* i32 [50304] lo acc */
#define S_ROW    (S_CLO + 50304u)      /* i32 [3072] requant out / g16 */
#define S_QKV    (S_ROW + 3072u)       /* i32 [2304] qkv int8 values */
#define S_O      (S_QKV + 2304u)       /* i32 [768] attn out int15 */
#define S_RED    (S_O + DM)            /* u32 [192] hart partials */
#define S_PUB    (S_RED + 192u)        /* u32 [8] hart0 publishes */
#define S_LOGIT  (S_PUB + 8u)          /* i32 [50304] q8 logits */
#define S_KV     (S_LOGIT + 50304u)    /* KV cache, see kv_base() */

static volatile u32 *DESC;
#define DW(i) (DESC[i])

static inline i32 i8_ext(u8 v) { return (i32)(v ^ 128u) - 128; }

#define MK_NOINLINE __attribute__((noinline))
/* Compilation contract (docs/SOFTWARE_AND_VALIDATION.md#gpt-2-device-sequencer-notes): the FILE compiles
   at -O1; functions holding LANE-VARYING branches (the tid guards)
   are pinned to -O0 (GCC duplicates if-continuations at -O1, and a
   cloned continuation runs under partial mask = lanes skip real work
   — the ISS traced it at the B-beacon guard). Leaf functions are
   branchless or warp-uniform inside, so -O1 clones are harmless. */
/* .text.main links LAST: the v1 reconvergence contract requires every
   call made inside a divergent region to target a LOWER address (a
   callee above the join reads as premature arrival -> teleport merge;
   the ISS traced lane-0 ra pointing into h0_log while lanes 1-7 sat
   in requant64). Structure functions therefore live at the top of the
   address space, leaves below them. */
#define MK_STRUCT __attribute__((noinline, optimize("O0"), \
                                 section(".text.main")))
#define MK_LEAF   __attribute__((noinline))

/* -O0 lowers every * to __mulsi3 (libgcc's early-exit loop = divergent
   backward branch with lane-varying operands). Shift-adds instead. */
#define X2(v)     ((v) << 1)
#define MUL12(v)  (((v) << 3) + ((v) << 2))
#define MUL26(v)  (((v) << 4) + ((v) << 3) + ((v) << 1))
#define MUL48(v)  (((v) << 5) + ((v) << 4))
#define MUL768(v) (((v) << 9) + ((v) << 8))

static inline volatile u32 *scr(void) {
    return (volatile u32 *)DW(D_SCRATCH);
}

/* ---- barrier: WARP-granular (lanes are implicitly synced, so all 8
   lanes store the same value to their warp's flag — one coalesced
   beat — and poll only 8 words). Phase words per hart avoid a shared
   global. */
static MK_STRUCT void mk_barrier(void) {
    volatile u32 *bar = scr() + S_BAR;
    volatile u32 *ph = scr() + S_PHASE;
    u32 t = mk_tid();
    u32 p = ph[t] + 1u;
    ph[t] = p;
    bar[mk_warp_id()] = p;
    for (u32 w = 0u; w < 8u; w++)
        while (bar[w] < p) {
            /* ALU backoff: waiting warps otherwise saturate the single
               memory port with poll loads and starve the sidecar DMA
               (measured: 139M beats / 22M mem commits, TBUSY 3%) */
            for (u32 d = 0u; d < 64u; d++)
                __asm__ volatile("");
        }
}

/* ---- KV cache addressing.
   kT (K transposed) per (layer, head): int8 [64 j][ctx pos] bytes.
   v   per (layer, head): int8 [ctx pos][64 j] bytes.               */
static inline volatile u8 *kT_base(u32 li, u32 hd) {
    u32 ctx = DW(D_CTX);
    return (volatile u8 *)(scr() + S_KV)
           + (mk_mulu16((MUL12(li) + hd) << 6, ctx));
}
static inline volatile u8 *v_base(u32 li, u32 hd) {
    u32 ctx = DW(D_CTX);
    return (volatile u8 *)(scr() + S_KV) + mk_mulu16(9216u, ctx)
           + (mk_mulu16(MUL12(li) + hd, ctx) << 6);
}


/* Divergent-region bodies live in NOINLINE functions: `if (cond) f();`
   leaves the optimizer nothing to jump-thread, so the reconvergence
   point is always the instruction after the call. -O1's tail
   duplication otherwise gives the two sides disjoint code copies and
   the join drifts to the function epilogue (the lane-0 orphan the ISS
   traced). Single-active-lane callees are internally trap-immune;
   softmax_head (12 lanes) keeps its lane-varying ops branchless. */
static MK_LEAF void h0_log(u32 v) { mk_log(v); }

static MK_LEAF void ln_mean_combine(volatile u32 *red,
                                        volatile u32 *pub) {
    u64p sm = mk64_from_i32(0u);
    for (u32 h = 0u; h < 64u; h++) {
        u64p p2; p2.lo = red[X2(h)]; p2.hi = red[X2(h) + 1u];
        sm = mk64_add(sm, p2);
    }
    u64p m = mk64_muls64_u32(sm, 21845u);
    pub[0] = mk64_sra(m, 24u).lo;
}

static MK_LEAF void ln_s0_combine(volatile u32 *red,
                                      volatile u32 *pub) {
    u32 m2 = 0u;
    for (u32 h = 0u; h < 64u; h++) if (red[h] > m2) m2 = red[h];
    u32 bl = 32u - mk_clz(m2);
    pub[1] = (bl > 26u) ? (bl - 26u) : 0u;
}

static MK_LEAF void ln_rsqrt_combine(volatile u32 *red,
                                         volatile u32 *pub) {
    u64p vs = mk64_from_i32(0u);
    for (u32 h = 0u; h < 64u; h++) {
        u64p p2; p2.lo = red[X2(h)]; p2.hi = red[X2(h) + 1u];
        vs = mk64_add(vs, p2);
    }
    if (vs.lo == 0u && vs.hi == 0u) vs.lo = 1u;
    u32 pre = 0u, blen = mk64_bitlen(vs);
    if (blen > 31u) {
        pre = blen - 31u;
        if (pre & 1u) pre += 1u;
        vs = mk64_sra(vs, pre);
    }
    u32 v = vs.lo;
    u32 msb = 31u - mk_clz(v);
    i32 t = (i32)msb - 7;
    if (t & 1) t += 1;
    u32 b = (t >= 0) ? (v >> (u32)t) : (v << (u32)(-t));
    if (b < 1u) b = 1u;
    if (b > 255u) b = 255u;
    u32 r = ((volatile u32 *)DW(D_RSQ))[b - 1u];
    u32 k = 16u + (u32)(t / 2);
    u64p r2 = mk64_mulu32(r, r);
    u64p vr2 = mk64_mulu32(r2.lo, v);
    u64p three = mk64_sll(mk64_from_i32(3u), 2u * k);
    u64p diff = mk64_add(three, mk64_neg(vr2));
    u32 se = (2u * k > 47u) ? (2u * k - 47u) : 0u;
    u64p dsh = mk64_sra(diff, se);
    u64p p1 = mk64_mulu32(dsh.lo, r);
    p1.hi += mk_mulu16(dsh.hi & 0xFFFFu, r & 0xFFFFu)
             + (mk_mulu16(dsh.hi & 0xFFFFu, r >> 16) << 16);
    r = mk64_sra(p1, 2u * k + 1u - se).lo;
    pub[2] = r;
    pub[3] = k + pre / 2u;
}

static MK_LEAF void gemm_drive(u32 w_ptr, u32 kb_n, u32 nb_n,
                                   volatile i32 *chi, volatile i32 *clo) {
    u32 a_hi = (u32)(scr() + S_AHI);
    u32 a_lo = (u32)(scr() + S_ALO);
    for (u32 nb = 0u; nb < nb_n; nb++) {
        for (u32 kb = 0u; kb < kb_n; kb++) {
            u32 bt = w_ptr + ((mk_mulu16(kb, nb_n) + nb) << 12);
            mk_gemm8(a_hi + (kb << 6), bt, (u32)(chi + (nb << 6)),
                     1u, 64u, 64u, kb ? 1u : 0u);
        }
        for (u32 kb = 0u; kb < kb_n; kb++) {
            u32 bt = w_ptr + ((mk_mulu16(kb, nb_n) + nb) << 12);
            mk_gemm8(a_lo + (kb << 6), bt, (u32)(clo + (nb << 6)),
                     1u, 64u, 64u, kb ? 1u : 0u);
        }
    }
}

static MK_LEAF void attn_scores_drive(volatile i32 *qkv, u32 li,
                                          u32 ctx, volatile i32 *chi) {
    volatile u8 *qa = (volatile u8 *)(scr() + S_H);
    for (u32 i = 0u; i < DM; i++)
        qa[i] = (u8)qkv[i];
    for (u32 hd = 0u; hd < NH; hd++)
        mk_gemm8((u32)(qa + (hd << 6)), (u32)kT_base(li, hd),
                 (u32)(chi + mk_mulu16(hd, ctx)), 1u, ctx, HD, 0u);
}

static MK_LEAF void softmax_head(u32 tid, u32 ctx, u32 len,
                                     volatile i32 *clo, volatile u32 *p14,
                                     volatile u8 *ph8, volatile u8 *pl8) {
    volatile u32 *expt = (volatile u32 *)DW(D_EXP);
    volatile i32 *sc_ = clo + mk_mulu16(tid, ctx);
    volatile u32 *pw = p14 + mk_mulu16(tid, ctx);
    i32 m = sc_[0];
    for (u32 i = 1u; i < len; i++) {
        u32 mg = mk_mask(sc_[i] > m);
        m = (i32)(((u32)m & ~mg) | ((u32)sc_[i] & mg));
    }
    u32 sum = 0u;
    for (u32 i = 0u; i < len; i++) {
        i32 z = sc_[i] - m;
        u32 mz = mk_mask(z < -255);
        u32 e, e2, mo;
        z = (i32)(((u32)z & ~mz) | ((u32)(-255) & mz));
        e = expt[(u32)((z >> 1) + 128)];
        e2 = mk_mulu16(e, 33406u) >> 15;
        mo = mk_mask(z & 1);
        e = (e & ~mo) | (e2 & mo);
        pw[i] = e;
        sum += e;
    }
    sum += (u32)(sum == 0u);           /* branchless min-1 */
    for (u32 i = 0u; i < ctx; i++) {
        u32 v = 0u;
        if (i < len) {                 /* len uniform among active */
            v = mk_divu(pw[i] << 14, sum);
            u32 mgt = mk_mask(v > 16383u);
            v = (v & ~mgt) | (16383u & mgt);
        }
        ph8[mk_mulu16(tid, ctx) + i] = (u8)(v >> 7);
        pl8[mk_mulu16(tid, ctx) + i] = (u8)(v & 127u);
    }
}

static MK_LEAF void attn_pv_drive(volatile u8 *ph8, volatile u8 *pl8,
                                      u32 li, u32 ctx,
                                      volatile i32 *chi, volatile i32 *clo) {
    for (u32 hd = 0u; hd < NH; hd++) {
        mk_gemm8((u32)(ph8 + mk_mulu16(hd, ctx)), (u32)v_base(li, hd),
                 (u32)(chi + 4096u + (hd << 6)), 1u, HD, ctx, 0u);
        mk_gemm8((u32)(pl8 + mk_mulu16(hd, ctx)), (u32)v_base(li, hd),
                 (u32)(clo + 4096u + (hd << 6)), 1u, HD, ctx, 0u);
    }
}

static MK_LEAF void argmax_combine(volatile u32 *red,
                                       volatile u32 *trail, u32 slot) {
    i32 gm = (i32)0x80000000; u32 gi = 0u;
    for (u32 hh = 0u; hh < 64u; hh++) {
        i32 m2 = (i32)red[X2(hh)];
        u32 i2 = red[X2(hh) + 1u];
        if (m2 > gm || (m2 == gm && i2 < gi)) { gm = m2; gi = i2; }
    }
    trail[slot] = gi;
    mk_log(gi);
}

/* ---- layernorm (golden int_layernorm, D-018 exact-delta form) ---- */
static MK_LEAF void ln_part_sum(volatile i32 *x, volatile u32 *red,
                                u32 tid) {
    u64p ps = mk64_from_i32(0u);
    for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++)
        ps = mk64_add(ps, mk64_from_i32((u32)x[i]));
    red[X2(tid)] = ps.lo; red[X2(tid) + 1u] = ps.hi;
}

static MK_LEAF void ln_part_d(volatile i32 *x, volatile i32 *out,
                              volatile u32 *red, u32 tid, i32 mean) {
    u32 mx = 0u;
    for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++) {
        i32 d = x[i] - mean;
        u32 ms = (u32)(d >> 31);
        u32 ad = ((u32)d ^ ms) - ms;
        u32 mg = mk_mask(ad > mx);
        out[i] = d;
        mx = (mx & ~mg) | (ad & mg);
    }
    red[tid] = mx;
}

static MK_LEAF void ln_part_sq(volatile i32 *out, volatile u32 *red,
                               u32 tid, u32 s0) {
    u64p ps = mk64_from_i32(0u);
    for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++) {
        i32 ds = out[i] >> s0;
        u32 msk = (u32)(ds >> 31);
        u32 a = ((u32)ds ^ msk) - msk;
        ps = mk64_add(ps, mk64_mulu32(a, a));
    }
    red[X2(tid)] = ps.lo; red[X2(tid) + 1u] = ps.hi;
}

static MK_LEAF void ln_part_norm(volatile i32 *out, const i32 *gq,
                                 const i32 *bq, u32 tid, u32 r,
                                 i32 shn) {
    for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++) {
        u64p n64 = mk64_muls32((u32)out[i], r);
        u64p n = (shn >= 0) ? mk64_sra(n64, (u32)shn)
                            : mk64_sll(n64, (u32)(-shn));
        u32 gneg = (u32)(gq[i] >> 31);
        u32 gabs = ((u32)gq[i] ^ gneg) - gneg;
        u64p yg = mk64_muls64_u32(n, gabs);
        yg = mk64_cneg(yg, gneg);
        yg = mk64_add(yg, mk64_sll(mk64_from_i32(1u), 19u));
        i32 y = (i32)mk64_sra(yg, 20u).lo + bq[i];
        out[i] = (i32)mk_sat15((u32)y);
    }
}

static MK_LEAF void gemm_combine(volatile i32 *chi, volatile i32 *clo,
                                 const i32 *bacc, const u32 *Ms,
                                 const u32 *shs, volatile i32 *out,
                                 u32 sat, u32 tid, u32 nN) {
    for (u32 i = mk_mulu16(tid, nN >> 6);
         i < mk_mulu16(tid + 1u, nN >> 6); i++) {
        u64p acc = mk64_sll(mk64_from_i32((u32)chi[i]), 7u);
        acc = mk64_add(acc, mk64_from_i32((u32)clo[i]));
        if (bacc) acc = mk64_add(acc, mk64_from_i32((u32)bacc[i]));
        u32 y = mk_requant64(acc, Ms[i], shs[i]);
        out[i] = (i32)(sat == 0u ? mk_sat8(y)
                       : sat == 1u ? mk_sat15(y)
                       : sat == 2u ? mk_sat16(y) : y);
    }
}

static MK_STRUCT void ln_int(volatile i32 *x, const i32 *gq, const i32 *bq,
                   volatile i32 *out) {
    u32 tid = mk_tid();
    volatile u32 *red = scr() + S_RED;
    volatile u32 *pub = scr() + S_PUB;
    ln_part_sum(x, red, tid);
    mk_barrier();
    if (tid == 0u) ln_mean_combine(red, pub);
    mk_barrier();
    i32 mean = (i32)pub[0];
    ln_part_d(x, out, red, tid, mean);
    mk_barrier();
    if (tid == 0u) ln_s0_combine(red, pub);
    mk_barrier();
    u32 s0 = pub[1];
    ln_part_sq(out, red, tid, s0);
    mk_barrier();
    if (tid == 0u) ln_rsqrt_combine(red, pub);
    mk_barrier();
    u32 r = pub[2], k = pub[3];
    i32 shn = (i32)(k + s0) - 16;
    ln_part_norm(out, gq, bq, tid, r, shn);
    mk_barrier();
}

/* ---- hi/lo split of an int15 row into word-packed int8 arrays ---- */
static MK_LEAF void split_row(volatile i32 *h, u32 n) {
    u32 tid = mk_tid();
    volatile u8 *hi = (volatile u8 *)(scr() + S_AHI);
    volatile u8 *lo = (volatile u8 *)(scr() + S_ALO);
    for (u32 i = mk_mulu16(tid, n >> 6); i < mk_mulu16(tid + 1u, n >> 6); i++) {
        i32 v = h[i];
        hi[i] = (u8)(i32)(v >> 7);
        lo[i] = (u8)(v & 127);
    }
    mk_barrier();
}

/* ---- dual-GEMM row driver: A (1 x K int15, split) x W tiles ->
   requant to out[] with per-row (M, sh), optional bias, sat.
   W tiles laid out [kb][nb][64][64]; nb_total = N/64.
   sat: 0 = sat8 (int8 out), 1 = sat15, 2 = sat16 (gelu domain q8). */
static MK_STRUCT void gemm_row(u32 w_ptr, u32 nK, u32 nN,
                     const i32 *bacc, const u32 *Ms, const u32 *shs,
                     volatile i32 *out, u32 sat) {
    u32 tid = mk_tid();
    u32 kb_n = nK / 64u, nb_n = nN / 64u;
    volatile i32 *chi = (volatile i32 *)(scr() + S_CHI);
    volatile i32 *clo = (volatile i32 *)(scr() + S_CLO);
    if (tid == 0u) gemm_drive(w_ptr, kb_n, nb_n, chi, clo);
    mk_barrier();
    gemm_combine(chi, clo, bacc, Ms, shs, out, sat, tid, nN);
    mk_barrier();
}

__attribute__((optimize("O0"), section(".text.main")))
void kernel_main(u32 tid) {
    DESC = (volatile u32 *)mk_arg(0);
    u32 n_prompt = DW(D_NPROMPT), n_gen = DW(D_NGEN);
    u32 ctx = DW(D_CTX);
    volatile i32 *x = (volatile i32 *)(scr() + S_X);
    volatile i32 *h = (volatile i32 *)(scr() + S_H);
    volatile i32 *qkv = (volatile i32 *)(scr() + S_QKV);
    volatile i32 *row = (volatile i32 *)(scr() + S_ROW);
    volatile i32 *o = (volatile i32 *)(scr() + S_O);
    volatile u32 *trail = (volatile u32 *)DW(D_TRAIL);
    volatile u32 *layers = (volatile u32 *)DW(D_LAYERS);
    u32 total = n_prompt + n_gen;

    for (u32 step = 0u; step < total - 1u; step++) {
        u32 tok = (step < n_prompt) ? ((volatile u32 *)DW(D_PROMPT))[step]
                                    : trail[step - n_prompt];
        if (tid == 0u) h0_log(0xA0000000u | step);
        /* ---- embedding: x = requant(wte8[tok]) + wpe[step] ---- */
        {
            volatile u8 *wr = (volatile u8 *)DW(D_WTE) + MUL768(tok);
            u32 M = ((volatile u32 *)DW(D_EMBM))[tok];
            u32 sh = ((volatile u32 *)DW(D_EMBSH))[tok];
            volatile i32 *wpe = (volatile i32 *)DW(D_WPE) + MUL768(step);
            for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++) {
                i32 w8 = (i32)(i8_ext(wr[i]));
                u64p a = mk64_from_i32((u32)w8);
                x[i] = (i32)mk_requant64(a, M, sh) + wpe[i];
            }
        }
        mk_barrier();

        for (u32 li = 0u; li < NL; li++) {
            volatile u32 *L = layers + MUL26(li);
            if (tid == 0u && step == 0u) h0_log(0xB0000000u | li);
            /* ln1 -> h (int15) */
            ln_int(x, (const i32 *)L[L_LN1G], (const i32 *)L[L_LN1B], h);
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000001u);
            split_row(h, DM);
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000002u);
            /* qkv = gemm(h, Wqkv) -> int8 in qkv[] */
            gemm_row(L[L_WQKV], DM, 2304u, (const i32 *)L[L_BQKV],
                     (const u32 *)L[L_MQKV], (const u32 *)L[L_SHQKV],
                     qkv, 0u);
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000003u);
            /* append k, v to the caches (all harts split 768) */
            for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++) {
                u32 hd = i / HD, j = i % HD;
                kT_base(li, hd)[mk_mulu16(j, ctx) + step] = (u8)qkv[DM + i];
                v_base(li, hd)[(step << 6) + j] = (u8)qkv[(DM << 1) + i];
            }
            mk_barrier();
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000004u);
            /* attention (D-018 layout): socket phases on hart 0,
               requant split 64-way, softmax one HEAD per hart (12 of
               64), PV combine split 64-way — 6 barriers, not 36 */
            {
                volatile i32 *chi = (volatile i32 *)(scr() + S_CHI);
                volatile i32 *clo = (volatile i32 *)(scr() + S_CLO);
                volatile u32 *p14 = (volatile u32 *)(scr() + S_ROW);
                volatile u8 *ph8 = (volatile u8 *)(scr() + S_AHI);
                volatile u8 *pl8 = (volatile u8 *)(scr() + S_ALO);
                u32 len = step + 1u;
                if (tid == 0u)
                    attn_scores_drive(qkv, li, ctx, chi);
                mk_barrier();
                /* B: score requant, 12*len elements split by tid */
                {
                    /* grid-stride with PADDED bound: a raw e < tot
                       backedge trips lane-unevenly (12*len is not a
                       multiple of 64) = divergent backward branch —
                       the exact v1 trap MK_GRID_STRIDE's host padding
                       exists to prevent. Guard is forward divergence. */
                    u32 tot = mk_mulu16(NH, len);
                    u32 totp = (tot + 63u) & ~63u;
                    for (u32 e = tid; e < totp; e += 64u) {
                        /* dead lanes redirect to a dump slot: keeps
                           the body unconditional (no lane-varying
                           branch for -O1 to lay out backward) */
                        u32 live = mk_mask(e < tot);
                        u32 e2 = e & live;
                        u32 hd = mk_divu(e2, len);
                        u32 pp2 = e2 - mk_mulu16(hd, len);
                        u32 off = mk_mulu16(hd, ctx) + pp2;
                        u32 doff = (NH << 6);       /* past live rows */
                        u32 sM = ((volatile u32 *)L[L_SM])[hd];
                        u32 ssh = ((volatile u32 *)L[L_SSH])[hd];
                        u64p a = mk64_from_i32((u32)chi[off]);
                        clo[(off & live) | (doff & ~live)] =
                            (i32)mk_sat16(mk_requant64(a, sM, ssh));
                    }
                }
                mk_barrier();
                /* C: softmax, one head per hart (uniform loops) */
                if (tid < NH)
                    softmax_head(tid, ctx, len, clo, p14, ph8, pl8);
                mk_barrier();
                if (tid == 0u)
                    attn_pv_drive(ph8, pl8, li, ctx, chi, clo);
                mk_barrier();
                /* E: o combine + per-channel requant, 768 split */
                for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++) {
                    u64p acc = mk64_sll(
                        mk64_from_i32((u32)chi[4096u + i]), 7u);
                    acc = mk64_add(acc,
                                   mk64_from_i32((u32)clo[4096u + i]));
                    u32 oM = ((volatile u32 *)L[L_OM])[i];
                    u32 osh = ((volatile u32 *)L[L_OSH])[i];
                    o[i] = (i32)mk_sat15(mk_requant64(acc, oM, osh));
                }
                mk_barrier();
            }
            if (li == 0u && step == 0u && DW(D_DBG)) {
                volatile i32 *dbg = (volatile i32 *)DW(D_DBG);
                for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++)
                    dbg[i] = o[i];                 /* [0..768) attn o */
            }
            /* c_proj: o (int15) -> residual add */
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000005u);
            split_row(o, DM);
            gemm_row(L[L_WPRJ], DM, DM, (const i32 *)L[L_BPRJ],
                     (const u32 *)L[L_MPRJ], (const u32 *)L[L_SHPRJ],
                     row, 3u /* no sat: raw i32 resid units */);
            for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++)
                x[i] += row[i];
            mk_barrier();
            if (li == 0u && step == 0u && DW(D_DBG)) {
                volatile i32 *dbg = (volatile i32 *)DW(D_DBG);
                for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++)
                    dbg[768u + i] = x[i];          /* [768..1536) x+attn */
            }
            /* ln2 -> h; c_fc -> g16 (q8 domain, sat16) */
            ln_int(x, (const i32 *)L[L_LN2G], (const i32 *)L[L_LN2B], h);
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000006u);
            split_row(h, DM);
            gemm_row(L[L_WFC], DM, 3072u, (const i32 *)L[L_BFC],
                     (const u32 *)L[L_MFC], (const u32 *)L[L_SHFC],
                     row, 2u);
            if (li == 0u && step == 0u && DW(D_DBG)) {
                volatile i32 *dbg = (volatile i32 *)DW(D_DBG);
                for (u32 i = MUL48(tid); i < MUL48(tid) + 48u; i++)
                    dbg[4608u + i] = row[i];       /* [4608..7680) g16 pre-gelu */
            }
            mk_barrier();
            /* delta-gelu + per-channel requant to int15 (in place) */
            {
                volatile i32 *geld = (volatile i32 *)DW(D_GELUD);
                for (u32 i = MUL48(tid); i < MUL48(tid) + 48u; i++) {
                    i32 g16 = row[i];
                    i32 gi = (i32)mk_satq((u32)(g16 >> 3), 127);
                    i32 gi1 = (i32)mk_satq((u32)(gi + 1), 127);
                    i32 d0 = geld[(u32)(gi + 128)];
                    i32 d1 = geld[(u32)(gi1 + 128)];
                    i32 dd = d1 - d0;
                    u32 fr = (u32)g16 & 7u;
                    u32 md = (u32)(dd >> 31);
                    u32 ad = ((u32)dd ^ md) - md;
                    i32 prod = (i32)((mk_mulu16(ad, fr) ^ md) - md);
                    i32 relu = g16 & ~(g16 >> 31);
                    i32 dq9 = (d0 << 3) + prod;
                    i32 y16 = relu + ((dq9 + 8) >> 4);
                    if (li == 0u && step == 0u && DW(D_DBG) && i < 8u) {
                        volatile i32 *dbx = (volatile i32 *)DW(D_DBG);
                        dbx[7688u + MUL12(i) / 2u] = y16;
                        dbx[7689u + MUL12(i) / 2u] =
                            (i32)((volatile u32 *)L[L_GM])[i];
                        dbx[7690u + MUL12(i) / 2u] =
                            (i32)((volatile u32 *)L[L_GSH])[i];
                        dbx[7691u + MUL12(i) / 2u] = (i32)mk_requant64(
                            mk64_from_i32((u32)y16),
                            ((volatile u32 *)L[L_GM])[i],
                            ((volatile u32 *)L[L_GSH])[i]);
                        dbx[7692u + MUL12(i) / 2u] =
                            (i32)mk_sat15((u32)dbx[7691u + MUL12(i) / 2u]);
                        dbx[7693u + MUL12(i) / 2u] = row[i];  /* readback post-store */
                    }
                    u64p a = mk64_from_i32((u32)y16);
                    u32 gM = ((volatile u32 *)L[L_GM])[i];
                    u32 gsh = ((volatile u32 *)L[L_GSH])[i];
                    row[i] = (i32)mk_sat15(mk_requant64(a, gM, gsh));
                }
            }
            mk_barrier();
            if (tid == 0u && step == 0u && li == 0u) h0_log(0xC0000007u);
            split_row(row, 3072u);
            gemm_row(L[L_WMP], 3072u, DM, (const i32 *)L[L_BMP],
                     (const u32 *)L[L_MMP], (const u32 *)L[L_SHMP],
                     (volatile i32 *)(scr() + S_O), 3u);
            for (u32 i = MUL12(tid); i < MUL12(tid) + 12u; i++)
                x[i] += ((volatile i32 *)(scr() + S_O))[i];
            mk_barrier();
        }

        /* ---- head (only when this step must emit a token) ---- */
        if (step >= n_prompt - 1u) {
            ln_int(x, (const i32 *)DW(D_LNFG), (const i32 *)DW(D_LNFB), h);
            split_row(h, DM);
            volatile i32 *lg = (volatile i32 *)(scr() + S_LOGIT);
            gemm_row(DW(D_HEADW), DM, DW(D_NVPAD), (const i32 *)0,
                     (const u32 *)DW(D_HM), (const u32 *)DW(D_HSH),
                     lg, 3u);
            /* argmax over real vocab (first-max tie-break, split) */
            {
                volatile u32 *red = scr() + S_RED;
                u32 nv = DW(D_NVOCAB), per = DW(D_NVPAD) / 64u;
                i32 bm = (i32)0x80000000; u32 bi = 0u;
                for (u32 i = mk_mulu16(tid, per);
                     i < mk_mulu16(tid + 1u, per); i++) {
                    u32 mv = mk_mask(i < nv);
                    i32 v = (i32)(((u32)lg[i] & mv)
                                  | ((u32)0x80000000u & ~mv));
                    u32 mg = mk_mask(v > bm);
                    bm = (i32)(((u32)bm & ~mg) | ((u32)v & mg));
                    bi = (bi & ~mg) | (i & mg);
                }
                red[X2(tid)] = (u32)bm; red[X2(tid) + 1u] = bi;
                mk_barrier();
                if (tid == 0u)
                    argmax_combine(red, trail, step - (n_prompt - 1u));
                mk_barrier();
            }
        }
    }
}
