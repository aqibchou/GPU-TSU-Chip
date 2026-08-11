#!/usr/bin/env python3
"""M19: the INTEGER GPT-2 chain (gpt2_spec §2/§3) — every operation here
is implementable exactly on the device (RV32I + M18 sidecar), and the
device kernels must match THIS bit-for-bit (G6). numpy is used only as an
integer calculator: all intermediates are int32-representable.

Pipeline (frozen after calibration):
  weights   : per-row symmetric int8 + requant (M,sh) pairs
  activs    : per-tensor static int8 scales (max-abs calibrated)
  residual  : int16 at a per-layer static scale
  layernorm : integer mean (mult-shift /768), integer variance,
              1/sqrt via RSQRT-LUT seed + 2 integer Newton steps
  softmax   : max-subtract -> EXP LUT (exp(x/16)*256) -> u15 normalize
              (integer divide) -> requant to int8 attention weights
  gelu      : sidecar GELU LUT (x/32 domain)
  lm head   : tied wte, per-row requant to a common logit scale, int argmax
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt2_load import Bpe, D_MODEL, N_HEAD, N_LAYER, forward, load_weights
from tensor import exp_table, gelud_table, rsqrt_table

MK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = D_MODEL // N_HEAD
EXP_T = exp_table()
GELUD_T = gelud_table()
RSQ_T = rsqrt_table()


# ---------------- requant helpers (device-exact) ----------------
def mkq(scale_ratio, bits=15):
    """float ratio -> (M, sh) with M < 2^bits (gemmlowp style)."""
    if scale_ratio <= 0:
        return 0, 0
    sh = 0
    while scale_ratio * (1 << sh) < (1 << (bits - 1)) and sh < 30:
        sh += 1
    sh -= 1
    M = int(round(scale_ratio * (1 << sh)))
    return max(M, 1), sh


def mkq_vec(ratios):
    """vector mkq: per-row (M, sh) arrays."""
    r = np.asarray(ratios, dtype=np.float64).ravel()
    Ms = np.zeros(len(r), dtype=np.int64)
    shs = np.zeros(len(r), dtype=np.int64)
    for i, v in enumerate(r):
        Ms[i], shs[i] = mkq(float(v))
        assert 1 <= shs[i] <= 30, (i, v, shs[i])
    return Ms, shs


def requant(acc, M, sh):
    """int32 -> int32 via (acc*M + rnd) >> sh. Device: mk_mul + shifts.
    acc*M can exceed 32 bits — device uses a 64-bit software multiply;
    golden mirrors with int64 then truncates identically."""
    a = acc.astype(np.int64) * M
    return ((a + (1 << (sh - 1))) >> sh).astype(np.int64)


def requant_vec(acc, Ms, shs):
    """per-output-row (M, sh): acc (..., N) int, Ms/shs (N,)."""
    a = acc.astype(np.int64) * Ms
    return (a + (np.int64(1) << (shs - 1))) >> shs


def imm(A, B):
    """exact int8/int16-range matmul via float64 BLAS (|acc| < 2^53)."""
    return np.rint(A.astype(np.float64) @ B.astype(np.float64)
                   ).astype(np.int64)


def sat8(v):
    return np.clip(v, -128, 127).astype(np.int64)


def sat16(v):
    return np.clip(v, -32768, 32767).astype(np.int64)


def sat15(v):
    """int15 activations (D-018): weights stay int8 (the Q8_0 claim);
    a 15-bit activation feeds the sidecar as TWO int8 GEMMs
    (hi7 = x>>7, lo7 = x&127; acc = acc_hi*128 + acc_lo, combined in
    the kernel's 64-bit software path before requant). Partial accs
    stay int32-safe for K up to 4096."""
    return np.clip(v, -16384, 16383).astype(np.int64)


def quant_rows(W):
    """per-output-row symmetric int8. W (in, out) -> Wq, row scales."""
    s = np.abs(W).max(axis=0) / 127.0
    s[s == 0] = 1.0
    Wq = np.clip(np.round(W / s), -128, 127).astype(np.int64)
    return Wq, s


# ---------------- integer primitives ----------------
def int_rsqrt(v):
    """1/sqrt(v) for integer v >= 1, device-exact.
    Normalize v ~= b * 2^t with b in [64, 255] and t EVEN; the frozen
    table is RSQ_T[i] = 2^16 / sqrt(i+1) exactly, so
       1/sqrt(v) = RSQ_T[b-1] * 2^-(16 + t/2).
    Returns (r, sh) with 1/sqrt(v) ~= r * 2^-sh. Table gives ~2^-6 rel
    accuracy — measured against the G7 ppl budget before freezing."""
    v = int(v)
    assert v >= 1
    # range-reduce to < 2^31 by an EVEN shift so the Newton product
    # v*r^2 stays within int64 on device (2^31 * 2^26 = 2^57)
    pre = 0
    if v.bit_length() > 31:
        pre = v.bit_length() - 31
        if pre % 2:
            pre += 1
        v >>= pre
    msb = v.bit_length() - 1
    t = msb - 7
    if t % 2 != 0:
        t += 1                        # keep t even, b in [64, 255]
    b = (v >> t) if t >= 0 else (v << (-t))
    b = min(max(b, 1), 255)
    r = int(RSQ_T[b - 1])             # seed: 1/sqrt(v) ~ r * 2^-k
    k = 16 + t // 2
    # one Newton step against the FULL v (gpt2_spec §3) — the seed's
    # dominant error is the mantissa truncation b = v>>t, so refining
    # against b would converge to the same wrong point. The correction
    # term is pre-shifted so every intermediate fits the device's
    # 64-bit software arithmetic (r*diff would need ~70 bits at k=28;
    # the truncation costs ~2^-40 relative — far below the target):
    #   diff = 3*2^2k - v*r^2            (<= ~2^(2k+2))
    #   r'   = (r * (diff >> se)) >> (2k + 1 - se),  se = max(0, 2k-47)
    se = max(0, 2 * k - 47)
    diff = (3 << (2 * k)) - v * r * r
    r = (r * (diff >> se)) >> (2 * k + 1 - se)
    return r, k + pre // 2


LNC = 65536.0 / np.sqrt(768.0)   # n_raw = (d/sigma) * LNC
# derivation: vs = sum((d>>s0)^2), r*2^-k ~ 1/sqrt(vs); with the s0 fold
#   n = d*r >> (k + s0 - 16) = d*2^16/sqrt(sum d^2) = (d/sigma)*2^16/sqrt(768)


def int_layernorm(x32, gq, bq):
    """x32 int (T, 768) residual at ANY scale (LN is scale-invariant).
    Device-exact per token, all deltas kept EXACT in int32:
      mean = (sum(x) * 21845) >> 24                  (1/768 frozen)
      d    = x - mean                                (exact int32)
      s0   = max(0, msb(max|d|) - 26)                (int64 Sum-sq guard)
      vs   = sum((d>>s0)^2)                          (int64 accumulate)
      r,k  = int_rsqrt(vs)                           (LUT + Newton)
      n    = (d * r) >> (k + s0 - 16)   -> n = (d/sigma) * LNC
      y    = sat15( (n * gq + 1<<19) >> 20 + bq )
    gq folds gamma*16256/(s_ln*LNC)*2^20; bq folds beta*16256/s_ln.
    Wide products (sum*21845, d*r, Newton) use the device's 64-bit
    software multiply; golden mirrors in int64."""
    x = x32.astype(np.int64)
    T = x.shape[0]
    out = np.zeros_like(x)
    for t in range(T):
        v = x[t]
        mean = (int(v.sum()) * 21845) >> 24        # exact-ish /768
        d = v - mean                               # EXACT int32 deltas
        # (the old x>>sh0 pre-shift starved ordinary channels when a
        # 3000x outlier channel set the shift — D-018)
        mx = int(np.abs(d).max())
        s0 = max(0, mx.bit_length() - 26)          # sum fits int64
        ds = d >> s0
        vs = int((ds * ds).sum())
        r, k = int_rsqrt(max(vs, 1))
        shn = k + s0 - 16
        out[t] = (d * r) >> shn if shn >= 0 else (d * r) << (-shn)
    y = (out * gq + (1 << 19)) >> 20
    return sat15(y + bq)


EXP_HALF = 33406   # round(exp(-1/32) * 2^15): the odd x/32 half-step


def int_softmax_row(scores16):
    """int16 scores in the x/32 domain -> u14 attention weights.
    u14 (D-018): u7 weights (prob step 1/128) zeroed most of a diffuse
    1024-position attention row — ppl blew up 8x at full context. u14
    keeps 1/16384 steps; the device realizes the PV product as TWO int8
    sidecar GEMMs (hi7 = p>>7, lo7 = p&127, acc = (hi<<7)+lo — exactly
    the single u14 GEMM golden computes here).
    Scores stay int16 until AFTER max-subtraction (D-018): int8 scores
    clipped at +-8 score units flattened sharp attention (true scores
    reach 14+); post-subtract z is safely clipped to the LUT domain
    because exp(z/16)*256 rounds to 0 below z = -128 anyway."""
    m = scores16.max()
    z = np.clip(scores16 - m, -255, 0)
    e = EXP_T[((z >> 1) + 128).astype(np.int64)].astype(np.int64)
    odd = (z & 1).astype(bool)
    e[odd] = (e[odd] * EXP_HALF) >> 15              # exp(-1/32) half-step
    ssum = int(e.sum())
    p14 = (e << 14) // max(ssum, 1)
    return np.minimum(p14, 16383)   # hi7 must fit int8 (p=1.0 rows)


class IntGpt2:
    def __init__(self, W, calib_windows):
        """calib_windows: list of token-id windows (or one flat list)."""
        self.bpe = None
        self.Wf = W
        self.q = {}
        self.scale = {}
        if calib_windows and isinstance(calib_windows[0], (int, np.integer)):
            calib_windows = [calib_windows]
        agg = {}
        for w in calib_windows:
            self.scale = {}
            self._calibrate(list(w))
            for k, v in self.scale.items():
                if k == "resid":
                    cur = agg.get(k, [0.0] * len(v))
                    agg[k] = [max(a, b) for a, b in zip(cur, v)]
                elif k in agg:
                    agg[k] = np.maximum(agg[k], v)
                else:
                    agg[k] = v
        agg["resid_max"] = max(agg["resid"])
        self.scale = agg
        self._quantize_weights()
        self._freeze()

    def _quantize_weights(self):
        """Per-output-row int8 with the CONSUMER-SIDE equalization fold
        (D-018): every int8 activation x is stored per-channel scaled,
        x8_c = x_c / u_c, and the consuming GEMM absorbs u_c into its
        weights BEFORE quantization (W~_cr = W_cr * u_c) — so the acc has
        a single per-row scale and the device kernel is unchanged. This
        is what rescues GPT-2's outlier channels (the LN-exact ablation:
        ppl 171 -> 49 with everything else still int8)."""
        q, sc = self.q, self.scale

        def u(name):
            return sc[name] / 127.0

        def fold_quant(key, u_in):
            Wt = self.Wf[key + ".weight"].astype(np.float64) * u_in[:, None]
            Wq, srow = quant_rows(Wt)
            q[key + ".w"] = Wq
            q[key + ".s"] = srow
            q[key + ".b"] = self.Wf[key + ".bias"].astype(np.float64)

        for li in range(N_LAYER):
            p = f"h.{li}."
            fold_quant(p + "attn.c_attn", u(p + "ln1_c"))
            fold_quant(p + "attn.c_proj", u(p + "attnout_c"))
            fold_quant(p + "mlp.c_fc", u(p + "ln2_c"))
            fold_quant(p + "mlp.c_proj", u(p + "gelu_c"))
        # tied wte: RAW rows for the embedding lookup, lnf-folded rows
        # for the head (the fold breaks the tie in int8 space only)
        Wq, srw = quant_rows(self.Wf["wte.weight"].T.astype(np.float64))
        q["emb.w"], q["emb.s"] = Wq, srw
        Wq, srw = quant_rows(self.Wf["wte.weight"].T.astype(np.float64)
                             * u("lnf_c")[:, None])
        q["head.w"], q["head.s"] = Wq, srw

    def _calibrate(self, ids):
        """Static per-tensor activation scales from the FP32 forward's
        intermediate ranges on the calibration slice. 99.9th-percentile
        of |x| (D-018): max-abs let single outliers starve the typical
        range of int8 levels (attnout at L11: typical 0.3 vs outlier
        13.5 -> 3 levels); percentile scales saturate the outliers
        instead. Residual keeps true max (int32 headroom check only)."""
        # run fp32 forward capturing ranges
        W = self.Wf
        from gpt2_load import _gelu, _ln
        T = len(ids)
        x = W["wte.weight"][ids] + W["wpe.weight"][:T]
        sc = self.scale

        def pct(h):
            return float(np.percentile(np.abs(h), 99.9))

        def pcv(h, margin=1.0):
            """per-channel equalization scale (D-018): s_c^alpha *
            S^(1-alpha), s_c = per-channel 99.9 pct (floored at max/64),
            S = per-tensor 99.9 pct; alpha splits the outlier burden
            between activation saturation and weight resolution.
            `margin` widens the range: int15 interfaces take margin 4
            (clipping was their ENTIRE quantization cost — the ln
            noclip/arith split isolated it — while 4x margin costs them
            only 2 of 7 spare resolution bits); int8 interfaces stay at
            margin 1 (they cannot afford the resolution)."""
            # int15 interfaces (margin > 1): the margin covers the
            # activation tails, so alpha here trades ONLY weight-row
            # spread (the fold pushes u_c into consumer weight rows;
            # alpha<1 compresses that spread with no clipping cost)
            a = (float(os.environ.get("MK_A15", "0.5")) if margin > 1.0
                 else float(os.environ.get("MK_ALPHA", "0.75")))
            v = np.percentile(np.abs(h), float(
                os.environ.get("MK_PCV", "99.9")), axis=0)
            v = np.maximum(v, v.max() / 64.0)
            S = float(np.percentile(np.abs(h), 99.9))
            return v ** a * S ** (1.0 - a) * margin

        sc["resid"] = [float(np.abs(x).max())]
        mask = np.triu(np.full((T, T), -1e10, dtype=np.float32), 1)
        for li in range(N_LAYER):
            p = f"h.{li}."
            h = _ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
            sc[p + "ln1_c"] = pcv(h, float(os.environ.get("MK_B15", "16.0")))
            qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
            qh, kh, vh = np.split(qkv, 3, axis=-1)
            sc[p + "q"] = np.array([pct(qh[:, h_ * HD:(h_ + 1) * HD])
                                    for h_ in range(N_HEAD)])
            sc[p + "k"] = np.array([pct(kh[:, h_ * HD:(h_ + 1) * HD])
                                    for h_ in range(N_HEAD)])
            sc[p + "v_c"] = pcv(vh)
            qh = qh.reshape(T, N_HEAD, HD).transpose(1, 0, 2)
            kh = kh.reshape(T, N_HEAD, HD).transpose(1, 0, 2)
            vh = vh.reshape(T, N_HEAD, HD).transpose(1, 0, 2)
            att = qh @ kh.transpose(0, 2, 1) / np.sqrt(HD) + mask
            att = att - att.max(-1, keepdims=True)
            e = np.exp(att)
            att = e / e.sum(-1, keepdims=True)
            o = (att @ vh).transpose(1, 0, 2).reshape(T, D_MODEL)
            sc[p + "attnout_c"] = pcv(o, float(os.environ.get("MK_B15", "16.0")))
            x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
            sc["resid"].append(float(np.abs(x).max()))
            h = _ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
            sc[p + "ln2_c"] = pcv(h, float(os.environ.get("MK_B15", "16.0")))
            g = h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]
            sc[p + "fc"] = pct(g)
            h = _gelu(g)
            # gelu activations are sparse spikes: a percentile scale
            # saturates the rare large values that carry the signal ->
            # per-channel MAX here (g8 relrms 0.25 -> the L0 resid break)
            gc = np.abs(h).max(axis=0)
            sc[p + "gelu_c"] = np.maximum(gc, gc.max() / 64.0) * float(os.environ.get("MK_B15", "16.0"))
            x = x + h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
            sc["resid"].append(float(np.abs(x).max()))
        h = _ln(x, W["ln_f.weight"], W["ln_f.bias"])
        sc["lnf_c"] = pcv(h, float(os.environ.get("MK_B15", "16.0")))
        sc["resid_max"] = max(sc["resid"])

    # ---------------- freeze: every integer constant ----------------
    def _freeze(self):
        """Fold calibrated scales into the frozen integer constants the
        device kernels carry verbatim. Residual: int32 at a GLOBAL 2^-16
        scale (exact adds; LN is scale-invariant so the 650x per-layer
        outlier growth costs nothing). All int8 activation interfaces
        are per-channel equalized; the folds live in the already-
        quantized weights, so every requant below is per-row/{M,sh}."""
        RS = 65536.0
        f, q, sc = self.Wf, self.q, self.scale
        z = self.iz = {}

        def u(name):
            return sc[name] / 127.0

        def ln_const(pfx, sname):
            g = f[pfx + ".weight"].astype(np.float64)
            b = f[pfx + ".bias"].astype(np.float64)
            gq = np.round(g * 16256.0 * (1 << 20) / (sc[sname] * LNC)
                          ).astype(np.int64)
            bq = np.round(b * 16256.0 / sc[sname]).astype(np.int64)
            return gq, bq

        def lin_const(key, u_out):
            """input unit folded into the weights at u_c; the int15
            activation is stored at u_c/128, so acc unit = srow/128.
            Per-row bias (acc units) + requant into u_out units."""
            srow = q[key + ".s"].astype(np.float64) / 128.0
            bacc = np.round(q[key + ".b"] / srow).astype(np.int64)
            Ms, shs = mkq_vec(srow / u_out)
            return bacc, Ms, shs

        z["wpe"] = np.round(f["wpe.weight"].astype(np.float64) * RS
                            ).astype(np.int64)
        z["emb_M"], z["emb_sh"] = mkq_vec(
            q["emb.s"].astype(np.float64) * RS)
        for li in range(N_LAYER):
            pp = f"h.{li}."
            z[pp + "ln1"] = ln_const(pp + "ln_1", pp + "ln1_c")
            uo = np.concatenate([np.repeat(u(pp + "q"), HD),
                                 np.repeat(u(pp + "k"), HD),
                                 u(pp + "v_c")])
            z[pp + "qkv"] = lin_const(pp + "attn.c_attn", uo)
            # scores -> x/32 domain, PER-HEAD (M, sh); softmax uses
            # EXP_T on z>>1 plus one multiply for the odd half-step
            sMs, sshs = mkq_vec(u(pp + "q") * u(pp + "k") / 8.0 * 32.0)
            z[pp + "sM"], z[pp + "ssh"] = sMs, sshs
            # PV: p is u14; o -> int8 per-channel (o inherits v's channel
            # structure only via attention mixing over TIME, so o gets
            # its own calibrated per-channel scale)
            z[pp + "oM"], z[pp + "osh"] = mkq_vec(
                (u(pp + "v_c") / 16384.0) / (u(pp + "attnout_c") / 128.0))
            z[pp + "cproj"] = lin_const(pp + "attn.c_proj",
                                        np.full(D_MODEL, 1.0 / RS))
            z[pp + "ln2"] = ln_const(pp + "ln_2", pp + "ln2_c")
            # c_fc lands at q8 (x*256): sat16 clips at |x|=128 (the q9
            # domain clipped at 64 -- calibrated fc max 62.6, eval
            # exceeds it); the interpolated LUT absorbs the coarser step
            z[pp + "cfc"] = lin_const(pp + "mlp.c_fc",
                                      np.full(4 * D_MODEL, 1.0 / 256.0))
            z[pp + "gM"], z[pp + "gsh"] = mkq_vec(
                (1.0 / 256.0) / (u(pp + "gelu_c") / 128.0))
            z[pp + "mproj"] = lin_const(pp + "mlp.c_proj",
                                        np.full(D_MODEL, 1.0 / RS))
        z["lnf"] = ln_const("ln_f", "lnf_c")
        z["hM"], z["hsh"] = mkq_vec(
            q["head.s"].astype(np.float64) / 128.0 * 256.0)

    # ---------------- the integer forward (G6's bit contract) -------
    def forward_int(self, ids, trace=None):
        """ids -> integer logits (T, 50257) in q8 (logit * 256). Every op
        below has a direct device realization (gpt2_spec §3); full-context
        vectorization == the device's incremental loop bit-for-bit because
        every op is causal and per-token."""
        q, z = self.q, self.iz
        T = len(ids)
        x = requant_vec(q["emb.w"][:, ids].T, z["emb_M"][ids][:, None],
                        z["emb_sh"][ids][:, None]) + z["wpe"][:T]
        for li in range(N_LAYER):
            pp = f"h.{li}."
            gq, bq = z[pp + "ln1"]
            h8 = int_layernorm(x, gq, bq)
            bacc, Ms, shs = z[pp + "qkv"]
            qkv = sat8(requant_vec(imm(h8, q[pp + "attn.c_attn.w"]) + bacc,
                                   Ms, shs))
            if trace is not None:
                trace.setdefault("k8", []).append(
                    qkv[:, D_MODEL:2 * D_MODEL].copy())
            o8 = np.zeros((T, D_MODEL), dtype=np.int64)
            for hd in range(N_HEAD):
                sl = slice(hd * HD, (hd + 1) * HD)
                qh = qkv[:, sl]
                kh = qkv[:, D_MODEL + hd * HD:D_MODEL + (hd + 1) * HD]
                vh = qkv[:, 2 * D_MODEL + hd * HD:2 * D_MODEL + (hd + 1) * HD]
                s16 = sat16(requant(imm(qh, kh.T),
                                    int(z[pp + "sM"][hd]),
                                    int(z[pp + "ssh"][hd])))
                s16[np.triu_indices(T, 1)] = -(1 << 15)  # causal mask
                p14 = np.zeros((T, T), dtype=np.int64)
                for t in range(T):
                    p14[t] = int_softmax_row(s16[t])
                o8[:, sl] = sat15(requant_vec(imm(p14, vh),
                                              z[pp + "oM"][sl],
                                              z[pp + "osh"][sl]))
            if trace is not None and li == 0:
                trace["o0"] = o8.copy()
            bacc, Ms, shs = z[pp + "cproj"]
            x = x + requant_vec(imm(o8, q[pp + "attn.c_proj.w"]) + bacc,
                                Ms, shs)
            if trace is not None and li == 0:
                trace["x_attn0"] = x.copy()
            gq, bq = z[pp + "ln2"]
            h8 = int_layernorm(x, gq, bq)
            bacc, Ms, shs = z[pp + "cfc"]
            g16 = sat16(requant_vec(imm(h8, q[pp + "mlp.c_fc.w"]) + bacc,
                                    Ms, shs))
            if trace is not None and li == 0:
                trace["g16_0"] = g16.copy()
            # gelu at q8: exact relu + bounded q9 delta, LUT linearly
            # interpolated on the 3 fractional q8 bits (kernel-side
            # arithmetic only: idx, idx+1, one mul, one shift); the
            # delta is halved into q8 to add to the q8 relu part
            gi = np.clip(g16 >> 3, -128, 127)
            d0 = GELUD_T[gi + 128]
            d1 = GELUD_T[np.clip(gi + 1, -128, 127) + 128]
            dq9 = (d0 << 3) + ((d1 - d0) * (g16 & 7))
            y16 = np.maximum(g16, 0) + ((dq9 + 8) >> 4)
            g8 = sat15(requant_vec(y16, z[pp + "gM"],
                                   z[pp + "gsh"]))
            if trace is not None and li == 0:
                trace["g0"] = g8.copy()
            bacc, Ms, shs = z[pp + "mproj"]
            x = x + requant_vec(imm(g8, q[pp + "mlp.c_proj.w"]) + bacc,
                                Ms, shs)
        gq, bq = z["lnf"]
        h8 = int_layernorm(x, gq, bq)
        if trace is not None:
            trace["x"] = x.copy()
            trace["h_lnf"] = h8.copy()
        return requant_vec(imm(h8, q["head.w"]), z["hM"], z["hsh"])

    def greedy(self, ids, n):
        ids = list(ids)
        for _ in range(n):
            ids.append(int(np.argmax(self.forward_int(ids)[-1])))
        return ids

    def ppl_int(self, ids, window=1024):
        """windowed ppl from the q8 integer logits (metric is host-side
        float over device-representable logits — the chain stays integer)."""
        nlls = []
        for s0 in range(0, len(ids) - 1, window):
            chunk = ids[s0:s0 + window]
            if len(chunk) < 2:
                break
            lg = self.forward_int(chunk).astype(np.float64) / 256.0
            lp = lg[:-1] - lg[:-1].max(-1, keepdims=True)
            lse = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
            nlls.extend(-lse[np.arange(len(chunk) - 1),
                             np.asarray(chunk[1:])])
        return float(np.exp(np.mean(nlls)))


def calib_windows(bpe):
    """FROZEN calibration protocol (G7): 4 windows x 1024 tokens from
    the text8 TRAIN region (disjoint from the frozen eval slice)."""
    from text8 import splits
    tr, _, _ = splits()
    raw = "".join(" " if c == 0 else chr(96 + c) for c in tr[:60000])
    ids = bpe.encode(raw)
    return [ids[i * 1024:(i + 1) * 1024] for i in range(4)]


def eval_slice(bpe):
    """The sha-pinned 2048-token eval slice (G7)."""
    from text8 import splits
    _, _, test = splits()
    raw = "".join(" " if c == 0 else chr(96 + c) for c in test[:12000])
    return bpe.encode(raw)[:2048]


if __name__ == "__main__":
    W = load_weights()
    bpe = Bpe()
    m = IntGpt2(W, calib_windows(bpe))
    errs = []
    for v in [1, 100, 123456, 10**7, 2**31 - 1, 10**14, (1 << 62) - 1]:
        r, sh = int_rsqrt(v)
        errs.append(abs(r * 2.0 ** (-sh) * v ** 0.5 - 1))
    print("rsqrt rel errs:", [f"{e:.6f}" for e in errs])
    assert max(errs) < 3e-4, max(errs)
    prompt = bpe.encode("The capital of France is")
    gi = m.greedy(prompt, 6)
    print("int greedy :", repr(bpe.decode(gi)))
    gf = list(prompt)
    for _ in range(6):
        gf.append(int(np.argmax(forward(W, gf)[-1])))
    print("fp32 greedy:", repr(bpe.decode(gf)))
    p_int = m.ppl_int(eval_slice(bpe))
    print(f"Q8 golden-chain ppl on the frozen slice: {p_int:.2f}")
