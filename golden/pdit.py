#!/usr/bin/env python3
"""M10: p-dit (27-way categorical cell) design study — CDF-scan vs
Gumbel-max in fixed point, settled against golden BEFORE any RTL (the one
open design choice named in the roadmap).

Shared front-end (inherits the frozen D13/D15 p-bit contract): per symbol
k an accumulator in s1.10.3 (units 1/8), z_k = acc*beta with sign-magnitude
half-away rounding to units 2^-8, |z| saturated to 12 - 2^-8 — identical
to golden/pbit.py's z pipeline, so the p-dit reuses the proven MAC/beta
hardware unchanged.

Design A — CDF-scan (division-free):
  z'_k = z_k - max_j z_j  (max tree), saturated to (-12, 0]
  w_k  = EXP_LUT[|z'_k| * 256]  (u0.16, 3072 entries — same geometry as
         the sigmoid LUT), S = sum w_k  (<= 27*65535 < 2^21)
  T    = (u16 * S) >> 16  (one 16x21 multiply), emit first k with
         cumsum(w)_k > T
  Exact quantized distribution in closed form: q_k = (ceil(cum_k*2^16/S)
  - ceil(cum_{k-1}*2^16/S)) / 2^16 — S2-style exactness gate applies
  directly. PRNG cost: 16 bits/sample. ROM: 6 KB. One multiplier.

Design B — Gumbel-max:
  g_k = GUMBEL_LUT[u12_k] (s4.8, 4096 entries — 12-bit uniform per symbol;
  a 16-bit-indexed LUT would be 106KB more ROM than the whole exp table),
  argmax_k(z_k + g_k), ties -> lowest index.
  Exact distribution via discrete order statistics over the LUT PMF.
  PRNG cost: 27*12 = 324 bits/sample. ROM: ~6.5 KB. No multiplier.

Verdict metric: TVD(exact quantized kernel, ideal softmax(z)) across field
regimes, weighed against PRNG bandwidth and area. Run: python pdit.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbit import LUT_N  # noqa: E402  (3072; shared LUT geometry)

K = 27
Z_MAX_RAW = 12 * 256 - 1          # z in units 2^-8, |z| <= 12 - 2^-8

EXP_LUT = np.minimum(np.rint(np.exp(-np.arange(LUT_N) / 256.0) * 65535.0),
                     65535.0).astype(np.int64)   # x in [0,12) step 1/256

_u12 = (np.arange(4096, dtype=np.float64) + 0.5) / 4096.0
GUMBEL_LUT = np.clip(np.rint(-np.log(-np.log(_u12)) * 256.0),
                     -(1 << 12), (1 << 13) - 1).astype(np.int64)  # s4.8


def quantize_z(z_float):
    """Float z vector -> raw ints (units 2^-8), the frozen z pipeline."""
    raw = np.sign(z_float) * np.floor(np.abs(z_float) * 256.0 + 0.5)
    return np.clip(raw, -Z_MAX_RAW, Z_MAX_RAW).astype(np.int64)


# --------------- bit-true cell model (RTL diffs against these) ------------
def z_from_acc(acc_raw: int, beta_raw: int) -> int:
    """(acc s1.10.3 raw, beta u2.6 raw) -> signed z raw (units 2^-8).
    Identical arithmetic to pbit_cell stage 1: sign-magnitude half-away."""
    assert -(1 << 13) <= acc_raw < (1 << 13) and 0 <= beta_raw < 256
    prod = acc_raw * beta_raw            # units 2^-9
    neg = prod < 0
    magr = (abs(prod) + 1) >> 1          # units 2^-8, half-away
    magr = min(magr, Z_MAX_RAW)
    return -magr if neg else magr


def sample_symbol(z_raw, u16: int) -> int:
    """Deterministic (z vector, u16) -> symbol under design A. The RTL is
    bit-true to this function."""
    z_raw = np.asarray(z_raw, dtype=np.int64)
    assert 0 <= u16 < 65536
    idx = np.minimum(z_raw.max() - z_raw, LUT_N - 1)
    w = EXP_LUT[idx]
    S = int(w.sum())
    T = (u16 * S) >> 16
    cum = 0
    for k in range(len(z_raw)):
        cum += int(w[k])
        if cum > T:
            return k
    raise AssertionError("scan fell through (T >= S impossible)")
# ---------------------------------------------------------------------------


def exact_cdf_scan(z_raw):
    """Exact sampled distribution of design A for one z vector."""
    zp = z_raw - z_raw.max()                    # <= 0, units 2^-8
    idx = np.minimum(-zp, LUT_N - 1)            # saturate below exp(-12)
    w = EXP_LUT[idx]
    S = int(w.sum())
    cum = np.cumsum(w)
    bounds = -((-cum * 65536) // S)             # ceil(cum*2^16/S)
    bounds = np.clip(bounds, 0, 65536)
    counts = np.diff(np.concatenate([[0], bounds]))
    q = counts / 65536.0
    assert abs(q.sum() - 1.0) < 1e-12
    return q


def exact_gumbel_max(z_raw):
    """Exact argmax distribution of design B (ties -> lowest index)."""
    # per-symbol discrete score PMF over integer grid (units 2^-8)
    lo = int(z_raw.min() + GUMBEL_LUT.min())
    hi = int(z_raw.max() + GUMBEL_LUT.max())
    n = hi - lo + 1
    pmf = np.zeros((K, n))
    lut_pmf = np.bincount(GUMBEL_LUT - GUMBEL_LUT.min(),
                          minlength=GUMBEL_LUT.max() - GUMBEL_LUT.min() + 1
                          ) / 4096.0
    for k in range(K):
        off = int(z_raw[k]) + GUMBEL_LUT.min() - lo
        pmf[k, off:off + len(lut_pmf)] = lut_pmf
    cdf = np.cumsum(pmf, axis=1)                 # P(s_k <= v)
    cdf_lt = cdf - pmf                           # P(s_k < v)
    q = np.zeros(K)
    for k in range(K):
        others_lt = np.prod([cdf_lt[j] for j in range(k)], axis=0) \
            if k else np.ones(n)
        others_le = np.prod([cdf[j] for j in range(K - 1, k, -1)], axis=0) \
            if k < K - 1 else np.ones(n)
        q[k] = np.sum(pmf[k] * others_lt * others_le)
    assert abs(q.sum() - 1.0) < 1e-9, q.sum()
    return q


def ideal_softmax(z_float):
    e = np.exp(z_float - z_float.max())
    return e / e.sum()


def tvd(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def study(n_per_regime=100, seed=20260707):
    rng = np.random.default_rng(seed)
    regimes = {"weak(s=1)": 1.0, "mid(s=3)": 3.0, "strong(s=8)": 8.0}
    out = {"designs": {}}
    rows = {d: {} for d in ("cdf_scan", "gumbel_max")}
    for name, s in regimes.items():
        ta, tb = [], []
        for _ in range(n_per_regime):
            z = rng.normal(0, s, K)
            zr = quantize_z(z)
            ideal = ideal_softmax(zr / 256.0)   # vs the quantized-z ideal:
            # isolates SAMPLER error; z-quantization is the shared frozen
            # front end already covered by the M4 spec
            ta.append(tvd(exact_cdf_scan(zr), ideal))
            tb.append(tvd(exact_gumbel_max(zr), ideal))
        rows["cdf_scan"][name] = {"mean": float(np.mean(ta)),
                                  "max": float(np.max(ta))}
        rows["gumbel_max"][name] = {"mean": float(np.mean(tb)),
                                    "max": float(np.max(tb))}
    out["designs"]["cdf_scan"] = {
        "tvd": rows["cdf_scan"], "prng_bits_per_sample": 16,
        "rom_bytes": LUT_N * 2, "multipliers": 1}
    out["designs"]["gumbel_max"] = {
        "tvd": rows["gumbel_max"], "prng_bits_per_sample": K * 12,
        "rom_bytes": 4096 * 2, "multipliers": 0}
    worst = {d: max(v["max"] for v in rows[d].values())
             for d in rows}
    out["worst_tvd"] = worst
    out["verdict"] = ("cdf_scan" if worst["cdf_scan"] <= worst["gumbel_max"]
                      else "gumbel_max")
    return out


if __name__ == "__main__":
    r = study()
    mk = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(mk, "ci/logs/t1"), exist_ok=True)
    with open(os.path.join(mk, "ci/logs/t1/pdit_design_study.json"),
              "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))
