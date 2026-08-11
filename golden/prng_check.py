"""S1 cross-stream independence battery (Σ.4c, thresholds frozen per R3).

Statistics computed:
- pairwise Pearson correlation at lag 0 and lags {1,2,4,8} for every ordered
  pair (stream i vs stream j shifted) — detects inter-stream structure
- within-stream: mean z-test and serial autocorrelation at lags {1,2,4,8}
- per-stream bit balance (popcount binomial z)

Null model: for N word pairs, r*sqrt(N) ~ N(0,1). FROZEN decision rule:
two-sided p-values, Bonferroni-corrected family-wise alpha = 0.01 across ALL
statistics computed in one battery invocation. (The unified doc's "within 3σ"
per-stat rule would false-alarm ~50 times across the ~18k statistics of the
full battery; risk #4 explicitly mandates pre-registered corrections — this
is the retained D-011 rationale.)
"""
from __future__ import annotations

import math

import numpy as np

LAGS = (1, 2, 4, 8)


def _z_to_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def battery(words: np.ndarray) -> dict:
    """words: (nstreams, nwords) uint32. Returns dict with stats and verdict."""
    ns, nw = words.shape
    # float32 keeps the full-size battery inside 8 GB RAM; dot-product error
    # (~1e-4 relative) is negligible against the |z| ~ 4.8 decision threshold
    x = words.astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    x /= sd

    stats = []  # (name, z)

    # within-stream mean (of raw uniform words)
    u = words.astype(np.float64) / 2**32 - 0.5
    for i in range(ns):
        z = u[i].mean() / (math.sqrt(1.0 / 12.0) / math.sqrt(nw))
        stats.append((f"mean[{i}]", z))

    # within-stream serial autocorrelation
    for i in range(ns):
        for lag in LAGS:
            r = float(np.dot(x[i, :-lag], x[i, lag:])) / (nw - lag)
            stats.append((f"acf[{i},{lag}]", r * math.sqrt(nw - lag)))

    # bit balance: popcount per word ~ Binomial(32, 1/2) -> mean 16, var 8
    pop = np.bitwise_count(words)
    for i in range(ns):
        z = (float(pop[i].mean()) - 16.0) / (math.sqrt(8.0) / math.sqrt(nw))
        stats.append((f"bits[{i}]", z))

    # pairwise cross-correlation, lag 0 and ±LAGS (i leads / j leads)
    for i in range(ns):
        for j in range(i + 1, ns):
            r0 = float(np.dot(x[i], x[j])) / nw
            stats.append((f"xc[{i},{j},0]", r0 * math.sqrt(nw)))
            for lag in LAGS:
                r = float(np.dot(x[i, :-lag], x[j, lag:])) / (nw - lag)
                stats.append((f"xc[{i},{j},+{lag}]", r * math.sqrt(nw - lag)))
                r = float(np.dot(x[j, :-lag], x[i, lag:])) / (nw - lag)
                stats.append((f"xc[{i},{j},-{lag}]", r * math.sqrt(nw - lag)))

    m = len(stats)
    alpha = 0.01
    thr_p = alpha / m
    worst = max(stats, key=lambda s: abs(s[1]))
    fails = [(n, z, _z_to_p(z)) for n, z in stats if _z_to_p(z) < thr_p]
    return {
        "n_stats": m,
        "alpha_family": alpha,
        "p_per_stat": thr_p,
        "worst": {"name": worst[0], "z": worst[1], "p": _z_to_p(worst[1])},
        "failures": fails,
        "ok": not fails,
    }


if __name__ == "__main__":
    # sanity: golden streams should pass; a deliberately correlated pair must fail
    from golden.xoshiro import FarmVec, stream_states

    fv = FarmVec(stream_states(0xC0FFEE, 8))
    w = fv.next_block(1 << 16)
    r = battery(w)
    assert r["ok"], r["failures"][:5]
    # negative control: stream 7 := stream 0 XOR small noise -> must fail
    w2 = w.copy()
    w2[7] = w2[0] ^ (w2[1] & np.uint32(0xF))
    r2 = battery(w2)
    assert not r2["ok"], "battery failed to catch a planted correlation"
    print(f"prng_check self-check OK — {r['n_stats']} stats clean; "
          f"planted correlation caught ({len(r2['failures'])} stats fired)")
