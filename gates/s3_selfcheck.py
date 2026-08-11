#!/usr/bin/env python3
"""S3 gate: at-scale self-consistency of the scaled engine (fabric_grid).

════════ FROZEN DECISION RULES (R3, set 2026-07-04 before data collection) ══
A. Small-N exactness cross-engine: the S2 instance family (king-4x4 idx 0,
   beta 32) run on fabric_grid (lane-chunked schedule) must match pi_hw —
   the same reference as fabric16, valid because within-color per-site
   kernels commute. n = 1e6 thinned x8, chi2 tail-pooled >= 5, p >= 1e-3.
B. ESS at scale: 32x32 torus, J ~ +/-1, beta raw 32, 100k sweeps, energy
   recorded every sweep. tau = emcee integrated autocorrelation time of the
   energy trace; PASS: estimator converges and ESS = n/tau >= 500.
C. Within-color order invariance: instance A rerun with each color segment
   REVERSED (kernel-identical by commutation): two-sample chi2 on pooled
   (>=5) buckets between the two 1e6-sample histograms, p >= 1e-3.
Also reports flips/s sigma projections at 100/200 MHz from run B's counters.
Evidence: ci/logs/s3/. All numbers sigma-labeled (simulated).
════════════════════════════════════════════════════════════════════════════
"""
import json
import pathlib
import struct
import subprocess
import sys
import time

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))

import numpy as np                                    # noqa: E402
from emcee.autocorr import integrated_time            # noqa: E402
from scipy.stats import chi2 as chi2_dist             # noqa: E402

from golden.gibbs import make_instance, pi_hw          # noqa: E402
from golden.gibbs_grid import Graph, build_schedule, torus_king  # noqa: E402
from golden.xoshiro import stream_states               # noqa: E402

RES = MK / "ci/logs/s3"
MC = MK / "tb/fabric_grid/mc/grid_mc"
P = 8
SEED = 0x53C0FFEE


def graph_from_J(J, h):
    g = Graph(J.shape[0])
    for i in range(J.shape[0]):
        for j in range(i + 1, J.shape[1]):
            if J[i, j]:
                g.add_edge(i, j, int(J[i, j]))
    return g, h


def write_cfg(g, bias, order, bounds, sched, seed, d, tag, reverse=False):
    if reverse:
        order = list(order)
        for lo, hi in bounds:
            order[lo:hi] = order[lo:hi][::-1]
    rows = np.zeros((g.n, 9), dtype=np.uint32)
    for i in range(g.n):
        for k, (j, jr) in enumerate(g.adj[i]):
            rows[i, k] = (1 << 23) | (j << 10) | (jr & 0x3FF)
        rows[i, 8] = int(bias[i]) & 0x3FF
    (d / f"{tag}_rows.bin").write_bytes(rows.tobytes())
    (d / f"{tag}_ord.bin").write_bytes(np.array(order, dtype=np.uint16).tobytes())
    cb = np.zeros((16, 2), dtype=np.uint16)
    for c, (lo, hi) in enumerate(bounds):
        cb[c] = (lo, hi)
    (d / f"{tag}_cb.bin").write_bytes(cb.tobytes())
    (d / f"{tag}_sched.bin").write_bytes(
        np.array([(b << 24) | s for b, s in sched], dtype=np.uint32).tobytes())
    (d / f"{tag}_seeds.bin").write_bytes(
        b"".join(struct.pack("<4I", *s) for s in stream_states(seed, P)))
    return len(order), len(bounds)


def run_mc(d, tag, n, ncol, nord, nsched, bipolar, dump, every):
    cmd = [MC, f"+rows={d}/{tag}_rows.bin", f"+ord={d}/{tag}_ord.bin",
           f"+cb={d}/{tag}_cb.bin", f"+sched={d}/{tag}_sched.bin",
           f"+seeds={d}/{tag}_seeds.bin", f"+n={n}", f"+nord={nord}",
           f"+ncol={ncol}", f"+nsched={nsched}", f"+bipolar={int(bipolar)}",
           f"+dump={dump}", f"+every={every}"]
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       timeout=7200)
    assert r.returncode == 0 and "%" not in r.stdout, \
        (r.stdout + r.stderr)[:400]
    kv = dict(p.split("=") for p in r.stdout.split())
    return {k: int(v) for k, v in kv.items()}


def hist_from_dump(path, n_sites, drop=0):
    raw = np.frombuffer(pathlib.Path(path).read_bytes(), dtype=np.uint8)
    nb = (n_sites + 7) // 8
    arr = raw.reshape(-1, nb)[drop:]
    states = arr @ (1 << (8 * np.arange(nb, dtype=np.uint64)))
    return np.bincount(states.astype(np.int64), minlength=1 << n_sites)


def pooled(exp, obs):
    order = np.argsort(-exp)
    e, o = exp[order], obs[order]
    keep = e >= 5.0
    return (np.append(e[keep], e[~keep].sum()),
            np.append(o[keep], o[~keep].sum()))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    d = RES / "cfg"
    d.mkdir(exist_ok=True)
    ok = True
    res = {}

    # --- A: cross-engine exactness at N=16 ---
    J, h = make_instance(0x2E5EED)                    # S2 instance 0
    g, bias = graph_from_J(J, h)
    _, order, bounds = build_schedule(g)
    n_samp, thin, burn = 1_000_000, 8, 2000
    nord, ncol = write_cfg(g, bias, order, bounds,
                           [(32, burn + n_samp * thin)], SEED, d, "a")
    run_mc(d, "a", g.n, ncol, nord, 1, False, d / "a_dump.bin", thin)
    hist = hist_from_dump(d / "a_dump.bin", g.n, drop=burn // thin)
    n_eff = int(hist.sum())
    pi = pi_hw(J, h, 32)[0]
    e, o = pooled(pi * n_eff, hist.astype(np.float64))
    stat = float(((o - e) ** 2 / e).sum())
    p_a = float(chi2_dist.sf(stat, len(e) - 1))
    res["A"] = {"p": p_a, "n": n_eff}
    ok &= p_a >= 1e-3
    print(f"[A] cross-engine chi2 p={p_a:.4f} ({'PASS' if p_a >= 1e-3 else 'FAIL'})")

    # --- B: ESS at 1024 sites ---
    rng = np.random.default_rng(7)
    gt = torus_king(32, 32, lambda r, c, dr, dc: int(rng.choice([-8, 8])), deg4=True)
    bt = np.zeros(gt.n, dtype=np.int32)
    _, order, bounds = build_schedule(gt)
    nord, ncol = write_cfg(gt, bt, order, bounds, [(32, 100_000)], SEED + 1, d, "b")
    t0 = time.time()
    cnt = run_mc(d, "b", gt.n, ncol, nord, 1, True, d / "b_dump.bin", 1)
    wall = time.time() - t0
    raw = np.frombuffer((d / "b_dump.bin").read_bytes(), dtype=np.uint8)
    arr = np.unpackbits(raw.reshape(-1, gt.n // 8), axis=1, bitorder="little")
    m = 2.0 * arr.astype(np.float64) - 1.0
    deg = max(len(a) for a in gt.adj)
    nbr = np.zeros((gt.n, deg), dtype=np.int32)
    jw = np.zeros((gt.n, deg))
    for i, a in enumerate(gt.adj):
        for k, (j, jr) in enumerate(a):
            nbr[i, k], jw[i, k] = j, jr / 8.0
    energy = -0.5 * np.einsum("ti,ik,tik->t", m, jw, m[:, nbr])
    tau = float(integrated_time(energy[2000:], tol=50)[0])
    ess = (len(energy) - 2000) / tau
    res["B"] = {"tau": tau, "ess": ess, "upd": cnt["upd"], "cycles": cnt["cycles"],
                "flips": cnt["flip"], "wall_s": wall}
    ok &= ess >= 500
    ups_100 = cnt["upd"] / cnt["cycles"] * 100e6
    print(f"[B] tau={tau:.1f} ESS={ess:.0f} ({'PASS' if ess >= 500 else 'FAIL'}); "
          f"sigma flips/s @100MHz={ups_100:.3g} @200MHz={ups_100 * 2:.3g}")

    # --- C: within-color order invariance ---
    nord, ncol = write_cfg(g, bias, order if False else build_schedule(g)[1],
                           build_schedule(g)[2],
                           [(32, burn + n_samp * thin)], SEED + 2, d, "c1")
    run_mc(d, "c1", g.n, ncol, nord, 1, False, d / "c1_dump.bin", thin)
    nord, ncol = write_cfg(g, bias, build_schedule(g)[1], build_schedule(g)[2],
                           [(32, burn + n_samp * thin)], SEED + 3, d, "c2",
                           reverse=True)
    run_mc(d, "c2", g.n, ncol, nord, 1, False, d / "c2_dump.bin", thin)
    h1 = hist_from_dump(d / "c1_dump.bin", g.n, drop=burn // thin).astype(np.float64)
    h2 = hist_from_dump(d / "c2_dump.bin", g.n, drop=burn // thin).astype(np.float64)
    tot = h1 + h2
    keep = tot >= 10
    a1, a2 = np.append(h1[keep], h1[~keep].sum()), np.append(h2[keep], h2[~keep].sum())
    n1, n2 = a1.sum(), a2.sum()
    exp1 = (a1 + a2) * n1 / (n1 + n2)
    exp2 = (a1 + a2) * n2 / (n1 + n2)
    stat = float((((a1 - exp1) ** 2) / exp1).sum() + (((a2 - exp2) ** 2) / exp2).sum())
    p_c = float(chi2_dist.sf(stat, len(a1) - 1))
    res["C"] = {"p": p_c}
    ok &= p_c >= 1e-3
    print(f"[C] order-invariance chi2 p={p_c:.4f} ({'PASS' if p_c >= 1e-3 else 'FAIL'})")

    res["ok"] = ok
    (RES / "s3.json").write_text(json.dumps(res, indent=1))
    print("S3:", "GREEN" if ok else "RED", "(sigma — simulated)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
