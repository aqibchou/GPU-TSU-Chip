#!/usr/bin/env python3
"""S2 gate: small-fabric exactness — RTL histograms vs the EXACT stationary
distribution of the bit-true sweep kernel (golden/gibbs.py pi_hw), plus the
D15 precision dataset TVD(pi_hw, pi_ideal).

════════ FROZEN DECISION RULES (R3, set 2026-07-04 before data collection) ══
Suite: 100 instances (make_instance(SEED_BASE+idx): king-4x4, J ~ U{-16..16}
raw on edges, h ~ U{-8..8} raw) x beta_raw {16, 32, 64}. Per run: burn 2,000
sweeps, thinning K=8, n = 10,000,000 recorded samples, farm streams =
stream_states(SEED_BASE+idx, 16) — all three betas share the instance's
streams (independent runs restart the chain from state 0).
Reference: pi_hw — power-iterated exact stationary distribution of the
quantized chromatic sweep kernel (tol 1e-13). Testing against pi_hw, not
pi_ideal, is deliberate: the RTL must match ITS OWN exact math; the gap to
ideal Boltzmann is reported separately as the D15 measurement (first data
point: TVD ~ 1.5e-5 — far below chi^2 detectability at n=1e7).
Test: Pearson chi^2 with tail pooling — states sorted by pi_hw descending,
states with expected < 5 pooled into one bucket; dof = buckets - 1.
PASS: every run's p-value >= 0.01/300 (Bonferroni, family alpha 0.01).
Thinning diagnostic (pre-registered): f_same = fraction of consecutive
recorded samples identical; under independence E[f_same] = sum(pi_hw^2).
|z| > 5 -> rerun that run once with K=32; a second trip -> RED.
Evidence: ci/logs/s2/run_*.json + tvd.json (the D15 dataset).
════════════════════════════════════════════════════════════════════════════

Modes: --run IDX BETA | --shard K M [--full] | --verdict | --quick (nightly)
"""
import argparse
import json
import pathlib
import struct
import subprocess
import sys
import time

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))

import numpy as np                                     # noqa: E402
from scipy.stats import chi2 as chi2_dist              # noqa: E402

from golden.gibbs import (make_instance, neighbors, pi_hw, pi_ideal,  # noqa: E402
                          tvd)
from golden.xoshiro import stream_states               # noqa: E402

RES = MK / "ci/logs/s2"
REF = RES / "ref"
MC = MK / "tb/fabric16/mc/fabric_mc"
SEED_BASE = 0x2E5EED
N_INST = 100
BETAS = [16, 32, 64]
N_SAMP = 10_000_000
THIN = 8
BURN = 2000
ALPHA = 0.01
M_TESTS = N_INST * len(BETAS)


def write_inputs(idx: int, scratch: pathlib.Path, tag: str = ""):
    J, h = make_instance(SEED_BASE + idx)
    jf = scratch / f"j_{idx}{tag}.bin"
    slots = np.zeros((16, 9), dtype=np.int16)
    for i in range(16):
        for k, j in neighbors(i):
            slots[i, k] = J[i, j]
        slots[i, 8] = h[i]
    jf.write_bytes(slots.tobytes())
    sf = scratch / f"seeds_{idx}{tag}.bin"
    sf.write_bytes(b"".join(struct.pack("<4I", *s)
                            for s in stream_states(SEED_BASE + idx, 16)))
    return J, h, jf, sf


def reference(idx: int, beta: int):
    REF.mkdir(parents=True, exist_ok=True)
    p = REF / f"pi_{idx}_{beta}.npy"
    J, h = make_instance(SEED_BASE + idx)
    if p.exists():
        return np.load(p)
    pi, iters = pi_hw(J, h, beta)
    np.save(p, pi)
    t = tvd(pi, pi_ideal(J, h, beta))
    (REF / f"tvd_{idx}_{beta}.json").write_text(json.dumps(
        {"idx": idx, "beta": beta, "tvd_hw_ideal": t, "power_iters": iters}))
    return pi


def chi2_pooled(hist: np.ndarray, pi: np.ndarray, n: int):
    exp = pi * n
    order = np.argsort(-exp)
    exp_s, obs_s = exp[order], hist[order].astype(np.float64)
    keep = exp_s >= 5.0
    if (~keep).any():
        pool_e, pool_o = exp_s[~keep].sum(), obs_s[~keep].sum()
        exp_k = np.append(exp_s[keep], pool_e)
        obs_k = np.append(obs_s[keep], pool_o)
    else:
        exp_k, obs_k = exp_s, obs_s
    stat = float(((obs_k - exp_k) ** 2 / exp_k).sum())
    dof = len(exp_k) - 1
    return stat, dof, float(chi2_dist.sf(stat, dof))


def one_run(idx: int, beta: int, n=N_SAMP, thin=THIN, quick=False) -> dict:
    scratch = RES / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    _, _, jf, sf = write_inputs(idx, scratch, tag=f"_b{beta}")
    pi = reference(idx, beta)
    hist_f = scratch / f"h_{idx}_{beta}.bin"
    t0 = time.time()
    r = subprocess.run([MC, f"+jfile={jf}", f"+seeds={sf}", f"+beta={beta}",
                        f"+burn={BURN}", f"+thin={thin}", f"+n={n}",
                        f"+hist={hist_f}"], capture_output=True, text=True,
                       timeout=7200)
    if r.returncode or "%" in r.stdout:
        return {"idx": idx, "beta": beta, "ok": False,
                "err": (r.stdout + r.stderr)[:300]}
    same = int(r.stdout.split("same=")[1].split()[0])
    hist = np.frombuffer(hist_f.read_bytes(), dtype="<u4").astype(np.int64)
    hist_f.unlink()
    stat, dof, p = chi2_pooled(hist, pi, n)
    p_same = float((pi ** 2).sum())
    z_same = (same / (n - 1) - p_same) / np.sqrt(p_same * (1 - p_same) / (n - 1))
    out = {"idx": idx, "beta": beta, "n": n, "thin": thin, "chi2": stat,
           "dof": dof, "p": p, "z_same": float(z_same),
           "secs": round(time.time() - t0, 1)}
    if abs(z_same) > 5.0 and thin == THIN and not quick:
        out2 = one_run(idx, beta, n=n, thin=32)
        out2["escalated_from"] = out
        return out2
    out["ok"] = bool((p >= (ALPHA / M_TESTS if not quick else 1e-6))
                     and abs(z_same) <= 5.0)
    return out


def save_run(res: dict):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"run_{res['idx']}_{res['beta']}.json").write_text(json.dumps(res))
    print(f"[{res['idx']:3d},b{res['beta']:3d}] "
          f"{'PASS' if res.get('ok') else 'FAIL'} p={res.get('p', 0):.4g} "
          f"z_same={res.get('z_same', 0):+.2f} ({res.get('secs', '?')}s)")


def verdict() -> int:
    missing, red, tvds = [], [], []
    for idx in range(N_INST):
        for b in BETAS:
            p = RES / f"run_{idx}_{b}.json"
            if not p.exists():
                missing.append((idx, b))
                continue
            r = json.loads(p.read_text())
            if not r.get("ok"):
                red.append((idx, b, r.get("p")))
            tj = REF / f"tvd_{idx}_{b}.json"
            if tj.exists():
                tvds.append(json.loads(tj.read_text())["tvd_hw_ideal"])
    if tvds:
        arr = np.array(tvds)
        (RES / "tvd.json").write_text(json.dumps(
            {"n": len(tvds), "median": float(np.median(arr)),
             "max": float(arr.max()), "mean": float(arr.mean())}, indent=1))
        print(f"D15 dataset: {len(tvds)} refs, TVD median {np.median(arr):.2e} "
              f"max {arr.max():.2e}")
    if missing or red:
        print(f"S2: NOT GREEN — missing={len(missing)} red={red[:5]}")
        return 1
    print(f"S2 GREEN — {M_TESTS} runs, all p >= {ALPHA / M_TESTS:.2e}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, type=int)
    ap.add_argument("--shard", nargs=2, type=int)   # k m
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.verdict:
        return verdict()
    if a.quick:
        ok = True
        for idx in range(4):
            r = one_run(idx, 32, n=1_000_000, quick=True)
            r["ok"] = bool(r.get("p", 0) >= 1e-6
                           and abs(r.get("z_same", 99)) <= 6)
            save_run({**r, "idx": 900 + idx})     # don't collide with gate runs
            ok &= r["ok"]
        print("S2-QUICK:", "GREEN" if ok else "RED")
        return 0 if ok else 1
    if a.run:
        save_run(one_run(a.run[0], a.run[1]))
        return 0
    if a.shard:
        k, m = a.shard
        # longest-first (high beta mixes slowest, escalates most) shrinks the
        # makespan tail that static slicing produced on the first suite run
        jobs = [(i, b) for b in sorted(BETAS, reverse=True) for i in range(N_INST)]
        for i, b in jobs[k::m]:
            if not (RES / f"run_{i}_{b}.json").exists():
                save_run(one_run(i, b))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
