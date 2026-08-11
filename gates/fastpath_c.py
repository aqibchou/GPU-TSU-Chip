#!/usr/bin/env python3
"""FASTPATH leg C bar gate (D-032c): DMA-phase cycles on converted
paths improve >= 2x at sim memory latency >= 4, word order preserved
exactly (result images bit-identical across builds).

Stages (--stage, each pre-registered in docs/FPGA_IMPLEMENTATION.md#fast-path-optimizations):

  c1  the PDRAIN streamer. Workload = the S7.4 frozen C* shape
      (n=1024 torus): kernel mcycle stamps fence PCONFIG(ROWS) /
      PSAMPLE / PDRAIN(MOMENTS). Measured = the drain phase; the ROWS
      config phase is the UNCONVERTED control. Baseline = stage C0
      (aefbef2), timing-identical to the v1 face by the C0 proof.

  c2  the sidecar loaders (LD_A/LD_B/LD_C0/ST_C). Workload =
      sw/kernels/fastpath_gemm.c: per iteration GEMM8 ops at N=1 and
      N=8 (M=K=64 fixed). MEASURED = the DIFFERENTIAL dDMA =
      op(N8) - op(N1): COMP cancels exactly (same ceil(N/8) loop
      count), LD_A and doorbell/poll overhead cancel — the difference
      is pure converted-path DMA (LD_B 112 + ST_C 448 beats). This is
      the card's 2026-07-15 instrument amendment: the original
      whole-op shapes had arithmetic errors (COMP lane padding;
      ST_C word beats), banked in the card; the >= 2x bar is
      unchanged. Controls = the C* drain+config phases re-run.
      Baseline = the branch fork point from main (post-C1 = pre-C2
      for the tensor path). The measurement kernel compiles from THIS
      tree for BOTH builds (identical stimulus; the baseline tree
      predates the kernel file).

The baseline builds in its own git worktree; this gate retargets its
OWN mkcuda at each tree (MK/RT_DIR/HARNESS globals), so both builds
are driven by the same fixed instrument (the 2026-07-14 gate-tree
finding: baseline trees predate the mkcuda fix).

PASS c1: drain ratio >= 2.0 at LAT {4,8}; drain sha identical per
LAT; config control in [0.7, 1.4].
PASS c2: dDMA ratio >= 2.0 at LAT {4,8}; C-result sha and drain sha
identical per LAT; controls (C* drain, C* config) each in
[0.8, 1.25]; N=1 whole-op ratio recorded informationally.
Evidence: ci/logs/fastpath/<stage>_bar.json (in the measured tree).
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

import mkcuda                                                    # noqa: E402

from golden.gibbs_grid import Graph                              # noqa: E402
from golden.sampling_isa import (SF_IMM, SF_RECORD,              # noqa: E402
                                 SF_STATS_RESET, build_image,
                                 image_from_graph)

BASE_REV_C1 = "aefbef2"       # stage C0: pre-conversion timing shape
LATS = (1, 4, 8)
ITERS = 3
N = 1024
DRAIN_WORDS = 1 + N + 8 * N   # MOMENTS: cnt + m1 + m2
GEMM_N1 = (64, 1, 64)         # (M, N, K) — differential minuend pair:
GEMM_N8 = (64, 8, 64)         # dDMA = op(N8) - op(N1); COMP/LD_A cancel


def rnz(rng, m=16):
    v = 0
    while v == 0:
        v = int(rng.integers(-m, m + 1))
    return v


def torus(nx, ny, rng):
    g = Graph(nx * ny)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for j in (y * nx + (x + 1) % nx, ((y + 1) % ny) * nx + x):
                g.add_edge(i, j, rnz(rng))
    return g


def measure_cstar(rt, tree, out):
    """C* phases per LAT: config (t1-t0) and drain (t3-t2)."""
    rng = np.random.default_rng(0x5337C0DE)
    g = torus(32, 32, rng)
    bias = np.array([rnz(rng, 8) for _ in range(N)], np.int64)
    full, ffl = image_from_graph(g, bias, [(128, 1)], seed=0x5337C0DE)
    rows = [(g.adj[i], int(bias[i])) for i in range(N)]
    rimg, rfl = build_image(N, rows=rows)
    kern = rt.compile(tree / "sw" / "kernels" / "s7_ctax.c")
    for lat in LATS:
        rt.set_latency(lat)
        fb = rt.to_device(np.array(full, dtype="<u4"))
        rb = rt.to_device(np.array(rimg, dtype="<u4"))
        dest = rt.alloc(4 * DRAIN_WORDS)
        st_b = rt.alloc(4 * 4 * ITERS)
        sf = SF_IMM | SF_RECORD | SF_STATS_RESET
        rt.launch(kern, grid_n=64,
                  args=[fb, len(full), ffl, rb, len(rimg), rfl,
                        8, 128, sf, dest, st_b, ITERS],
                  max_cycles=60_000_000, chunk=500_000)
        st = rt.from_device(st_b, np.uint32, 4 * ITERS)
        st = st.reshape(ITERS, 4).astype(np.int64)
        drain = rt.from_device(dest, np.uint32, DRAIN_WORDS).tobytes()
        cfg = [int(t1 - t0) for t0, t1, t2, t3 in st[1:]]
        drn = [int(t3 - t2) for t0, t1, t2, t3 in st[1:]]
        d = out["lats"].setdefault(str(lat), {})
        d.update({"config": statistics.median(cfg),
                  "drain": statistics.median(drn),
                  "cstar_stamps": st.tolist(),
                  "drain_sha": hashlib.sha256(drain).hexdigest()})
        print(f"  [{tree.name} LAT={lat}] C* config={cfg} drain={drn}")
        rt.free_all()


def measure_gemm(rt, tree, out):
    """Stage-C2 differential per LAT (card amendment 2026-07-15):
    dDMA = op(N=8) - op(N=1) at fixed M=K — COMP (same ceil(N/8)),
    LD_A, and doorbell/poll overhead cancel; the difference is pure
    converted-path DMA (LD_B + ST_C). Kernel and operand data come
    from the GATE's own tree so both builds see identical stimulus."""
    m1, n1, k1 = GEMM_N1
    m2, n2, k2 = GEMM_N8
    rng = np.random.default_rng(0xC2C2)
    a_bytes = rng.integers(-128, 128, size=m2 * k2, dtype=np.int8)
    b_bytes = rng.integers(-128, 128, size=k2 * n2, dtype=np.int8)
    kern = rt.compile(MK / "sw" / "kernels" / "fastpath_gemm.c")
    for lat in LATS:
        rt.set_latency(lat)
        ab = rt.to_device(a_bytes.view(np.uint8))
        bb = rt.to_device(b_bytes.view(np.uint8))
        cb = rt.alloc(4 * m2 * n2)
        st_b = rt.alloc(4 * 3 * ITERS)
        rt.launch(kern, grid_n=64,
                  args=[ab, bb, cb, m1, n1, k1, m2, n2, k2, st_b, ITERS],
                  max_cycles=20_000_000, chunk=200_000)
        st = rt.from_device(st_b, np.uint32, 3 * ITERS)
        st = st.reshape(ITERS, 3).astype(np.int64)
        c_img = rt.from_device(cb, np.uint32, m2 * n2).tobytes()
        op_n1 = [int(t1 - t0) for t0, t1, t2 in st[1:]]
        op_n8 = [int(t2 - t1) for t0, t1, t2 in st[1:]]
        ddma = [b - a for a, b in zip(op_n1, op_n8)]
        d = out["lats"].setdefault(str(lat), {})
        d.update({"ddma": statistics.median(ddma),
                  "op_n1": statistics.median(op_n1),
                  "op_n8": statistics.median(op_n8),
                  "gemm_stamps": st.tolist(),
                  "c_sha": hashlib.sha256(c_img).hexdigest()})
        print(f"  [{tree.name} LAT={lat}] op_n1={op_n1} op_n8={op_n8} "
              f"ddma={ddma}")
        rt.free_all()


def measure(tree: pathlib.Path, stage: str):
    """Run the stage's workloads on `tree`'s harness. Retargets
    mkcuda's globals at the tree (RT_DIR from the same tree keeps the
    cache key honest; toolchain/cache stay main-tree per mkcuda)."""
    tree = tree.resolve()
    mkcuda.MK = tree
    mkcuda.RT_DIR = tree / "sw" / "rt"
    mkcuda.HARNESS = tree / "tb" / "simt_soc" / "build" / "soc_harness"
    rt = mkcuda.Runtime()
    try:
        rev = subprocess.run(["git", "-C", str(tree), "rev-parse",
                              "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        out = {"tree": str(tree), "rev": rev, "lats": {}}
        measure_cstar(rt, tree, out)
        if stage in ("c2", "c3"):
            measure_gemm(rt, tree, out)
        return out
    finally:
        rt.close()


def ensure_baseline(base_dir: pathlib.Path, rev: str):
    """The baseline worktree is a measurement CACHE keyed by rev —
    create it detached, retarget it when a later stage wants a
    different fork point (checkout touches files, so the harness
    rebuild is automatic via make's dependency tracking)."""
    if not (base_dir / "sw" / "kernels" / "s7_ctax.c").exists():
        subprocess.run(["git", "-C", str(MK), "worktree", "add",
                        "--detach", str(base_dir), rev], check=True)
    def head():
        return subprocess.run(["git", "-C", str(base_dir), "rev-parse",
                               "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    got = head()
    if not (got.startswith(rev) or rev.startswith(got)):
        subprocess.run(["git", "-C", str(base_dir), "checkout",
                        "--detach", rev], check=True)
        got = head()
    assert got.startswith(rev) or rev.startswith(got), \
        f"baseline worktree at {got}, want {rev} — remove {base_dir}"


def band(x, lo, hi):
    return lo <= x <= hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("c1", "c2", "c3"), default="c1")
    ap.add_argument("--base-rev", default=None)
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()
    stage = args.stage

    base_rev = args.base_rev
    if base_rev is None:
        if stage == "c1":
            base_rev = BASE_REV_C1
        else:
            # c2: pre-C2 fork point; c3: pre-C3 fork point — both are
            # merge-base with main at the time the stage branched
            # pre-C2 = the fork point from main (post-C1 tensor path)
            base_rev = subprocess.run(
                ["git", "-C", str(MK), "merge-base", "HEAD", "main"],
                capture_output=True, text=True).stdout.strip()[:9]
    base_dir = pathlib.Path(args.base_dir) if args.base_dir else \
        MK.parent / (MK.name + "-base")

    print(f"[{stage} bar] new tree: {MK}")
    new = measure(MK, stage)
    ensure_baseline(base_dir, base_rev)
    print(f"[{stage} bar] baseline: {base_dir} @ {base_rev}")
    base = measure(base_dir, stage)

    rep = {"stage": stage, "new": new, "base": base, "bars": {}}
    ok = True
    for lat in ("4", "8"):
        bl, nl = base["lats"][lat], new["lats"][lat]
        if stage == "c1":
            r = bl["drain"] / max(1, nl["drain"])
            c = bl["config"] / max(1, nl["config"])
            v = bl["drain_sha"] == nl["drain_sha"]
            a = r >= 2.0
            b = band(c, 0.7, 1.4)
            rep["bars"][lat] = {"drain_ratio": r, "config_ratio": c,
                                "values_identical": v,
                                "C1.a": a, "C1.b": v, "C1.c": b}
            ok &= a and v and b
            print(f"[c1 bar] LAT={lat}: drain {bl['drain']} -> "
                  f"{nl['drain']} cyc = {r:.2f}x (bar >= 2.0) "
                  f"{'PASS' if a else 'FAIL'}; config control {c:.2f}x "
                  f"{'ok' if b else 'MOVED'}; values "
                  f"{'identical' if v else 'DIFFER'}")
        elif stage == "c3":
            # measured = the config phase (the last conversion);
            # controls = both already-converted paths sit still
            r = bl["config"] / max(1, nl["config"])
            ctl = {"cstar_drain": bl["drain"] / max(1, nl["drain"]),
                   "gemm_ddma": bl["ddma"] / max(1, nl["ddma"])}
            v = (bl["c_sha"] == nl["c_sha"]
                 and bl["drain_sha"] == nl["drain_sha"])
            a = r >= 2.0
            b = all(band(x, 0.8, 1.25) for x in ctl.values())
            rep["bars"][lat] = {"config_ratio": r, "controls": ctl,
                                "values_identical": v,
                                "C3.a": a, "C3.b": v, "C3.c": b}
            ok &= a and v and b
            print(f"[c3 bar] LAT={lat}: config {bl['config']} -> "
                  f"{nl['config']} cyc = {r:.2f}x (bar >= 2.0) "
                  f"{'PASS' if a else 'FAIL'}; controls "
                  + " ".join(f"{k}={x:.2f}x" for k, x in ctl.items())
                  + f" {'ok' if b else 'MOVED'}; values "
                  f"{'identical' if v else 'DIFFER'}")
        else:
            r = bl["ddma"] / max(1, nl["ddma"])
            ctl = {"cstar_drain": bl["drain"] / max(1, nl["drain"]),
                   "cstar_config": bl["config"] / max(1, nl["config"])}
            v = (bl["c_sha"] == nl["c_sha"]
                 and bl["drain_sha"] == nl["drain_sha"])
            a = r >= 2.0
            b = all(band(x, 0.8, 1.25) for x in ctl.values())
            info_n1 = bl["op_n1"] / max(1, nl["op_n1"])
            rep["bars"][lat] = {"ddma_ratio": r, "controls": ctl,
                                "op_n1_ratio_info": info_n1,
                                "values_identical": v,
                                "C2.a": a, "C2.b": v, "C2.c": b}
            ok &= a and v and b
            print(f"[c2 bar] LAT={lat}: dDMA {bl['ddma']} -> "
                  f"{nl['ddma']} cyc = {r:.2f}x (bar >= 2.0) "
                  f"{'PASS' if a else 'FAIL'}; controls "
                  + " ".join(f"{k}={x:.2f}x" for k, x in ctl.items())
                  + f" {'ok' if b else 'MOVED'}; values "
                  f"{'identical' if v else 'DIFFER'}; "
                  f"n1 whole-op {info_n1:.2f}x (info)")
    key = {"c1": "drain", "c2": "ddma", "c3": "config"}[stage]
    r1 = base["lats"]["1"][key] / max(1, new["lats"]["1"][key])
    rep["bars"]["1_info"] = {f"{key}_ratio": r1}
    print(f"[{stage} bar] LAT=1 (info): {r1:.2f}x")

    ev = MK / "ci" / "logs" / "fastpath"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / f"{stage}_bar.json").write_text(json.dumps(rep, indent=1))
    print(f"[{stage} bar] evidence: {ev / (stage + '_bar.json')}")
    print(f"FASTPATH-{stage.upper()}: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
