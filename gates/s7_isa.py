#!/usr/bin/env python3
"""S7σ gate (M21) — the sampling ISA from compiled SIMT kernels through
the D4 socket on the full simt_soc. Bars FROZEN 2026-07-07 in
docs/HARDWARE_ARCHITECTURE.md#sampling-isa §10:

S7.1 trajectory exactness  per-sweep STATE drains == golden, D1-D5 + 100 fuzz
S7.2 statistics mode exact MOMENTS integer-exact on every RECORD run
S7.3 D30/D33 semantics     COV/WORK integer-exact + the independent floating-point anchor
                           (anchor half = golden self-check, re-run here)
S7.4 border tax ≤ 5%       frozen C* workload, B/S from kernel mcycle stamps
S7.5 socket contract       GO-while-busy + divergent-write red paths trap
                           mcause 2; 16-shape G2 subsample green alongside

Per-sweep observation: schedules are expanded into single-sweep IMM
commands (identical trajectory — the farm persists across commands, the
D5 leg proves the split==whole equivalence explicitly), each followed by
a PDRAIN(STATE). Evidence: ci/logs/s7/s7_report.json.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

from mkcuda import Runtime, KernelAssert  # noqa: E402
from golden.sampling_isa import (SClusterGolden, SF_IMM, SF_RECORD,  # noqa: E402
                                 SF_STATS_RESET, build_image,
                                 image_from_graph)
from golden.gibbs_grid import Graph  # noqa: E402
from golden.tensor import gemm8  # noqa: E402

RES = MK / "ci/logs/s7"
S7_SEED_BASE = 0x53370000
TRAP2 = 0x80000000 | 2


def rnz(rng, m=16):
    v = 0
    while v == 0:
        v = int(rng.integers(-m, m + 1))
    return v


def torus(nx, ny, rng, jval=None, king=False):
    g = Graph(nx * ny)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            nbrs = [y * nx + (x + 1) % nx, ((y + 1) % ny) * nx + x]
            if king:
                nbrs += [((y + 1) % ny) * nx + (x + 1) % nx,
                         ((y + 1) % ny) * nx + (x - 1) % nx]
            for j in nbrs:
                g.add_edge(i, j, jval if jval is not None else rnz(rng))
    return g


def bipartite_d014():
    """D-014 shape: 12 visibles + 8 hiddens, hidden degree 8."""
    g = Graph(20)
    for h in range(8):
        for k in range(8):
            g.add_edge(12 + h, (h + k) % 12, 8 if (h + k) % 2 else -8)
    return g


class Plan:
    """Builds the device command table and the golden expectation
    TOGETHER, so both sides consume the identical op sequence."""

    def __init__(self, rt):
        self.rt = rt
        self.gold = SClusterGolden()
        self.rows = []
        self.checks = []
        self._first = SF_STATS_RESET

    def pconfig(self, img, fl):
        self.gold.pconfig(img, fl)
        buf = self.rt.to_device(np.array(img, dtype="<u4"))
        self.rows.append([8, buf.addr, len(img), 0, fl])

    def psample(self, sweeps, beta, flags):
        flags |= self._first
        self._first = 0
        self.gold.psample(flags, t_m=sweeps, t_k=beta)
        self.rows.append([9, 0, sweeps, beta, flags])

    def sweep_step(self, sched, record):
        """Single-sweep IMM expansion of a schedule + STATE drain each."""
        s = 0
        for beta, sweeps in sched:
            for _ in range(sweeps):
                self.psample(1, beta, SF_IMM | (SF_RECORD if record else 0))
                self.pdrain(0, f"state@sweep{s}")
                s += 1

    def pdrain(self, mode, tag):
        want = self.gold.pdrain(mode)
        buf = self.rt.alloc(4 * len(want))
        self.rows.append([10, buf.addr, 0, 0, mode])
        self.checks.append((buf, want, tag))

    def run(self, kern, tag, max_cycles=50_000_000):
        tbl = np.array(self.rows, dtype="<u4").ravel()
        tb = self.rt.to_device(tbl)
        self.rt.launch(kern, grid_n=64, args=[tb, len(self.rows)],
                       max_cycles=max_cycles)
        fails = []
        for buf, want, sub in self.checks:
            got = self.rt.from_device(buf, np.uint32, len(want))
            w = want.astype(np.uint32)
            if not np.array_equal(got, w):
                i = int(np.nonzero(got != w)[0][0])
                fails.append(f"{tag}/{sub}[{i}] got {got[i]:#x} "
                             f"want {w[i]:#x}")
        return fails


def drain_modes(plan, record, work):
    plan.pdrain(0, "state")
    if record:
        plan.pdrain(1, "moments")
        plan.pdrain(2, "cov")
    if work:
        plan.pdrain(3, "work")
    plan.pdrain(4, "telemetry")


def directed(rt, kern):
    fails = []
    rng = np.random.default_rng(0xD1D5)

    # D1: 4x4 deg-4 ferro torus, h=0, binary, beta 32, 60 sweeps
    p = Plan(rt)
    g = torus(4, 4, rng, jval=8)
    img, fl = image_from_graph(g, np.zeros(16, np.int64), [(32, 60)],
                               seed=S7_SEED_BASE + 1)
    p.pconfig(img, fl)
    p.sweep_step([(32, 60)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "D1")
    rt.free_all()

    # D2: 6x6 king deg-8, random J/h, bipolar, beta 48, 40 sweeps
    p = Plan(rt)
    g = torus(6, 6, rng, king=True)
    bias = np.array([rnz(rng, 8) for _ in range(36)], np.int64)
    img, fl = image_from_graph(g, bias, [(48, 40)], seed=S7_SEED_BASE + 2,
                               bipolar=True)
    p.pconfig(img, fl)
    p.sweep_step([(48, 40)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "D2")
    rt.free_all()

    # D3: D2 topology, binary, 20% clamped (order exclusion + STATE)
    p = Plan(rt)
    g = torus(6, 6, rng, king=True)
    bias = np.array([rnz(rng, 8) for _ in range(36)], np.int64)
    clamp = rng.random(36) < 0.2
    st0 = (rng.random(36) < 0.5).astype(np.int8)
    img, fl = image_from_graph(g, bias, [(48, 40)], seed=S7_SEED_BASE + 3,
                               clamp=clamp, state=st0)
    p.pconfig(img, fl)
    p.sweep_step([(48, 40)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "D3")
    rt.free_all()

    # D4: 12+8 bipartite, all visibles clamped -> empty color segment
    p = Plan(rt)
    g = bipartite_d014()
    bias = np.array([rnz(rng, 8) for _ in range(20)], np.int64)
    clamp = np.zeros(20, bool)
    clamp[:12] = True
    st0 = (rng.random(20) < 0.5).astype(np.int8)
    img, fl = image_from_graph(g, bias, [(64, 20)], seed=S7_SEED_BASE + 4,
                               clamp=clamp, state=st0)
    p.pconfig(img, fl)
    p.sweep_step([(64, 20)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "D4")
    rt.free_all()

    # D5: multi-entry schedule via SCHED-mode command, plus the
    # persistence split: 20+20 IMM (no SEEDS reload) vs one 40 IMM
    sched = [(16, 10), (64, 20)]
    g5 = torus(4, 4, rng)
    bias5 = np.array([rnz(rng, 8) for _ in range(16)], np.int64)
    pa = Plan(rt)
    img, fl = image_from_graph(g5, bias5, sched, seed=S7_SEED_BASE + 5)
    pa.pconfig(img, fl)
    pa.psample(0, 0, 0)                        # SCHED form, whole schedule
    drain_modes(pa, record=False, work=False)
    fails += pa.run(kern, "D5-sched")
    rt.free_all()
    pb = Plan(rt)
    pb.pconfig(img, fl)
    pb.psample(20, 33, SF_IMM)
    pb.psample(20, 33, SF_IMM)                 # persistence: no reload
    pb.pdrain(0, "split-state")
    fails += pb.run(kern, "D5-split")
    rt.free_all()
    pc = Plan(rt)
    pc.pconfig(img, fl)
    pc.psample(40, 33, SF_IMM)
    pc.pdrain(0, "whole-state")
    fails += pc.run(kern, "D5-whole")
    if not np.array_equal(pb.gold.s, pc.gold.s):
        fails.append("D5 persistence: golden split != whole")
    rt.free_all()
    return fails


def fuzz(rt, kern, n_cfg=100):
    fails = []
    for i in range(n_cfg):
        rng = np.random.default_rng(S7_SEED_BASE + 0x100 + i)
        king = bool(rng.random() < 0.4)
        nx = int(rng.integers(2, 9))
        ny = int(rng.integers(2, 9))
        n = nx * ny
        if n < 4:
            nx, ny, n = 2, 2, 4
        g = torus(nx, ny, rng, king=king)
        bias = np.array([int(rng.integers(-8, 9)) for _ in range(n)],
                        np.int64)
        bip = bool(rng.random() < 0.5)
        clamp = (rng.random(n) < 0.2) if rng.random() < 0.5 else None
        st0 = (rng.random(n) < 0.5).astype(np.int8)
        n_ent = int(rng.integers(1, 5))
        total = int(rng.integers(n_ent, 65))
        cuts = sorted(rng.choice(np.arange(1, total), n_ent - 1,
                                 replace=False).tolist()) if n_ent > 1 else []
        spans = np.diff([0] + cuts + [total]).tolist()
        sched = [(int(rng.integers(0, 256)), int(sp)) for sp in spans]
        record = bool(rng.random() < 0.7)
        work = bool((not bip) and rng.random() < 0.3)
        p = Plan(rt)
        img, fl = image_from_graph(g, bias, sched,
                                   seed=int(rng.integers(0, 1 << 31)),
                                   bipolar=bip, clamp=clamp, state=st0,
                                   work_track=work)
        p.pconfig(img, fl)
        p.sweep_step(sched, record)
        if work:                     # D33 bracket on-device: rows rewrite
            j2 = np.clip(p.gold.jraw * 2, -511, 511) * p.gold.valid
            rows2 = [([(int(p.gold.nbr[s, k]), int(j2[s, k]))
                       for k in range(8) if p.gold.valid[s, k]],
                      int(bias[s])) for s in range(n)]
            img2, fl2 = build_image(n, rows=rows2, work_track=True)
            p.pconfig(img2, fl2)
        drain_modes(p, record, work)
        f = p.run(kern, f"fuzz{i}")
        fails += f
        if f:
            print(f"  fuzz {i}: {f[0]}")
        rt.free_all()
        if (i + 1) % 20 == 0:
            print(f"  fuzz {i + 1}/{n_cfg} ({len(fails)} fails)")
    return fails


def border_tax(rt, iters=2):
    """S7.4 frozen workload C*: n=1024 32x32 deg-4 torus, seed
    0x5337C0DE. Returns (ratios, S, B) from kernel mcycle stamps."""
    rng = np.random.default_rng(0x5337C0DE)
    g = torus(32, 32, rng)
    bias = np.array([rnz(rng, 8) for _ in range(1024)], np.int64)
    full, ffl = image_from_graph(g, bias, [(128, 1)], seed=0x5337C0DE)
    gold = SClusterGolden()
    gold.pconfig(full, ffl)
    rows = [(g.adj[i], int(bias[i])) for i in range(1024)]
    rimg, rfl = build_image(1024, rows=rows)
    kern = rt.compile(MK / "sw/kernels/s7_ctax.c")
    fb = rt.to_device(np.array(full, dtype="<u4"))
    rb = rt.to_device(np.array(rimg, dtype="<u4"))
    dest = rt.alloc(4 * (1 + 1024 + 8192))
    out = rt.alloc(4 * 4 * iters)
    sf = SF_IMM | SF_RECORD | SF_STATS_RESET
    rt.launch(kern, grid_n=64,
              args=[fb, len(full), ffl, rb, len(rimg), rfl,
                    4096, 128, sf, dest, out, iters],
              max_cycles=400_000_000, chunk=2_000_000)
    st = rt.from_device(out, np.uint32, 4 * iters).reshape(iters, 4)
    ratios = []
    for t0, t1, t2, t3 in st.astype(np.int64):
        S = int(t2 - t1)
        B = int(t3 - t0) - S
        ratios.append(B / S)
    rt.free_all()
    return ratios, st.tolist()


def red_paths(rt):
    """S7.5 fault injection: both must trap mcause 2."""
    fails = []
    rng = np.random.default_rng(1)
    g = torus(8, 8, rng)
    img, fl = image_from_graph(g, np.zeros(64, np.int64), [(32, 1)], seed=2)
    ib = rt.to_device(np.array(img, dtype="<u4"))
    k1 = rt.compile(MK / "sw/kernels/s7_gowb.c")
    try:
        rt.launch(k1, grid_n=64, args=[ib, len(img), fl],
                  max_cycles=3_000_000)
        fails.append("gowb: no trap")
    except KernelAssert as e:
        if not any(c == TRAP2 for _, c in e.failures):
            fails.append(f"gowb: wrong code {e.failures[:2]}")
    rt.free_all()
    k2 = rt.compile(MK / "sw/kernels/s7_div.c")
    try:
        rt.launch(k2, grid_n=64, args=[], max_cycles=3_000_000)
        fails.append("div: no trap")
    except KernelAssert as e:
        if not any(c == TRAP2 for _, c in e.failures):
            fails.append(f"div: wrong code {e.failures[:2]}")
    rt.free_all()
    return fails


def g2_subsample(rt, n_shapes=16):
    """S7.5 coexistence: G2 GEMM shapes green with the S-cluster in."""
    kern = rt.compile(MK / "sw/kernels/gemm_socket.c")
    rng = np.random.default_rng(0x615)
    fails = []
    for _ in range(n_shapes):
        M, N, K = (int(rng.integers(1, 65)) for _ in range(3))
        A = rng.integers(-128, 128, (M, K), dtype=np.int8)
        B = rng.integers(-128, 128, (K, N), dtype=np.int8)
        da, db = rt.to_device(A.ravel()), rt.to_device(B.ravel())
        dc = rt.alloc(4 * M * N)
        rt.launch(kern, grid_n=64, args=[da, db, dc, M, N, K, 0],
                  chunk=100_000)
        got = rt.from_device(dc, np.int32, M * N).reshape(M, N)
        if not np.array_equal(got, gemm8(A, B)):
            fails.append(f"gemm {M}x{N}x{K}")
        rt.free_all()
    return fails


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = {}

    # S7.3 anchor half: the golden self-check (independent forced-state)
    r = subprocess.run([sys.executable, "-m", "golden.sampling_isa"],
                       capture_output=True, text=True, cwd=MK)
    anchor_ok = r.returncode == 0 and "ALL OK" in r.stdout
    rep["anchor"] = {"pass": anchor_ok}
    print(f"S7.3 anchor (golden self-check): {'PASS' if anchor_ok else 'FAIL'}")

    rt = Runtime()
    try:
        kern = rt.compile(MK / "sw/kernels/s7_run.c")

        d_fails = directed(rt, kern)
        print(f"directed D1-D5: {len(d_fails)} fails")
        f_fails = fuzz(rt, kern)
        print(f"fuzz 100: {len(f_fails)} fails")
        suite_fails = d_fails + f_fails
        # S7.1/S7.2/S7.3 device halves all ride the same exact diffs,
        # split by drain tag:
        rep["S7.1"] = {"pass": not [f for f in suite_fails if "state" in f],
                       "fails": [f for f in suite_fails if "state" in f][:8]}
        rep["S7.2"] = {"pass": not [f for f in suite_fails if "moments" in f],
                       "fails": [f for f in suite_fails if "moments" in f][:8]}
        dev33 = [f for f in suite_fails if "cov" in f or "work" in f]
        rep["S7.3"] = {"pass": anchor_ok and not dev33, "fails": dev33[:8]}
        other = [f for f in suite_fails
                 if not any(t in f for t in ("state", "moments", "cov",
                                             "work"))]
        rep["telemetry"] = {"pass": not other, "fails": other[:8]}

        ratios, stamps = border_tax(rt)
        rep["S7.4"] = {"pass": all(r <= 0.05 for r in ratios),
                       "ratios": ratios, "stamps": stamps}
        print(f"S7.4 border tax: {['%.4f' % r for r in ratios]} "
              f"(bar 0.05)")

        rp = red_paths(rt)
        g2 = g2_subsample(rt)
        rep["S7.5"] = {"pass": not rp and not g2,
                       "red_paths": rp, "g2": g2}
        print(f"S7.5 socket contract: red {rp or 'ok'}, G2x16 "
              f"{g2 or 'ok'}")
    finally:
        rt.close()

    rep["wall_s"] = round(time.time() - t0, 1)
    bars = ["S7.1", "S7.2", "S7.3", "S7.4", "S7.5"]
    ok = all(rep[b]["pass"] for b in bars) and rep["telemetry"]["pass"]
    rep["pass"] = ok
    (RES / "s7_report.json").write_text(json.dumps(rep, indent=1))
    for b in bars:
        print(f"  {b}: {'PASS' if rep[b]['pass'] else 'FAIL'}")
    print(f"S7σ: {'ALL PASS' if ok else 'RED'} ({rep['wall_s']}s) -> "
          f"{RES / 's7_report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
