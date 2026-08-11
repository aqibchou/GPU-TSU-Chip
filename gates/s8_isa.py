#!/usr/bin/env python3
"""S8σ gate — the q-ary sampling ISA certified ON the SoC (QSITE S4).
Bars frozen in docs/HARDWARE_ARCHITECTURE.md#categorical-q-site-architecture "S4 pre-registration" BEFORE any
measurement:

  S8.1  trajectory exactness: device STATE-q drains == golden after
        EVERY sweep, zero tolerance, Q1-Q5 + 60 fuzz configs
        (q in {4,8}).
  S8.2  statistics exact: MOMENTS-q (cnt, m1[site][a], m2 agreement)
        integer-exact on every RECORD run; TELEMETRY upd/flip exact.
  S8.3  work/E2-q: WORK drains integer-exact incl. the D33 bracket on
        a live q8 ROWS rewrite.
  S8.4  regression: S7σ + battery green on the same tree (run
        separately; cited in the close-out, not re-run here).

All runs from compiled kernels through the D4 socket on simt_soc —
sw/kernels/s7_run.c is a generic op-table replayer and is reused
verbatim. COV-q is deferred (spec §12): mode 2 is never drained on
q images. Evidence: ci/logs/s8/s8_report.json.
"""
import json
import pathlib
import sys
import time

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

from mkcuda import Runtime  # noqa: E402
from golden.gibbs_grid import Graph, build_schedule  # noqa: E402
from golden.qsite_golden import (QsiteGolden, enc_slot,  # noqa: E402
                                 qsite_image)

RES = MK / "ci/logs/s8"
S8_SEED_BASE = 0x53380000
SF_RECORD, SF_STATS_RESET, SF_IMM = 1, 2, 4


class QPlan:
    """gates/s7_isa.py's Plan, pointed at the q-ary oracle: builds the
    device command table and the golden expectation together."""

    def __init__(self, rt):
        self.rt = rt
        self.gold = QsiteGolden()
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

    def psample_sched(self, flags):
        flags |= self._first
        self._first = 0
        self.gold.psample(flags)
        self.rows.append([9, 0, 0, 0, flags])

    def sweep_step(self, sched, record):
        """Single-sweep IMM expansion + a STATE drain after each (the
        S8.1 per-sweep observation)."""
        s = 0
        for beta, sweeps in sched:
            for _ in range(sweeps):
                self.psample(1, beta, SF_IMM | (SF_RECORD if record else 0))
                self.pdrain(0, f"state@sweep{s}")
                s += 1

    def pdrain(self, mode, tag):
        want = np.asarray(self.gold.pdrain(mode))
        buf = self.rt.alloc(4 * len(want))
        self.rows.append([10, buf.addr, 0, 0, mode])
        self.checks.append((buf, want, tag))

    def run(self, kern, tag, max_cycles=80_000_000):
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
        plan.pdrain(1, "moments")            # COV-q deferred: no mode 2
    if work:
        plan.pdrain(3, "work")
    plan.pdrain(4, "telemetry")


def rand_qgraph(rng, n, q):
    """Delta-coupled sparse graph (deg <= 8) + q-1 bias lanes/site +
    a proper-coloring schedule with clamp prob 0.2."""
    g = Graph(n)
    deg = np.zeros(n, int)
    seen = set()
    for _ in range(2 * n):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j or (min(i, j), max(i, j)) in seen:
            continue
        if deg[i] >= 8 or deg[j] >= 8:
            continue
        jv = 0
        while jv == 0:
            jv = int(rng.integers(-16, 17))
        g.add_edge(i, j, jv)
        seen.add((min(i, j), max(i, j)))
        deg[i] += 1
        deg[j] += 1
    rows = []
    for i in range(n):
        slots = [enc_slot(nb, jv) for nb, jv in g.adj[i]]
        braw = [int(v) for v in rng.integers(-8, 9, q - 1)]
        rows.append((slots, braw))
    clamp = rng.random(n) < 0.2
    _, order, bounds = build_schedule(g, clamp)
    st0 = [int(v) for v in rng.integers(0, q, n)]
    return rows, order, bounds, clamp, st0


def chain_rows(rng, n, q, jr=6):
    rows = []
    for i in range(n):
        slots = []
        if i > 0:
            slots.append(enc_slot(i - 1, jr))
        if i < n - 1:
            slots.append(enc_slot(i + 1, jr))
        rows.append((slots, [int(v) for v in rng.integers(-6, 7, q - 1)]))
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    bounds = [(0, (n + 1) // 2), ((n + 1) // 2, n)]
    return rows, order, bounds


def king_rows(rng, n_side, q):
    n = n_side * n_side
    g = Graph(n)
    for y in range(n_side):
        for x in range(n_side):
            i = y * n_side + x
            for j in (y * n_side + (x + 1) % n_side,
                      ((y + 1) % n_side) * n_side + x,
                      ((y + 1) % n_side) * n_side + (x + 1) % n_side,
                      ((y + 1) % n_side) * n_side + (x - 1) % n_side):
                jv = 0
                while jv == 0:
                    jv = int(rng.integers(-16, 17))
                g.add_edge(i, j, jv)
    rows = []
    for i in range(n):
        slots = [enc_slot(nb, jv) for nb, jv in g.adj[i]]
        rows.append((slots, [int(v) for v in rng.integers(-8, 9, q - 1)]))
    _, order, bounds = build_schedule(g, np.zeros(n, bool))
    return g, rows, order, bounds


def directed(rt, kern):
    fails = []
    rng = np.random.default_rng(0xD8D8)

    # Q1: q4 ferro-delta chain (the S2 shape as a full ISA image)
    p = QPlan(rt)
    rows, order, bounds = chain_rows(rng, 24, 4)
    img, fl = qsite_image(24, 4, rows=rows, order=order, bounds=bounds,
                          n_colors=len(bounds), seed=S8_SEED_BASE + 1)
    p.pconfig(img, fl)
    p.sweep_step([(64, 20)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "Q1")
    rt.free_all()

    # Q2: 6x6 king q8 + the D33 bracket on a live ROWS rewrite (S8.3)
    p = QPlan(rt)
    g, rows, order, bounds = king_rows(rng, 6, 8)
    rid = [i % 4 for i in range(36)]
    img, fl = qsite_image(36, 8, rows=rows, order=order, bounds=bounds,
                          n_colors=len(bounds), seed=S8_SEED_BASE + 2,
                          state=[int(v) for v in rng.integers(0, 8, 36)],
                          rid=rid, work_track=True)
    p.pconfig(img, fl)
    p.sweep_step([(48, 20)], record=True)
    rows2 = [(s, [int(v) for v in rng.integers(-8, 9, 7)])
             for s, _ in rows]
    img2, fl2 = qsite_image(36, 8, rows=rows2, work_track=True)
    p.pconfig(img2, fl2)                     # ROWS-only: bracket fires
    p.sweep_step([(48, 10)], record=True)
    drain_modes(p, record=True, work=True)
    fails += p.run(kern, "Q2")
    rt.free_all()

    # Q3: Q2 topology, 25% clamped to values >= 4 (STATE-q + exclusion)
    p = QPlan(rt)
    g, rows, order, bounds = king_rows(rng, 6, 8)
    clamp = rng.random(36) < 0.25
    st0 = [int(v) for v in rng.integers(0, 8, 36)]
    for i in range(36):
        if clamp[i]:
            st0[i] = int(rng.integers(4, 8))
    _, order, bounds = build_schedule(g, clamp)
    img, fl = qsite_image(36, 8, rows=rows, order=order, bounds=bounds,
                          n_colors=len(bounds), seed=S8_SEED_BASE + 3,
                          state=st0)
    p.pconfig(img, fl)
    p.sweep_step([(64, 24)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "Q3")
    rt.free_all()

    # Q4: D-014-q8 shape — empty color segment + odd-count chunk tail
    p = QPlan(rt)
    n = 21                                    # 12 vis + 9 hid, odd tail
    g = Graph(n)
    for h in range(9):
        for k in range(6):
            g.add_edge(12 + h, (h + k) % 12, 8 if (h + k) % 2 else -8)
    clamp = np.zeros(n, bool)
    clamp[:12] = True                         # visibles clamped: color 0
    rows = [( [enc_slot(nb, jv) for nb, jv in g.adj[i]],
              [int(v) for v in rng.integers(-6, 7, 7)]) for i in range(n)]
    _, order, bounds = build_schedule(g, clamp)
    st0 = [int(v) for v in rng.integers(0, 8, n)]
    img, fl = qsite_image(n, 8, rows=rows, order=order, bounds=bounds,
                          n_colors=len(bounds), seed=S8_SEED_BASE + 4,
                          state=st0)
    p.pconfig(img, fl)
    p.sweep_step([(96, 30)], record=True)
    drain_modes(p, record=True, work=False)
    fails += p.run(kern, "Q4")
    rt.free_all()

    # Q5: q8 persistence — split IMM vs whole, same farm (I-12/D5)
    for tag, split in (("Q5w", False), ("Q5s", True)):
        p = QPlan(rt)
        rows, order, bounds = chain_rows(rng, 17, 8)
        img, fl = qsite_image(17, 8, rows=rows, order=order,
                              bounds=bounds, n_colors=len(bounds),
                              seed=S8_SEED_BASE + 5)
        p.pconfig(img, fl)
        if split:
            p.sweep_step([(16, 10)], record=True)
            p.sweep_step([(64, 20)], record=True)
        else:
            p.sweep_step([(16, 10), (64, 20)], record=True)
        drain_modes(p, record=True, work=False)
        fails += p.run(kern, tag)
        rt.free_all()
    return fails


def fuzz(rt, kern, count=60):
    fails = []
    for i in range(count):
        rng = np.random.default_rng(S8_SEED_BASE + 100 + i)
        q = 8 if (i % 2) else 4
        n = int(rng.integers(4, 65))
        rows, order, bounds, clamp, st0 = rand_qgraph(rng, n, q)
        nsch = int(rng.integers(1, 4))
        sched = [(int(rng.integers(8, 129)), int(rng.integers(1, 17)))
                 for _ in range(nsch)]
        rid = [int(v) for v in rng.integers(0, 3, n)] \
            if rng.random() < 0.5 else None
        wt = rid is not None
        p = QPlan(rt)
        img, fl = qsite_image(n, q, rows=rows, order=order, bounds=bounds,
                              n_colors=len(bounds),
                              seed=S8_SEED_BASE + 200 + i, state=st0,
                              rid=rid, sched=sched, work_track=wt)
        p.pconfig(img, fl)
        p.sweep_step(sched, record=bool(rng.random() < 0.8))
        if rng.random() < 0.3:               # SCHED-form coverage
            p.psample_sched(SF_RECORD)
        drain_modes(p, record=True, work=wt)
        fails += p.run(kern, f"fuzz{i}")
        rt.free_all()
        if (i + 1) % 10 == 0:
            print(f"  fuzz {i + 1}/{count} ({len(fails)} fails)",
                  flush=True)
    return fails


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rt = Runtime()
    kern = rt.compile(MK / "sw/kernels/s7_run.c")
    f_dir = directed(rt, kern)
    print(f"directed Q1-Q5: {len(f_dir)} fails", flush=True)
    f_fuzz = fuzz(rt, kern)
    print(f"fuzz 60: {len(f_fuzz)} fails", flush=True)
    rt.close()
    for f in (f_dir + f_fuzz)[:10]:
        print(" ", f)
    state_f = [f for f in f_dir + f_fuzz if "/state" in f]
    mom_f = [f for f in f_dir + f_fuzz
             if "/moments" in f or "/telemetry" in f]
    work_f = [f for f in f_dir + f_fuzz if "/work" in f]
    rep = {
        "s8_1_pass": not state_f,
        "s8_2_pass": not mom_f,
        "s8_3_pass": not work_f,
        "fails_total": len(f_dir) + len(f_fuzz),
        "s8_4": "battery + S7σ run separately on the same tree "
                "(close-out evidence)",
        "verdict": "PASS" if not (f_dir or f_fuzz) else "FAIL",
        "wall_s": round(time.time() - t0, 1),
    }
    with open(RES / "s8_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
