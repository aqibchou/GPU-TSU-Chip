"""fabric_grid bench — trajectory equivalence vs BitTrueGridSampler on:
(a) a random sparse graph (N=24, binary), (b) an 8x8 deg-4 torus with +/-1
couplings (bipolar), (c) the torus with clamped sites (excluded from the
order list per the frozen contract). Counters sanity-checked per sweep.
"""
import cocotb
from cocotb.triggers import RisingEdge

import numpy as np

from golden.gibbs_grid import (BitTrueGridSampler, Graph, build_schedule,
                               torus_king)
from golden.qsite_golden import QsiteGolden, qsite_image
from golden.xoshiro import stream_states
from mkutil import Rng, check, drive_edge, get_nvec, get_seed, sample_edge, start_clock

TB = "fabric_grid"
P = 8


def random_graph(n, seed, dmax=6):
    rng = Rng(seed)
    g = Graph(n)
    have = [0] * n
    for _ in range(n * 2):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j or have[i] >= dmax or have[j] >= dmax:
            continue
        if any(j == x for x, _ in g.adj[i]):
            continue
        g.add_edge(i, j, rng.randint(-16, 16))
        have[i] += 1
        have[j] += 1
    return g


async def wr(dut, sig_we, fields):
    await drive_edge(dut)
    getattr(dut, sig_we).value = 1
    for k, v in fields.items():
        getattr(dut, k).value = v
    await sample_edge(dut)
    await drive_edge(dut)
    getattr(dut, sig_we).value = 0
    await sample_edge(dut)


async def load_all(dut, g, bias, beta, seed, bipolar, clamp=None, s0=None):
    n = g.n
    for i in range(n):
        for k in range(8):
            if k < len(g.adj[i]):
                j, jr = g.adj[i][k]
                word = (1 << 23) | (j << 10) | (jr & 0x3FF)
            else:
                word = 0
            await wr(dut, "row_we", {"row_addr": i, "row_slot": k, "row_data": word})
        await wr(dut, "row_we", {"row_addr": i, "row_slot": 8,
                                 "row_data": int(bias[i]) & 0x3FF})
    _, order, bounds = build_schedule(g, clamp)
    for a, s in enumerate(order):
        await wr(dut, "ord_we", {"ord_addr": a, "ord_data": s})
    for c, (lo, hi) in enumerate(bounds):
        await wr(dut, "cb_we", {"cb_idx": c, "cb_start": lo, "cb_end": hi})
    await drive_edge(dut)
    dut.n_colors.value = len(bounds)
    dut.bipolar.value = int(bipolar)
    await sample_edge(dut)
    await wr(dut, "sch_we", {"sch_idx": 0, "sch_beta": beta, "sch_sweeps": 1})
    await drive_edge(dut)
    dut.n_sched.value = 1
    await sample_edge(dut)
    for i, st in enumerate(stream_states(seed, P)):
        for sel in range(4):
            await wr(dut, "seed_we", {"seed_stream": i, "seed_sel": sel,
                                      "seed_word": st[sel]})
    init = 0
    if s0 is not None:
        for i in range(n):
            init |= int(s0[i]) << i
    for w in range(8):
        await wr(dut, "stw_we", {"stw_addr": w, "stw_data": (init >> (32 * w)) & 0xFFFFFFFF})
    return order


async def run_case(dut, g, bias, beta, seed, bipolar, sweeps, target,
                   clamp=None, s0=None):
    order = await load_all(dut, g, bias, beta, seed, bipolar, clamp, s0)
    gold = BitTrueGridSampler(g, bias, beta, seed, lanes=P, bipolar=bipolar,
                              state0=s0, clamp=clamp)
    upd0 = int(dut.upd_cnt.value)
    for t in range(sweeps):
        await drive_edge(dut)
        dut.start.value = 1
        await sample_edge(dut)
        await drive_edge(dut)
        dut.start.value = 0
        await sample_edge(dut)
        while int(dut.done.value) == 0:
            await RisingEdge(dut.clk)
        exp = gold.sweep()
        got = int(dut.state_flat.value)
        exp_int = 0
        for i in range(g.n):
            exp_int |= int(exp[i]) << i
        check(got & ((1 << g.n) - 1) == exp_int,
              f"state got {got & ((1 << g.n) - 1):#x} want {exp_int:#x} sweep {t}",
              t, seed, TB, target, f"(bip={int(bipolar)})")
    upd = int(dut.upd_cnt.value) - upd0
    check(upd == sweeps * len(order),
          f"upd_cnt got {upd} want {sweeps * len(order)}", 0, seed, TB, target)


async def run(dut, seed, n, target):
    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    for name in ("row_we", "ord_we", "cb_we", "sch_we", "stw_we", "seed_we",
                 "start", "bipolar", "n_colors", "n_sched"):
        getattr(dut, name).value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    await sample_edge(dut)

    t = max(n // 3, 15)
    rng = Rng(seed)
    g1 = random_graph(24, seed)
    b1 = np.array([rng.randint(-64, 64) for _ in range(g1.n)], dtype=np.int32)
    await run_case(dut, g1, b1, 48, seed, False, t, target)

    g2 = torus_king(8, 8, lambda r, c, dr, dc: 8 if (r + c + dr + dc) % 2 else -8,
                    deg4=True)
    b2 = np.zeros(g2.n, dtype=np.int32)
    await run_case(dut, g2, b2, 64, seed + 1, True, t, target)

    clamp = np.zeros(g2.n, dtype=bool)
    clamp[[0, 7, 21, 42]] = True
    s0 = np.zeros(g2.n, dtype=np.int8)
    s0[[0, 21]] = 1
    await run_case(dut, g2, b2, 32, seed + 2, True, t, target,
                   clamp=clamp, s0=s0)


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(45), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(240), "fuzz")


@cocotb.test()
async def d2king(dut):
    """D2-shaped regression (S7 2026-07-11): 6x6 KING graph (deg-8),
    random nonzero J, random bias, bipolar, per-sweep stop/resume with
    multi-chunk colors (4 colors x 9 sites). Closes the deg4=False
    coverage hole the smoke never exercised."""
    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    for name in ("row_we", "ord_we", "cb_we", "sch_we", "stw_we", "seed_we",
                 "start", "bipolar", "n_colors", "n_sched"):
        getattr(dut, name).value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    await sample_edge(dut)

    seed = get_seed() ^ 0xD2
    rng = Rng(seed)
    g = torus_king(6, 6,
                   lambda r, c, dr, dc: (rng.randint(-16, 16) or 1),
                   deg4=False)
    b = np.array([rng.randint(-64, 64) for _ in range(g.n)], dtype=np.int32)
    await run_case(dut, g, b, 48, seed, True, 12, "d2king")


# ---------------- QSITE S2: q4 trajectory equivalence ----------------
async def load_q4(dut, n, rows, order, bounds, beta, seed, s0=None):
    """rows: per site ([(nbr, jraw), ...], [b1, b2, b3])."""
    for i in range(n):
        slots, braw = rows[i]
        for k in range(8):
            if k < len(slots):
                j, jr = slots[k]
                word = (1 << 23) | (j << 10) | (jr & 0x3FF)
            else:
                word = 0
            await wr(dut, "row_we", {"row_addr": i, "row_slot": k,
                                     "row_data": word})
        bw = ((braw[2] & 0x3FF) << 20) | ((braw[1] & 0x3FF) << 10) \
             | (braw[0] & 0x3FF)
        await wr(dut, "row_we", {"row_addr": i, "row_slot": 8,
                                 "row_data": bw})
    for a, site in enumerate(order):
        await wr(dut, "ord_we", {"ord_addr": a, "ord_data": site})
    for c, (lo, hi) in enumerate(bounds):
        await wr(dut, "cb_we", {"cb_idx": c, "cb_start": lo, "cb_end": hi})
    await drive_edge(dut)
    dut.n_colors.value = len(bounds)
    dut.bipolar.value = 0
    dut.arity_q4.value = 1
    await sample_edge(dut)
    await wr(dut, "sch_we", {"sch_idx": 0, "sch_beta": beta,
                             "sch_sweeps": 1})
    await drive_edge(dut)
    dut.n_sched.value = 1
    await sample_edge(dut)
    for i, st in enumerate(stream_states(seed, P)):
        for sel in range(4):
            await wr(dut, "seed_we", {"seed_stream": i, "seed_sel": sel,
                                      "seed_word": st[sel]})
    if s0 is not None:
        nw = (n + 15) // 16
        for w in range(nw):
            word = 0
            for li in range(16):
                i = 16 * w + li
                if i < n:
                    word |= (int(s0[i]) & 3) << (2 * li)
            await wr(dut, "stw_we", {"stw_addr": w, "stw_data": word})


async def run_q4_case(dut, n, rows, order, bounds, beta, seed, sweeps,
                      target, s0=None):
    await load_q4(dut, n, rows, order, bounds, beta, seed, s0)
    g = QsiteGolden()
    w, f = qsite_image(n, 4, rows=[(list((1 << 23) | (j << 10)
                                         | (jr & 0x3FF)
                                         for j, jr in slots), braw)
                                   for slots, braw in rows],
                       order=order, bounds=bounds, n_colors=len(bounds),
                       seed=seed, state=s0)
    g.pconfig(w, f)
    upd0 = int(dut.upd_cnt.value)
    for t in range(sweeps):
        await drive_edge(dut)
        dut.start.value = 1
        await sample_edge(dut)
        await drive_edge(dut)
        dut.start.value = 0
        await sample_edge(dut)
        while int(dut.done.value) == 0:
            await RisingEdge(dut.clk)
        g.psample(0b100, t_m=1, t_k=beta)
        p0 = int(dut.state_flat.value)
        p1 = int(dut.state_flat2.value)
        for i in range(n):
            got = ((p0 >> i) & 1) | (((p1 >> i) & 1) << 1)
            check(got == int(g.x[i]),
                  f"q4 site {i} got {got} want {int(g.x[i])} sweep {t}",
                  t, seed, TB, target)
    upd = int(dut.upd_cnt.value) - upd0
    check(upd == int(g.upd_cnt),
          f"q4 upd_cnt got {upd} want {int(g.upd_cnt)}", 0, seed, TB, target)


def q4_chain(n, seed, jr=6):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        slots = []
        if i > 0:
            slots.append((i - 1, jr))
        if i < n - 1:
            slots.append((i + 1, jr))
        rows.append((slots, [int(v) for v in rng.integers(-6, 7, 3)]))
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    bounds = [(0, (n + 1) // 2), ((n + 1) // 2, n)]
    return rows, order, bounds


@cocotb.test()
async def smoke_q4(dut):
    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    for name in ("row_we", "ord_we", "cb_we", "sch_we", "stw_we", "seed_we",
                 "start", "bipolar", "arity_q4", "n_colors", "n_sched"):
        getattr(dut, name).value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    await sample_edge(dut)
    seed = get_seed(0xC0FFEE)
    # (a) chain, cold start
    rows, order, bounds = q4_chain(24, seed)
    await run_q4_case(dut, 24, rows, order, bounds, 64, seed, 6,
                      "q4 chain")
    # (b) chain, warm start + clamp (site 3 excluded, value 2)
    rows, order, bounds = q4_chain(24, seed + 1)
    order2 = [x for x in order if x != 3]
    b2 = [(0, 11), (11, 23)]
    s0 = np.asarray(np.random.default_rng(seed).integers(0, 4, 24))
    s0[3] = 2
    await run_q4_case(dut, 24, rows, order2, b2, 96, seed + 1, 6,
                      "q4 clamp", s0=s0)


# ---------------- QSITE S4: q8 trajectory equivalence ----------------
async def load_q8(dut, n, rows, order, bounds, beta, seed, s0=None):
    """rows: per site ([(nbr, jraw), ...], [b1..b7]). Arity is set
    BEFORE the row writes — the q8 bias assembly (slots 8/9 stash,
    slot 10 commit) is arity-keyed at load time, exactly as the
    s_cluster loader drives it (header parsed before ROWS)."""
    await drive_edge(dut)
    dut.arity_q4.value = 0
    dut.arity_q8.value = 1
    await sample_edge(dut)
    for i in range(n):
        slots, braw = rows[i]
        for k in range(8):
            if k < len(slots):
                j, jr = slots[k]
                word = (1 << 23) | (j << 10) | (jr & 0x3FF)
            else:
                word = 0
            await wr(dut, "row_we", {"row_addr": i, "row_slot": k,
                                     "row_data": word})
        bws = [((braw[2] & 0x3FF) << 20) | ((braw[1] & 0x3FF) << 10)
               | (braw[0] & 0x3FF),
               ((braw[5] & 0x3FF) << 20) | ((braw[4] & 0x3FF) << 10)
               | (braw[3] & 0x3FF),
               braw[6] & 0x3FF]
        # q8 ROWS: three bias words at slots 8, 9, 10 (spec §12)
        for w8i, bw in enumerate(bws):
            await wr(dut, "row_we", {"row_addr": i, "row_slot": 8 + w8i,
                                     "row_data": bw})
    for a, site in enumerate(order):
        await wr(dut, "ord_we", {"ord_addr": a, "ord_data": site})
    for c, (lo, hi) in enumerate(bounds):
        await wr(dut, "cb_we", {"cb_idx": c, "cb_start": lo, "cb_end": hi})
    await drive_edge(dut)
    dut.n_colors.value = len(bounds)
    dut.bipolar.value = 0
    await sample_edge(dut)
    await wr(dut, "sch_we", {"sch_idx": 0, "sch_beta": beta,
                             "sch_sweeps": 1})
    await drive_edge(dut)
    dut.n_sched.value = 1
    await sample_edge(dut)
    for i, st in enumerate(stream_states(seed, P)):
        for sel in range(4):
            await wr(dut, "seed_we", {"seed_stream": i, "seed_sel": sel,
                                      "seed_word": st[sel]})
    if s0 is not None:
        nw = (n + 7) // 8
        for w in range(nw):
            word = 0
            for li in range(8):
                i = 8 * w + li
                if i < n:
                    word |= (int(s0[i]) & 7) << (4 * li)
            await wr(dut, "stw_we", {"stw_addr": w, "stw_data": word})


async def run_q8_case(dut, n, rows, order, bounds, beta, seed, sweeps,
                      target, s0=None):
    await load_q8(dut, n, rows, order, bounds, beta, seed, s0)
    g = QsiteGolden()
    w, f = qsite_image(n, 8, rows=[(list((1 << 23) | (j << 10)
                                         | (jr & 0x3FF)
                                         for j, jr in slots), braw)
                                   for slots, braw in rows],
                       order=order, bounds=bounds, n_colors=len(bounds),
                       seed=seed, state=s0)
    g.pconfig(w, f)
    upd0 = int(dut.upd_cnt.value)
    for t in range(sweeps):
        await drive_edge(dut)
        dut.start.value = 1
        await sample_edge(dut)
        await drive_edge(dut)
        dut.start.value = 0
        await sample_edge(dut)
        while int(dut.done.value) == 0:
            await RisingEdge(dut.clk)
        g.psample(0b100, t_m=1, t_k=beta)
        p0 = int(dut.state_flat.value)
        p1 = int(dut.state_flat2.value)
        p2 = int(dut.state_flat3.value)
        for i in range(n):
            got = ((p0 >> i) & 1) | (((p1 >> i) & 1) << 1) \
                | (((p2 >> i) & 1) << 2)
            check(got == int(g.x[i]),
                  f"q8 site {i} got {got} want {int(g.x[i])} sweep {t}",
                  t, seed, TB, target)
    upd = int(dut.upd_cnt.value) - upd0
    check(upd == int(g.upd_cnt),
          f"q8 upd_cnt got {upd} want {int(g.upd_cnt)}", 0, seed, TB, target)


def q8_chain(n, seed, jr=6):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        slots = []
        if i > 0:
            slots.append((i - 1, jr))
        if i < n - 1:
            slots.append((i + 1, jr))
        rows.append((slots, [int(v) for v in rng.integers(-6, 7, 7)]))
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    bounds = [(0, (n + 1) // 2), ((n + 1) // 2, n)]
    return rows, order, bounds


@cocotb.test()
async def smoke_q8(dut):
    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    for name in ("row_we", "ord_we", "cb_we", "sch_we", "stw_we", "seed_we",
                 "start", "bipolar", "arity_q4", "arity_q8", "n_colors",
                 "n_sched"):
        getattr(dut, name).value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    await sample_edge(dut)
    seed = get_seed(0xC0FFEE)
    # (a) chain, cold start (odd n: exercises the chunk-parity tail)
    rows, order, bounds = q8_chain(23, seed + 2)
    await run_q8_case(dut, 23, rows, order, bounds, 64, seed + 2, 6,
                      "q8 chain")
    # (b) chain, warm start + clamp (site 5 excluded, value 6)
    rows, order, bounds = q8_chain(24, seed + 3)
    order2 = [x for x in order if x != 5]
    b2 = [(0, 11), (11, 23)]
    s0 = np.asarray(np.random.default_rng(seed + 3).integers(0, 8, 24))
    s0[5] = 6
    await run_q8_case(dut, 24, rows, order2, b2, 96, seed + 3, 6,
                      "q8 clamp", s0=s0)
