"""M21 exact-diff bench: rtl/fusion/s_cluster.sv vs golden/sampling_isa.py.

Every op is issued through the command face exactly as the simt_core
CSR block will drive it; the DMA rides the v2 edge/credit face
(mem_spec §1b, D-032c): each m_req pulse is one beat, completions
return strictly in order after a settable service latency, several
beats overlap. The diff contract (S7.1-S7.3 RTL halves): after every
PSAMPLE command the fabric state equals golden's; every PDRAIN word
(all 5 modes) equals golden's, bit for bit — at full rate (lat=1) AND
under credit exhaustion (lat=4).
"""
import os
import sys
from collections import deque

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, MK)

from golden.sampling_isa import (DEG_MAX, SClusterGolden, SF_IMM, SF_RECORD,
                                 SF_STATS_RESET, build_image,
                                 image_from_graph)
from golden.gibbs_grid import Graph
from mkutil import drive_edge, get_nvec, get_seed, reset_n, sample_edge, \
    start_clock


class AckMem:
    def __init__(self, size=1 << 22):
        self.b = bytearray(size)
        self.lat = 1                       # service latency, sweepable

    def word(self, addr):
        a = addr % len(self.b)
        return int.from_bytes(self.b[a:a + 4], "little")

    def store(self, addr, data):
        a = addr % len(self.b)
        self.b[a:a + 4] = int(data).to_bytes(4, "little")

    def load_words(self, addr, words):
        for i, w in enumerate(words):
            self.store(addr + 4 * i, int(w) & 0xFFFFFFFF)


async def mem_server(dut, mem):
    """v2 edge/credit slave (mem_spec §1b): drive completions at the
    falling edge (oldest due beat, one pulse per cycle, strictly in
    order), then capture this cycle's m_req in the SAME settled window
    (ReadOnly) — m_req has a combinational leg (the lv shim), so the
    rising-edge post-NBA view would read the NEXT cycle's pulse one
    cycle early, answer while lv_infl is still clear, and the lost ack
    deadlocks the walker. This is the soc_harness capture discipline.
    Service latency mem.lat is sweepable mid-test."""
    q = deque()
    t = 0
    while True:
        await drive_edge(dut)
        if q and q[0][0] <= t:
            _, we, a, wd = q.popleft()
            if we:
                mem.store(a, wd)
                dut.m_rdata.value = 0
            else:
                dut.m_rdata.value = mem.word(a)
            dut.m_rsp_valid.value = 1
        else:
            dut.m_rsp_valid.value = 0
        await ReadOnly()
        if int(dut.m_req.value):
            we = int(dut.m_we.value)
            a = int(dut.m_addr.value)
            wd = int(dut.m_wdata.value) if we else 0
            q.append((t + mem.lat, we, a, wd))
        t += 1


async def issue(dut, op, a=0, c=0, m=0, k=0, flags=0):
    await drive_edge(dut)
    dut.op.value = op - 8
    dut.t_a.value = a
    dut.t_c.value = c
    dut.t_m.value = m
    dut.t_k.value = k
    dut.t_flags.value = flags
    dut.go.value = 1
    await drive_edge(dut)
    dut.go.value = 0
    for _ in range(200_000):
        await sample_edge(dut)
        if not int(dut.busy.value):
            return
    raise AssertionError(
        f"op {op} timeout st={int(dut.st.value)} "
        f"n_sched_r={int(dut.n_sched_r.value)} c_idx={int(dut.c_idx.value)} "
        f"c_sect={int(dut.c_sect.value)} n_sites={int(dut.n_sites.value)} "
        f"dma_left={int(dut.dma_left.value)}")


async def issue_timed(dut, op, a=0, c=0, m=0, k=0, flags=0):
    """issue() plus a busy-cycle count (the ACCWALK overlap tripwire)."""
    await drive_edge(dut)
    dut.op.value = op - 8
    dut.t_a.value = a
    dut.t_c.value = c
    dut.t_m.value = m
    dut.t_k.value = k
    dut.t_flags.value = flags
    dut.go.value = 1
    await drive_edge(dut)
    dut.go.value = 0
    cyc = 0
    for _ in range(200_000):
        await sample_edge(dut)
        cyc += 1
        if not int(dut.busy.value):
            return cyc
    raise AssertionError(f"timed op {op} timeout")


def rtl_state(dut, n):
    v = int(dut.u_fabric.state_flat.value)
    return np.array([(v >> i) & 1 for i in range(n)], dtype=np.int8)


# loader contract (D-027): the fabric zero-sweeps its state store for
# NMAX/32 cycles after reset; no stw/start may arrive inside that
# window (the SoC boot path exceeds it by orders of magnitude; the
# fabric tripwire enforces it in sim)
INIT_SWEEP = 300


async def boot(dut):
    start_clock(dut)
    mem = AckMem()
    await reset_n(dut, zero_signals=("go", "m_rsp_valid"), cycles=4)
    cocotb.start_soon(mem_server(dut, mem))
    for _ in range(INIT_SWEEP):
        await sample_edge(dut)
    return mem


async def new_life(dut):
    """Reset between configs (fresh device life). issue() returns inside
    the ReadOnly phase, so step to a drive point first."""
    await drive_edge(dut)
    await reset_n(dut, cycles=3)
    for _ in range(INIT_SWEEP):
        await sample_edge(dut)


IMG_AT, DRAIN_AT = 0x1000, 0x200000


async def drain_diff(dut, mem, gold, modes):
    """PDRAIN mutates nothing, so every mode drains twice: lat=1 (full
    rate, credits never exhaust) and lat=4 (the pool runs dry — the
    streamer's credit-stall/reload paths). Golden drains in lockstep
    (the telemetry drain-word counter advances per drain); the region
    is poisoned between passes so residue can't fake a pass."""
    for mode in modes:
        for lat in (1, 4):
            mem.lat = lat
            want = gold.pdrain(mode)
            for i in range(len(want)):
                mem.store(DRAIN_AT + 4 * i, 0xA5A5A5A5)
            await issue(dut, 10, c=DRAIN_AT, flags=mode)
            got = np.array([mem.word(DRAIN_AT + 4 * i)
                            for i in range(len(want))], dtype=np.uint32)
            assert np.array_equal(got, want.astype(np.uint32)), \
                f"drain mode {mode} lat {lat} mismatch at " \
                f"{np.nonzero(got != want.astype(np.uint32))[0][:4]}"
        mem.lat = 1


async def pconfig_both(dut, mem, gold, img, fl):
    gold.pconfig(img, fl)
    mem.load_words(IMG_AT, img)
    await issue(dut, 8, a=IMG_AT, m=len(img), flags=fl)


async def run_and_diff(dut, mem, g, bias, sched, seed, *, sweeps_list,
                       bipolar=False, clamp=None, state0=None,
                       record=False, drains=(0, 1, 2, 4), sched_psamples=0):
    """One config: PCONFIG, then per PSAMPLE command exact state diff,
    then every requested drain mode diffed word-for-word."""
    gold = SClusterGolden()
    img, fl = image_from_graph(g, bias, sched, seed=seed, bipolar=bipolar,
                               clamp=clamp, state=state0)
    await pconfig_both(dut, mem, gold, img, fl)
    # spec: accumulators never auto-reset (post-reset content undefined) —
    # each life's first PSAMPLE carries STATS_RESET to define them
    first = SF_STATS_RESET
    for beta, sw in sweeps_list:
        f = SF_IMM | (SF_RECORD if record else 0) | first
        first = 0
        gold.psample(f, t_m=sw, t_k=beta)
        await issue(dut, 9, m=sw, k=beta, flags=f)
        got = rtl_state(dut, g.n)
        assert np.array_equal(got, gold.s[:g.n]), \
            f"state diverged after ({beta},{sw})"
    for _ in range(sched_psamples):
        f = (SF_RECORD if record else 0) | first
        first = 0
        gold.psample(f)
        await issue(dut, 9, flags=f)
        got = rtl_state(dut, g.n)
        assert np.array_equal(got, gold.s[:g.n]), \
            "state diverged after SCHED-mode psample"
    await drain_diff(dut, mem, gold, drains)


def rnz(rng, m=16):
    v = 0
    while v == 0:
        v = int(rng.integers(-m, m + 1))
    return v


def torus(nx, ny, rng, jval=None):
    g = Graph(nx * ny)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for j in (y * nx + (x + 1) % nx, ((y + 1) % ny) * nx + x):
                g.add_edge(i, j, jval if jval is not None else rnz(rng))
    return g


def rings(k, m, rng):
    """k disjoint m-rings — an I-3-safe replica partition (tile t = rid t)."""
    g = Graph(k * m)
    for t in range(k):
        for i in range(m):
            g.add_edge(t * m + i, t * m + (i + 1) % m, rnz(rng))
    return g


def rewrite_rows(gold, jnew, bias):
    """ROWS list for build_image from the golden's current topology."""
    return [([(int(gold.nbr[i, k]), int(jnew[i, k])) for k in range(DEG_MAX)
              if gold.valid[i, k]], int(bias[i])) for i in range(gold.n)]


@cocotb.test()
async def smoke(dut):
    """D1-shaped ferro torus + RECORD + all drains."""
    mem = await boot(dut)
    rng = np.random.default_rng(0xC0FFEE)
    g = torus(4, 4, rng, jval=8)
    await run_and_diff(dut, mem, g, np.zeros(16, np.int64), [(32, 1)],
                       0xD1, sweeps_list=[(32, 5), (32, 5)], record=True,
                       sched_psamples=1)


@cocotb.test()
async def fuzz(dut):
    """Seed-keyed random configs: sizes, spins, clamps, schedules. Each
    config is its own device life (reset between: telemetry counters are
    lifetime totals and n_sites is immutable per life, spec v1)."""
    mem = await boot(dut)
    seed = get_seed()
    n_cfg = get_nvec(12)
    rng = np.random.default_rng(seed)
    for cfg in range(n_cfg):
        await new_life(dut)
        # stage C3: config loads stream on the credit face — exercise
        # them under credit stalls too (drain_diff sweeps its own lat)
        mem.lat = int(rng.choice([1, 1, 4]))
        nx = int(rng.integers(2, 6))
        ny = int(rng.integers(2, 5))
        g = torus(nx, ny, rng)
        n = g.n
        bias = np.array([int(rng.integers(-8, 9)) for _ in range(n)],
                        np.int64)
        bip = bool(rng.random() < 0.5)
        clamp = (rng.random(n) < 0.2) if rng.random() < 0.5 else None
        st0 = (rng.random(n) < 0.5).astype(np.int8)
        sched_img = [(int(rng.integers(0, 256)), int(rng.integers(1, 6)))
                     for _ in range(int(rng.integers(1, 4)))]
        sweeps = [(int(rng.integers(1, 128)), int(rng.integers(1, 12)))
                  for _ in range(int(rng.integers(1, 4)))]
        rec = bool(rng.random() < 0.7)
        n_sp = int(rng.random() < 0.4)
        print(f"[fuzz cfg {cfg}] n={n} bip={bip} "
              f"clamp={None if clamp is None else clamp.astype(int).tolist()} "
              f"rec={rec} sp={n_sp} sched={sched_img} sweeps={sweeps}")
        await run_and_diff(dut, mem, g, bias, sched_img,
                           int(rng.integers(0, 1 << 31)),
                           sweeps_list=sweeps,
                           bipolar=bip, clamp=clamp, state0=st0,
                           record=rec, sched_psamples=n_sp)


@cocotb.test()
async def smoke_min(dut):
    """Minimal bracket: 2 sites, one edge J=8, state held [1,1], rewrite
    J->16. By hand: E2_old=-16, E2_new=-32, WORK=[-16]."""
    mem = await boot(dut)
    g = Graph(2)
    g.add_edge(0, 1, 8)
    bias = np.zeros(2, np.int64)
    st0 = np.ones(2, np.int8)
    gold = SClusterGolden()
    img, fl = image_from_graph(g, bias, [(64, 1)], seed=3, state=st0,
                               work_track=True)
    await pconfig_both(dut, mem, gold, img, fl)
    j2 = np.clip(gold.jraw * 2, -511, 511) * gold.valid
    img2, fl2 = build_image(2, rows=rewrite_rows(gold, j2, bias),
                            work_track=True)
    await pconfig_both(dut, mem, gold, img2, fl2)
    assert int(gold.work[0]) == -16, f"golden self-drift {gold.work}"
    await drain_diff(dut, mem, gold, (3,))


@cocotb.test()
async def smoke_work(dut):
    """D33 work: bracket at command-entry state+rid (spec §5), WORK drain,
    WORK_RESET, replica partition and resize — golden-exact."""
    mem = await boot(dut)
    rng = np.random.default_rng(0xD33)

    # ---- life 1: single replica ----
    g = torus(4, 3, rng)
    n = g.n
    bias = np.array([int(rng.integers(-8, 9)) for _ in range(n)], np.int64)
    st0 = (rng.random(n) < 0.5).astype(np.int8)
    gold = SClusterGolden()
    img, fl = image_from_graph(g, bias, [(48, 2), (96, 1)], seed=7,
                               state=st0, work_track=True)
    await pconfig_both(dut, mem, gold, img, fl)
    gold.psample(SF_RECORD | SF_STATS_RESET)   # multi-entry SCHED-mode
    await issue(dut, 9, flags=SF_RECORD | SF_STATS_RESET)
    assert np.array_equal(rtl_state(dut, n), gold.s[:n]), "life1 sched psample"
    # bracketed ROWS rewrite (2x couplings): charges work at entry state
    j2 = np.clip(gold.jraw * 2, -511, 511) * gold.valid
    img2, fl2 = build_image(n, rows=rewrite_rows(gold, j2, bias),
                            work_track=True)
    await pconfig_both(dut, mem, gold, img2, fl2)
    await drain_diff(dut, mem, gold, (3, 0, 4))
    # ROWS + STATE in one image: bracket must use ENTRY state, not the
    # incoming STATE section (spec §5 "as of command entry")
    st1 = 1 - gold.s[:n].astype(np.int8)
    j3 = -gold.jraw * gold.valid
    img3, fl3 = build_image(n, rows=rewrite_rows(gold, j3, bias), state=st1,
                            work_track=True)
    await pconfig_both(dut, mem, gold, img3, fl3)
    await drain_diff(dut, mem, gold, (3, 0))
    # WORK_RESET zeroes after the bracket
    img4, fl4 = build_image(n, rows=rewrite_rows(gold, j2, bias),
                            work_track=True, work_reset=True)
    await pconfig_both(dut, mem, gold, img4, fl4)
    await drain_diff(dut, mem, gold, (3,))

    # ---- life 2: 3 disjoint rings, rid partition ----
    await new_life(dut)
    g2 = rings(3, 4, rng)
    n2 = g2.n
    bias2 = np.array([int(rng.integers(-8, 9)) for _ in range(n2)], np.int64)
    st02 = (rng.random(n2) < 0.5).astype(np.int8)
    rid = [t for t in range(3) for _ in range(4)]
    gold = SClusterGolden()
    img, fl = image_from_graph(g2, bias2, [(64, 1)], seed=11, state=st02,
                               rid=rid, n_replicas=3, work_track=True)
    await pconfig_both(dut, mem, gold, img, fl)
    gold.psample(SF_IMM | SF_STATS_RESET, t_m=3, t_k=64)
    await issue(dut, 9, m=3, k=64, flags=SF_IMM | SF_STATS_RESET)
    assert np.array_equal(rtl_state(dut, n2), gold.s[:n2]), "life2 psample"
    # per-replica bracket (header restates n_replicas: it applies each
    # PCONFIG — omitting it would resize the replica file to 1)
    j2 = np.clip(gold.jraw * 2, -511, 511) * gold.valid
    img2, fl2 = build_image(n2, rows=rewrite_rows(gold, j2, bias2),
                            n_replicas=3, work_track=True)
    await pconfig_both(dut, mem, gold, img2, fl2)
    await drain_diff(dut, mem, gold, (3, 0))
    # ROWS + RID in one image: bracket must use the ENTRY rid map
    rid2 = [(t + 1) % 3 for t in range(3) for _ in range(4)]
    j3 = -gold.jraw * gold.valid
    img3, fl3 = build_image(n2, rows=rewrite_rows(gold, j3, bias2),
                            rid=rid2, n_replicas=3, work_track=True)
    await pconfig_both(dut, mem, gold, img3, fl3)
    await drain_diff(dut, mem, gold, (3,))
    # unbracketed replica resize: shrink then regrow — regrown read 0
    for r in (1, 3):
        imgr, flr = build_image(n2, n_replicas=r, work_track=True)
        await pconfig_both(dut, mem, gold, imgr, flr)
        await drain_diff(dut, mem, gold, (3,))
    # final bracket on the restored map lands on zeroed high replicas
    img5, fl5 = build_image(n2, rows=rewrite_rows(gold, j2, bias2),
                            n_replicas=3, work_track=True)
    await pconfig_both(dut, mem, gold, img5, fl5)
    await drain_diff(dut, mem, gold, (3, 4))


@cocotb.test()
async def d2king(dut):
    """S7-D2-shaped regression (2026-07-11): gate-exact D1 (ferro 4x4)
    then gate-exact D2 (6x6 KING deg-8, bipolar, multi-chunk colors) as
    TWO device lives with a reset between — the device runtime resets
    the SoC per kernel launch and the D-suite images may omit STATE,
    so reset MUST mean state==0 (the post-reset zero sweep). Also
    closes the deg-8/n=36 coverage hole (fuzz torus is deg-4, n<=20)."""
    mem = await boot(dut)
    # life 1: D1 — ferro 4x4 torus, 60 one-sweep RECORD psamples
    g1 = Graph(16)
    for y in range(4):
        for x in range(4):
            i = y * 4 + x
            for j in (y * 4 + (x + 1) % 4, ((y + 1) % 4) * 4 + x):
                g1.add_edge(i, j, 8)
    gold1 = SClusterGolden()
    img1, fl1 = image_from_graph(g1, np.zeros(16, np.int64), [(32, 60)],
                                 seed=0x53370001)
    await pconfig_both(dut, mem, gold1, img1, fl1)
    first = SF_STATS_RESET
    for s in range(60):
        f = SF_IMM | SF_RECORD | first
        first = 0
        gold1.psample(f, t_m=1, t_k=32)
        await issue(dut, 9, m=1, k=32, flags=f)
        got = rtl_state(dut, g1.n)
        assert np.array_equal(got, gold1.s[:g1.n]), f"D1 diverged sweep {s}"
        await drain_diff(dut, mem, gold1, (0,))
    await drain_diff(dut, mem, gold1, (0, 1, 2))

    # per-launch reset, exactly like the device runtime between suites
    await new_life(dut)

    # life 2: D2 — gate-exact values (rng 0xD1D5; D1 consumes no draws)
    rng = np.random.default_rng(0xD1D5)
    g = Graph(36)
    for y in range(6):
        for x in range(6):
            i = y * 6 + x
            for j in (y * 6 + (x + 1) % 6, ((y + 1) % 6) * 6 + x,
                      ((y + 1) % 6) * 6 + (x + 1) % 6,
                      ((y + 1) % 6) * 6 + (x - 1) % 6):
                g.add_edge(i, j, rnz(rng))
    bias = np.array([rnz(rng, 8) for _ in range(36)], np.int64)
    gold = SClusterGolden()
    img, fl = image_from_graph(g, bias, [(48, 40)], seed=0x53370002,
                               bipolar=True)
    await pconfig_both(dut, mem, gold, img, fl)
    # image has no STATE section: reset semantics must give zeros
    await drain_diff(dut, mem, gold, (0,))
    first = SF_STATS_RESET
    for s in range(40):
        f = SF_IMM | SF_RECORD | first
        first = 0
        gold.psample(f, t_m=1, t_k=48)
        await issue(dut, 9, m=1, k=48, flags=f)
        got = rtl_state(dut, g.n)
        assert np.array_equal(got, gold.s[:g.n]), f"state diverged sweep {s}"
        await drain_diff(dut, mem, gold, (0,))
    await drain_diff(dut, mem, gold, (0, 1, 2))


@cocotb.test()
async def smoke_accwalk(dut):
    """ACCWALK (D-034, sampling_isa_spec §13) overlap seams, diffed
    bit-exact vs the golden: multi-sweep RECORD IMM (walk k overlaps
    sweep k+1), sweeps=1 commands back-to-back (the exposed tail walk
    plus the busy-exit gate — each following drain must read settled
    sums), STATS_RESET landing straight after a walking command
    (P_ZERO vs walker-idle ordering), clamps (held sites through the
    snapshot), binary AND q8 (3-plane snapshot, 7-lane m1). Ends with
    the overlap tripwire: a RECORD command's busy cycles must undercut
    the WALK COST ALONE (sweeps*18*n) — arithmetically impossible for
    the serial P_ACC, which paid walk + sweep."""
    mem = await boot(dut)
    rng = np.random.default_rng(get_seed(0xACC0))
    # life 1: binary, clamps, mixed sweep depths, SCHED tails
    g = torus(8, 8, rng)
    n = g.n
    bias = np.array([int(rng.integers(-8, 9)) for _ in range(n)], np.int64)
    clamp = rng.random(n) < 0.15
    st0 = (rng.random(n) < 0.5).astype(np.int8)
    await run_and_diff(dut, mem, g, bias, [(48, 2), (32, 1)], 0xACC1,
                       sweeps_list=[(32, 8), (16, 1), (16, 1), (40, 3)],
                       clamp=clamp, state0=st0, record=True,
                       sched_psamples=2)
    # life 2 (no DUT reset): the first PSAMPLE's STATS_RESET lands
    # right after life 1's tail walk; then a 16-sweep overlap burst
    # and one exposed single-sweep tail. state0 is EXPLICIT: without a
    # STATE section the DUT keeps life 1's state while a fresh golden
    # starts from zeros — at hot beta the chains coalesce on the shared
    # PRNG stream, so per-command state checks pass while the recorded
    # moments differ by the pre-coalescence sweeps (found the hard way;
    # the old serial P_ACC fails the same shape).
    st2 = (rng.random(n) < 0.5).astype(np.int8)
    # drains exclude TELEMETRY (4): its counters are LIFETIME totals
    # (the reason fuzz resets per config) and life 2 deliberately
    # skips the reset — a fresh golden can only match per-command
    # accumulators (MOMENTS/COV reset via STATS_RESET), not lifetimes
    await run_and_diff(dut, mem, g, bias, [(64, 1)], 0xACC2,
                       sweeps_list=[(24, 16), (24, 1)], record=True,
                       state0=st2, drains=(0, 1, 2))
    # life 3: q8 — RESET+RECORD+IMM overlap burst, then exposed tails
    # back-to-back with a MOMENTS drain after each (no RID section:
    # the RID-loading tests stay last, file convention)
    from golden.qsite_golden import QsiteGolden, qsite_image
    from golden.qsite_golden import enc_slot as q_enc
    await new_life(dut)
    nq = 23
    rows = []
    for i in range(nq):
        slots = []
        if i > 0:
            slots.append(q_enc(i - 1, 6))
        if i < nq - 1:
            slots.append(q_enc(i + 1, 6))
        rows.append((slots, [int(v) for v in rng.integers(-6, 7, 7)]))
    order = list(range(0, nq, 2)) + list(range(1, nq, 2))
    img, fl = qsite_image(nq, 8, rows=rows, order=order,
                          bounds=[(0, 12), (12, 23)], n_colors=2,
                          sched=[(64, 2)], seed=0xACC8,
                          state=[int(v) for v in rng.integers(0, 8, nq)])
    gold = QsiteGolden()
    await pconfig_both(dut, mem, gold, img, fl)
    gold.psample(0b111, t_m=6, t_k=80)
    await issue(dut, 9, m=6, k=80, flags=0b111)
    await drain_diff(dut, mem, gold, [1])
    for _ in range(3):
        gold.psample(0b101, t_m=1, t_k=96)
        await issue(dut, 9, m=1, k=96, flags=0b101)
        await drain_diff(dut, mem, gold, [1])
    await drain_diff(dut, mem, gold, [0, 4])
    # life 4: the overlap tripwire (binary, n=64; the P_ZERO-carrying
    # warmup is untimed so the RESET walk can't mask the measurement)
    await new_life(dut)
    gold2 = SClusterGolden()
    img2, fl2 = image_from_graph(g, bias, [(32, 1)], seed=0xACC4)
    await pconfig_both(dut, mem, gold2, img2, fl2)
    wf = SF_IMM | SF_RECORD | SF_STATS_RESET
    gold2.psample(wf, t_m=1, t_k=32)
    await issue(dut, 9, m=1, k=32, flags=wf)
    sweeps = 8
    gold2.psample(SF_IMM | SF_RECORD, t_m=sweeps, t_k=32)
    cyc = await issue_timed(dut, 9, m=sweeps, k=32,
                            flags=SF_IMM | SF_RECORD)
    walk_alone = sweeps * 18 * n
    print(f"[accwalk tripwire] n={n} sweeps={sweeps}: busy {cyc} cyc "
          f"vs walk-alone bound {walk_alone}")
    assert cyc < walk_alone, \
        f"overlap not engaged: {cyc} >= walk-alone {walk_alone}"
    got = rtl_state(dut, n)
    assert np.array_equal(got, gold2.s[:n]), "tripwire state diverged"
    await drain_diff(dut, mem, gold2, [1, 2])


@cocotb.test()
async def smoke_q4(dut):
    """QSITE S3 (runs LAST: loads nonzero RIDs into the
    config-persistent sh_rid store, which RID-less tests
    inherit by design): full q4 ISA path vs golden/qsite_golden.py — PCONFIG
    (arity header, 3-lane biases, STATE-q, RID), sched+IMM PSAMPLE,
    MOMENTS-q/STATE-q/WORK/TELEMETRY drains, and the D33 E2-q bracket
    on a ROWS-only rewrite."""
    from golden.qsite_golden import QsiteGolden, qsite_image
    from golden.qsite_golden import enc_slot as q_enc
    mem = await boot(dut)
    rng = np.random.default_rng(get_seed(0xC0FFEE))
    n = 24
    def mk_rows(jr):
        rows = []
        for i in range(n):
            slots = []
            if i > 0:
                slots.append(q_enc(i - 1, jr))
            if i < n - 1:
                slots.append(q_enc(i + 1, jr))
            rows.append((slots, [int(v) for v in rng.integers(-6, 7, 3)]))
        return rows
    rows = mk_rows(6)
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    bounds = [(0, 12), (12, 24)]
    st0 = [int(v) for v in rng.integers(0, 4, n)]
    rid = [i % 3 for i in range(n)]
    img, fl = qsite_image(n, 4, rows=rows, order=order, bounds=bounds,
                          n_colors=2, sched=[(64, 3)], seed=0xA5A5,
                          state=st0, rid=rid, work_track=True)
    gold = QsiteGolden()
    await pconfig_both(dut, mem, gold, img, fl)
    gold.psample(0b011)
    await issue(dut, 9, flags=0b011)
    await drain_diff(dut, mem, gold, [0, 1, 3, 4])
    # IMM continuation extends the running sums (I-5)
    gold.psample(0b101, t_m=5, t_k=96)
    await issue(dut, 9, m=5, k=96, flags=0b101)
    await drain_diff(dut, mem, gold, [1, 0])
    # ROWS-only rewrite fires the D33 E2-q bracket (track_pre = 1)
    img2, fl2 = qsite_image(n, 4, rows=mk_rows(4), work_track=True)
    await pconfig_both(dut, mem, gold, img2, fl2)
    await drain_diff(dut, mem, gold, [3, 4])


@cocotb.test()
async def smoke_q8(dut):
    """QSITE S4: full q8 ISA path vs golden/qsite_golden.py — PCONFIG
    (arity 8 header, 11-word rows with 7 bias lanes, STATE-q8, RID),
    sched+IMM PSAMPLE (the double farm advance), MOMENTS-q (7 m1
    lanes)/STATE-q8/WORK/TELEMETRY drains, and the D33 E2-q bracket on
    a ROWS-only rewrite. Runs after smoke_q4 (RID-loading tests last,
    same convention)."""
    from golden.qsite_golden import QsiteGolden, qsite_image
    from golden.qsite_golden import enc_slot as q_enc
    mem = await boot(dut)
    rng = np.random.default_rng(get_seed(0xC0FFEE) + 8)
    n = 23                              # odd: chunk-parity tail
    def mk_rows(jr):
        rows = []
        for i in range(n):
            slots = []
            if i > 0:
                slots.append(q_enc(i - 1, jr))
            if i < n - 1:
                slots.append(q_enc(i + 1, jr))
            rows.append((slots, [int(v) for v in rng.integers(-6, 7, 7)]))
        return rows
    rows = mk_rows(6)
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    bounds = [(0, 12), (12, 23)]
    st0 = [int(v) for v in rng.integers(0, 8, n)]
    rid = [i % 3 for i in range(n)]
    img, fl = qsite_image(n, 8, rows=rows, order=order, bounds=bounds,
                          n_colors=2, sched=[(64, 3)], seed=0xA5A8,
                          state=st0, rid=rid, work_track=True)
    gold = QsiteGolden()
    await pconfig_both(dut, mem, gold, img, fl)
    gold.psample(0b011)
    await issue(dut, 9, flags=0b011)
    await drain_diff(dut, mem, gold, [0, 1, 3, 4])
    # IMM continuation extends the running sums (I-5)
    gold.psample(0b101, t_m=5, t_k=96)
    await issue(dut, 9, m=5, k=96, flags=0b101)
    await drain_diff(dut, mem, gold, [1, 0])
    # ROWS-only rewrite fires the D33 E2-q bracket (track_pre = 1)
    img2, fl2 = qsite_image(n, 8, rows=mk_rows(4), work_track=True)
    await pconfig_both(dut, mem, gold, img2, fl2)
    await drain_diff(dut, mem, gold, [3, 4])
