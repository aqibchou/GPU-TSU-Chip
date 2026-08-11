"""RTL-PCD equivalence rig (M8/S5σ): the full PCD loop with the FABRIC as the
sampler, proven state-identical to the bit-true golden at every phase run.

Reduced config (frozen): 12x12 double-grid, jumps [1,4] (deg 8, 288 nodes),
40 visible upper nodes, synthetic 3-pattern data, B=6 wake items, 8 persistent
dream chains, k_wake=1, m_dream=2, T=25 PCD steps, beta raw 64.

Per phase run both sides get identical derived seeds and identical initial
states; the RTL end-state must equal the golden end-state EXACTLY, every run.
Moments and master updates then flow through ONE PcdTrainer.apply_update —
by induction the weight trajectory is bit-identical, no tolerances anywhere.
"""
import cocotb
from cocotb.triggers import RisingEdge

import numpy as np

from golden.gibbs_grid import BitTrueGridSampler, Graph, build_schedule
from golden.pcd import PcdTrainer, quantize_s163
from golden.xoshiro import stream_states
from mkutil import check, drive_edge, sample_edge, start_clock

TB = "fabric_grid"
P = 8
SIDE, N_UP = 12, 144
N = 2 * N_UP
N_VIS, B, C = 40, 6, 8
T_STEPS, K_WAKE, M_DREAM = 25, 1, 2
BETA = 64


def build_graph():
    g = Graph(N)
    seen = set()
    for idx in range(N_UP):
        r, c = divmod(idx, SIDE)
        for d in (1, 4):
            for dr, dc in ((-d, 0), (d, 0), (0, -d), (0, d)):
                j = ((r + dr) % SIDE) * SIDE + ((c + dc) % SIDE)
                key = (idx, j)
                if key not in seen:
                    seen.add(key)
                    g.add_edge(idx, N_UP + j, 8)
    return g


async def wr(dut, we, fields):
    await drive_edge(dut)
    getattr(dut, we).value = 1
    for k, v in fields.items():
        getattr(dut, k).value = int(v)
    await sample_edge(dut)
    await drive_edge(dut)
    getattr(dut, we).value = 0
    await sample_edge(dut)


async def load_couplings(dut, g, Jq_raw, hq_raw):
    for i in range(N):
        for k, (j, _) in enumerate(g.adj[i]):
            word = (1 << 23) | (j << 10) | (int(Jq_raw[i, j]) & 0x3FF)
            await wr(dut, "row_we", {"row_addr": i, "row_slot": k, "row_data": word})
        await wr(dut, "row_we", {"row_addr": i, "row_slot": 8,
                                 "row_data": int(hq_raw[i]) & 0x3FF})


async def load_order(dut, order, bounds):
    for a, s in enumerate(order):
        await wr(dut, "ord_we", {"ord_addr": a, "ord_data": s})
    for c, (lo, hi) in enumerate(bounds):
        await wr(dut, "cb_we", {"cb_idx": c, "cb_start": lo, "cb_end": hi})
    await drive_edge(dut)
    dut.n_colors.value = len(bounds)
    await sample_edge(dut)


async def run_phase(dut, seed, state0_bits, sweeps):
    for i, st in enumerate(stream_states(seed, P)):
        for sel in range(4):
            await wr(dut, "seed_we", {"seed_stream": i, "seed_sel": sel,
                                      "seed_word": st[sel]})
    for w in range(9):
        await wr(dut, "stw_we", {"stw_addr": w,
                                 "stw_data": (state0_bits >> (32 * w)) & 0xFFFFFFFF})
    await wr(dut, "sch_we", {"sch_idx": 0, "sch_beta": BETA, "sch_sweeps": sweeps})
    await drive_edge(dut)
    dut.n_sched.value = 1
    await sample_edge(dut)
    await drive_edge(dut)
    dut.start.value = 1
    await sample_edge(dut)
    await drive_edge(dut)
    dut.start.value = 0
    await sample_edge(dut)
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
    return int(dut.state_flat.value) & ((1 << N) - 1)


def bits_to_arr(bits):
    return np.array([(bits >> i) & 1 for i in range(N)], dtype=np.float64)


def arr_to_bits(a):
    v = 0
    for i in range(N):
        if a[i]:
            v |= 1 << i
    return v


def golden_phase(g, Jq_raw, hq_raw, seed, state0_arr, clamp, sweeps):
    gq = Graph(N)
    for i in range(N):
        for j, _ in g.adj[i]:
            if j > i:
                gq.add_edge(i, j, int(Jq_raw[i, j]))
    smp = BitTrueGridSampler(gq, hq_raw.astype(np.int32), BETA, seed, lanes=P,
                             state0=state0_arr.astype(np.int8), clamp=clamp)
    for _ in range(sweeps):
        smp.sweep()
    return smp.s.astype(np.float64)


@cocotb.test()
async def pcd_equiv(dut):
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

    g = build_graph()
    vis = np.arange(N_VIS)
    rng = np.random.default_rng(0xA11CE)
    patterns = rng.integers(0, 2, (3, N_VIS)).astype(np.float64)

    clamp_w = np.zeros(N, dtype=bool)
    clamp_w[vis] = True
    _, order_w, bounds_w = build_schedule(g, clamp_w)
    _, order_d, bounds_d = build_schedule(g, None)

    tr = PcdTrainer(g, visible_idx=vis, n_chains=C, eta=0.05, clip=0.5,
                    beta_raw=BETA, seed=1)
    chains = [np.zeros(N) for _ in range(C)]

    for step in range(T_STEPS):
        Jq_raw = quantize_s163(tr.J_master)
        hq_raw = quantize_s163(tr.h_master)
        await load_couplings(dut, g, Jq_raw, hq_raw)

        # ---- wake ----
        await load_order(dut, order_w, bounds_w)
        wake_states = []
        for b in range(B):
            data = patterns[(step + b) % 3]
            s0 = np.zeros(N)
            s0[vis] = data
            seed = (0x5A << 24) | (step << 8) | b
            got = await run_phase(dut, seed, arr_to_bits(s0), K_WAKE)
            exp = golden_phase(g, Jq_raw, hq_raw, seed, s0, clamp_w, K_WAKE)
            check(got == arr_to_bits(exp),
                  f"wake state diverged step {step} item {b}", step, seed, TB,
                  "pcd_equiv")
            wake_states.append(exp)

        # ---- dream (persistent chains) ----
        await load_order(dut, order_d, bounds_d)
        for c in range(C):
            seed = (0xD0 << 24) | (step << 8) | c
            got = await run_phase(dut, seed, arr_to_bits(chains[c]), M_DREAM)
            exp = golden_phase(g, Jq_raw, hq_raw, seed, chains[c], None, M_DREAM)
            check(got == arr_to_bits(exp),
                  f"dream state diverged step {step} chain {c}", step, seed, TB,
                  "pcd_equiv")
            chains[c] = exp

        ws = np.stack(wake_states)
        ds = np.stack(chains)
        w1, w2 = ws.mean(axis=0), np.array(
            [(ws[:, i] * ws[:, j]).mean() for i, j in tr.edges])
        d1, d2 = ds.mean(axis=0), np.array(
            [(ds[:, i] * ds[:, j]).mean() for i, j in tr.edges])
        r = tr.apply_update(w1, w2, d1, d2)
        if step % 5 == 0:
            print(f"[pcd_equiv] step {step}: grad {r['grad_norm']:.3f} "
                  f"(all states exact)", flush=True)

    print(f"[pcd_equiv] {T_STEPS} steps x {B + C} phase runs: RTL == bit-true "
          f"golden EXACTLY; weight trajectory shared by construction")
