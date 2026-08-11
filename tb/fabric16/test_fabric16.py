"""fabric16 bench — trajectory equivalence: with identical seeds the RTL chain
must equal golden/gibbs.py's BitTrueSampler state-for-state at every sweep
(covers scheduler order, MAC feeds, clamp behavior, farm stepping — any
divergence anywhere shows within a few sweeps).
"""
import cocotb
from cocotb.triggers import RisingEdge

from golden.gibbs import BitTrueSampler, make_instance, neighbors
from mkutil import check, drive_edge, get_nvec, get_seed, sample_edge, start_clock
from golden.xoshiro import stream_states

TB = "fabric16"


async def wr_cfg(dut, site, slot, val):
    await drive_edge(dut)
    dut.cfg_we.value = 1
    dut.cfg_site.value = site
    dut.cfg_slot.value = slot
    dut.cfg_data.value = val & 0x3FF
    await sample_edge(dut)


async def load_instance(dut, J, h):
    for i in range(16):
        slot_vals = {k: 0 for k in range(8)}
        for k, j in neighbors(i):
            slot_vals[k] = int(J[i, j])
        for k in range(8):
            await wr_cfg(dut, i, k, slot_vals[k])
        await wr_cfg(dut, i, 8, int(h[i]))
    await drive_edge(dut)
    dut.cfg_we.value = 0
    await sample_edge(dut)


async def load_seeds(dut, states):
    for i, st in enumerate(states):
        for sel in range(4):
            await drive_edge(dut)
            dut.seed_we.value = 1
            dut.seed_stream.value = i
            dut.seed_sel.value = sel
            dut.seed_word.value = st[sel]
            await sample_edge(dut)
    await drive_edge(dut)
    dut.seed_we.value = 0
    await sample_edge(dut)


async def run_case(dut, seed, beta, clamp, s0, sweeps, target):
    J, h = make_instance(seed)
    await load_instance(dut, J, h)
    await load_seeds(dut, stream_states(seed, 16))
    await drive_edge(dut)
    dut.beta.value = beta
    dut.clamp.value = clamp
    dut.s_init.value = s0
    dut.s_load.value = 1
    await sample_edge(dut)
    await drive_edge(dut)
    dut.s_load.value = 0
    await sample_edge(dut)

    gold = BitTrueSampler(J, h, beta, seed, state0=s0, clamp=clamp)
    for t in range(sweeps):
        await drive_edge(dut)
        dut.start.value = 1
        dut.sweeps.value = 1
        await sample_edge(dut)
        await drive_edge(dut)
        dut.start.value = 0
        await sample_edge(dut)
        while int(dut.done.value) == 0:
            await RisingEdge(dut.clk)
        exp = gold.sweep()
        got = int(dut.s_out.value)
        check(got == exp,
              f"state got {got:#06x} want {exp:#06x} at sweep {t}",
              t, seed, TB, target, f"(beta={beta} clamp={clamp:#x})")


async def run(dut, seed, n, target):
    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    for name in ("cfg_we", "cfg_site", "cfg_slot", "cfg_data", "clamp",
                 "s_load", "s_init", "beta", "bipolar", "seed_we",
                 "seed_stream", "seed_sel", "seed_word", "start", "sweeps"):
        getattr(dut, name).value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    await sample_edge(dut)

    t = max(n // 3, 20)
    await run_case(dut, seed, 32, 0x0000, 0x0000, t, target)
    await run_case(dut, seed + 1, 64, 0x0000, 0xFFFF, t, target)
    await run_case(dut, seed + 2, 16, 0x0F0F, 0xA5A5, t, target)  # clamped


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(60), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(400), "fuzz")
