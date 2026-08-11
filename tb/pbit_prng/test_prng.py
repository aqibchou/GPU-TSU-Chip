"""PRNG farm bench — Σ.3 per-cycle diff: seed all streams via the seed port
from golden-computed jump-ahead states, then step and compare every stream's
word against golden every cycle. Also checks: seeding one stream doesn't
disturb another, and outputs hold steady when step is low.
"""
import cocotb

from golden.xoshiro import Xoshiro128pp, stream_states
from mkutil import (Rng, check, drive_edge, get_nvec, get_seed, reset_n,
                    sample_edge, start_clock)

TB = "pbit_prng"
NSTREAMS = 16


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


async def run(dut, seed: int, n: int, target: str):
    rng = Rng(seed)
    start_clock(dut)
    await reset_n(dut, ("seed_we", "seed_stream", "seed_sel", "seed_word", "step"))

    states = stream_states(seed, NSTREAMS)
    await load_seeds(dut, states)
    golden = [Xoshiro128pp(list(s)) for s in states]
    expected = [g.next() for g in golden]      # output of current state

    for i in range(n):
        do_step = rng.chance(0.8)
        await drive_edge(dut)
        dut.step.value = int(do_step)
        await sample_edge(dut)
        # rnd is a packed [NSTREAMS-1:0][31:0] port — cocotb 2.x can't index
        # packed arrays, so slice the flat value (stream k = bits 32k+:32)
        flat = int(dut.rnd.value)
        got = [(flat >> (32 * k)) & 0xFFFFFFFF for k in range(NSTREAMS)]
        if do_step:
            expected = [g.next() for g in golden]
        for k in range(NSTREAMS):
            check(got[k] == expected[k],
                  f"stream {k} got {got[k]:#010x} want {expected[k]:#010x}",
                  i, seed, TB, target, f"(step={int(do_step)})")


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(200), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(4096), "fuzz")
