"""FIFO bench — Σ.3 pattern: golden drives stimulus + expectations, per-cycle
compare of every output, mismatch raises with seed + repro line.

Per-edge protocol (Moore outputs): set inputs -> await edge -> compare
post-edge outputs against golden.step(inputs) -> loop.
"""
import cocotb

from golden.fifo import FifoModel
from mkutil import (Rng, check, drive_edge, get_nvec, get_seed, reset_n,
                    sample_edge, start_clock)

TB = "fifo"
DEPTH = 16


async def run(dut, seed: int, n: int, target: str):
    rng = Rng(seed)
    model = FifoModel(DEPTH)
    start_clock(dut)
    await reset_n(dut, ("wr_en", "rd_en", "wr_data"))

    for i in range(n):
        # biased phases so full and empty both get hammered
        phase = (i // 500) % 3
        p_wr = (0.7, 0.3, 0.5)[phase]
        p_rd = (0.3, 0.7, 0.5)[phase]
        wr = rng.chance(p_wr)
        rd = rng.chance(p_rd)
        data = rng.bits(8)

        await drive_edge(dut)
        dut.wr_en.value = int(wr)
        dut.rd_en.value = int(rd)
        dut.wr_data.value = data
        await sample_edge(dut)
        exp = model.step(wr, data, rd)

        ctx = f"(wr={int(wr)} rd={int(rd)} data={data:#x})"
        check(int(dut.count.value) == exp["count"],
              f"count got {int(dut.count.value)} want {exp['count']}", i, seed, TB, target, ctx)
        check(int(dut.full.value) == exp["full"],
              f"full got {int(dut.full.value)} want {exp['full']}", i, seed, TB, target, ctx)
        check(int(dut.empty.value) == exp["empty"],
              f"empty got {int(dut.empty.value)} want {exp['empty']}", i, seed, TB, target, ctx)
        if exp["rd_data"] is not None:
            check(int(dut.rd_data.value) == exp["rd_data"],
                  f"rd_data got {int(dut.rd_data.value)} want {exp['rd_data']:#x}",
                  i, seed, TB, target, ctx)


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(300), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(10000), "fuzz")
