"""Pipelined multiplier bench — per-cycle compare of (out_valid, p) against
the golden pipeline model, including bubbles and back-to-back operation.
"""
import cocotb

from golden.mul import MulPipeModel
from mkutil import (Rng, check, drive_edge, get_nvec, get_seed, reset_n,
                    sample_edge, start_clock)

TB = "mul_pipe"
W = 16
STAGES = 3


async def run(dut, seed: int, n: int, target: str):
    rng = Rng(seed)
    model = MulPipeModel(W, STAGES)
    start_clock(dut)
    await reset_n(dut, ("in_valid", "a", "b"))

    # corner vectors first, then random; valid-density phases exercise bubbles
    corners = [(0, 0), (0xFFFF, 0xFFFF), (1, 0xFFFF), (0xFFFF, 1), (1, 1),
               (0x8000, 0x8000), (0x8000, 2), (0x7FFF, 0x7FFF)]
    for i in range(n):
        if i < len(corners):
            v, (a, b) = True, corners[i]
        else:
            p_v = (0.9, 0.3, 1.0)[(i // 400) % 3]
            v = rng.chance(p_v)
            a, b = rng.bits(W), rng.bits(W)

        await drive_edge(dut)
        dut.in_valid.value = int(v)
        dut.a.value = a
        dut.b.value = b
        await sample_edge(dut)
        exp = model.step(v, a, b)

        ctx = f"(v={int(v)} a={a:#x} b={b:#x})"
        check(int(dut.out_valid.value) == exp["out_valid"],
              f"out_valid got {int(dut.out_valid.value)} want {exp['out_valid']}",
              i, seed, TB, target, ctx)
        if exp["p"] is not None:
            check(int(dut.p.value) == exp["p"],
                  f"p got {int(dut.p.value):#x} want {exp['p']:#x}",
                  i, seed, TB, target, ctx)


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(300), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(10000), "fuzz")
