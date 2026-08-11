"""Shared testbench plumbing — the Σ.3 universal pattern's common pieces.

Every bench uses: seeded Rng, start_clock, reset_n, and check() which raises
with full context (vector index, seed, one-line repro command) on mismatch.
Seed policy (ci/seeds.yaml): smoke = 0xC0FFEE fixed; fuzz = MK_SEED from the
environment (set by tb/common.mk, defaulting to epoch seconds).
"""
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge

SMOKE_SEED = 0xC0FFEE


def get_seed(default: int = SMOKE_SEED) -> int:
    s = os.environ.get("MK_SEED")
    return int(s, 0) if s else default


def get_nvec(default: int) -> int:
    s = os.environ.get("MK_NVEC")
    return int(s, 0) if s else default


class Rng(random.Random):
    def chance(self, p: float) -> bool:
        return self.random() < p

    def bits(self, n: int) -> int:
        return self.getrandbits(n)


def start_clock(dut, period_ns: int = 10):
    """cocotb 2.x: Clock.start() schedules the clock itself."""
    Clock(dut.clk, period_ns, unit="ns").start()


async def reset_n(dut, zero_signals=(), cycles: int = 3):
    dut.rst_n.value = 0
    for name in zero_signals:
        getattr(dut, name).value = 0
    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_edge(dut):
    """Await the safe drive point: the falling edge (half a cycle of setup)."""
    await FallingEdge(dut.clk)


async def sample_edge(dut):
    """Await the safe sample point: rising edge, then ReadOnly (NBA settled).
    The per-vector protocol every bench uses:
        await drive_edge(dut); <set inputs>
        await sample_edge(dut); <compare outputs vs golden.step(inputs)>
    """
    await RisingEdge(dut.clk)
    await ReadOnly()


def repro_line(tb_dir: str, target: str, seed: int) -> str:
    return f"MK_SEED={seed:#x} make -C tb/{tb_dir} {target}"


def check(cond: bool, what: str, i: int, seed: int, tb_dir: str, target: str, extra: str = ""):
    if not cond:
        raise AssertionError(
            f"MISMATCH {what} at vector {i} (seed {seed:#x}) {extra}\n"
            f"  repro: {repro_line(tb_dir, target, seed)}\n"
            f"  waves: fuzz reruns with VERILATOR_TRACE=1 automatically -> tb/{tb_dir}/dump.vcd"
        )
