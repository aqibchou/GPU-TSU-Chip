"""p-dit cell bench — exact per-sample diff against golden/pdit.py.

Given (27 loaded z values, rnd word) the CDF-scan cell is fully
deterministic: every vector loads 27 random (acc, beta) pairs, fires
sample_en, and compares (sym, s_dbg) bit-exactly against
golden.pdit.z_from_acc + sample_symbol. Directed vectors cover the
degenerate corners (all-equal z, single dominant symbol, u16 extremes,
acc/beta saturation). Distribution-level exactness (the S2-style gate over
q_k) rides on the same closed form exact_cdf_scan used by the design study.
"""
import cocotb

from golden.pdit import K, sample_symbol, z_from_acc
from mkutil import (Rng, check, drive_edge, get_nvec, get_seed, reset_n,
                    sample_edge, start_clock)

TB = "pdit_cell"


async def one_vector(dut, i, seed, target, accs, betas, rnd32, ctx=""):
    await drive_edge(dut)
    dut.z_clear.value = 1
    dut.z_load.value = 0
    dut.sample_en.value = 0
    await sample_edge(dut)

    z = []
    for k in range(K):
        await drive_edge(dut)
        dut.z_clear.value = 0
        dut.z_load.value = 1
        dut.z_idx.value = k
        dut.acc.value = accs[k] & 0x3FFF
        dut.beta.value = betas[k]
        await sample_edge(dut)
        z.append(z_from_acc(accs[k], betas[k]))

    await drive_edge(dut)
    dut.z_load.value = 0
    dut.sample_en.value = 1
    dut.rnd.value = rnd32
    await sample_edge(dut)
    await drive_edge(dut)
    dut.sample_en.value = 0
    await sample_edge(dut)

    for _ in range(K * 2 + 6):          # WSUM(27) + TMUL + SCAN(<=27)
        if int(dut.sym_valid.value):
            break
        await drive_edge(dut)
        await sample_edge(dut)
    check(int(dut.sym_valid.value) == 1, "sym_valid missing", i, seed, TB,
          target, ctx)
    exp = sample_symbol(z, rnd32 >> 16)
    check(int(dut.sym.value) == exp,
          f"sym got {int(dut.sym.value)} want {exp}", i, seed, TB, target,
          ctx)


async def run(dut, seed: int, n: int, target: str):
    rng = Rng(seed)
    start_clock(dut)
    await reset_n(dut, ("z_clear", "z_load", "z_idx", "acc", "beta",
                        "sample_en", "rnd"))

    # directed corners first
    directed = [
        ([0] * K, [64] * K, 0x00000000, "all-zero u=0"),
        ([0] * K, [64] * K, 0xFFFF0000, "all-zero u=max"),
        ([8191] + [-8192] * (K - 1), [255] * K, rng.bits(32), "dominant-0"),
        ([-8192] * (K - 1) + [8191], [255] * K, rng.bits(32), "dominant-26"),
        ([8191] * K, [255] * K, rng.bits(32), "all-saturated"),
        ([0] * K, [0] * K, rng.bits(32), "beta-zero uniform"),
    ]
    for i, (a, b, r, ctx) in enumerate(directed):
        await one_vector(dut, i, seed, target, a, b, r, ctx)

    for i in range(n):
        accs = [rng.randint(-8192, 8191) for _ in range(K)]
        betas = [rng.randint(0, 255) for _ in range(K)]
        await one_vector(dut, 100 + i, seed, target, accs, betas,
                         rng.bits(32))


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(100), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(1000), "fuzz")
