"""p-bit cell bench — exact per-update diff against golden/pbit.py.

Given the same rnd word the cell is fully deterministic, so equivalence needs
no statistics: every vector drives a random (mode, bias, neighbors, beta, rnd)
update through the serial MAC + 2-stage sample and compares (s_out, p_dbg)
bit-exactly. The statistical Bernoulli-rate gate (real farm randomness) lives
in gates/m4_pbit_cell.py on the pbit_mc_top wrapper.
"""
import cocotb

from golden.pbit import decision, mac
from mkutil import (Rng, check, drive_edge, get_nvec, get_seed, reset_n,
                    sample_edge, start_clock)

TB = "pbit_cell"


async def run(dut, seed: int, n: int, target: str):
    rng = Rng(seed)
    start_clock(dut)
    await reset_n(dut, ("bipolar", "acc_clear", "bias", "acc_en", "j_val",
                        "s_in", "sample_en", "beta", "rnd"))

    for i in range(n):
        bipolar = rng.chance(0.5)
        bias_raw = rng.randint(-511, 511)
        deg = rng.randint(0, 8)
        neigh = [(rng.randint(-511, 511), rng.randint(0, 1)) for _ in range(deg)]
        beta_raw = rng.randint(0, 255)
        rnd32 = rng.bits(32)

        await drive_edge(dut)
        dut.bipolar.value = int(bipolar)
        dut.acc_clear.value = 1
        dut.acc_en.value = 0
        dut.sample_en.value = 0
        dut.bias.value = bias_raw & 0x3FF
        await sample_edge(dut)

        for j_raw, s in neigh:
            await drive_edge(dut)
            dut.acc_clear.value = 0
            dut.acc_en.value = 1
            dut.j_val.value = j_raw & 0x3FF
            dut.s_in.value = s
            await sample_edge(dut)

        await drive_edge(dut)
        dut.acc_clear.value = 0
        dut.acc_en.value = 0
        dut.sample_en.value = 1
        dut.beta.value = beta_raw
        dut.rnd.value = rnd32
        await sample_edge(dut)          # stage 1
        check(int(dut.s_valid.value) == 0, "s_valid early", i, seed, TB, target)

        await drive_edge(dut)
        dut.sample_en.value = 0
        await sample_edge(dut)          # stage 2 -> registered outputs
        exp_s, exp_p = decision(mac(bias_raw, neigh, bipolar), beta_raw, rnd32)
        ctx = f"(bip={int(bipolar)} bias={bias_raw} deg={deg} beta={beta_raw})"
        check(int(dut.s_valid.value) == 1, "s_valid missing", i, seed, TB, target, ctx)
        check(int(dut.p_dbg.value) == exp_p,
              f"p17 got {int(dut.p_dbg.value)} want {exp_p}", i, seed, TB, target, ctx)
        check(int(dut.s_out.value) == exp_s,
              f"s got {int(dut.s_out.value)} want {exp_s}", i, seed, TB, target, ctx)


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(150), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(2000), "fuzz")
