"""SG0 bridge bench: rtl/sg0_bridge.sv vs a python mirror.

Three quarters, three referees:
  - AXI-Lite control: register readback (STATUS magic, MCYCLE run
    gating, IMEM_ADDR autoincrement) driven with fully handshaked
    single-beat transactions.
  - imem BRAM: words loaded through the AXI-Lite window must read
    back on the core face with the 1-cycle contract.
  - dmem credit face -> AXI4 master: request pulses (bursts up to
    the 8-credit bound) against a latency-randomizing AXI slave
    memory model; acks must return exactly one per beat IN ORDER,
    reads must match a python mirror (byte strobes applied), and
    every AXI address must carry CARVE_BASE.
"""
import os
import random
import sys
from collections import deque

import cocotb
from cocotb.triggers import RisingEdge

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, MK)

from mkutil import get_nvec, get_seed, reset_n, start_clock  # noqa: E402

CARVE = 0x4000_0000


async def axil_write(dut, addr, data):
    dut.s_axil_awvalid.value = 1
    dut.s_axil_awaddr.value = addr
    dut.s_axil_wvalid.value = 1
    dut.s_axil_wdata.value = data
    dut.s_axil_wstrb.value = 0xF
    aw = w = False
    for _ in range(20):
        await RisingEdge(dut.clk)
        if not aw and dut.s_axil_awready.value:
            aw = True
            dut.s_axil_awvalid.value = 0
        if not w and dut.s_axil_wready.value:
            w = True
            dut.s_axil_wvalid.value = 0
        if aw and w:
            break
    assert aw and w, "AXIL write address/data never accepted"
    dut.s_axil_bready.value = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.s_axil_bvalid.value:
            break
    else:
        raise AssertionError("AXIL write response timeout")
    dut.s_axil_bready.value = 0


async def axil_read(dut, addr):
    dut.s_axil_arvalid.value = 1
    dut.s_axil_araddr.value = addr
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.s_axil_arready.value:
            break
    else:
        raise AssertionError("AXIL AR never accepted")
    dut.s_axil_arvalid.value = 0
    dut.s_axil_rready.value = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.s_axil_rvalid.value:
            data = int(dut.s_axil_rdata.value)
            dut.s_axil_rready.value = 0
            return data
    raise AssertionError("AXIL read data timeout")


async def axi_mem_model(dut, mem, rng):
    """One-outstanding AXI4 slave with randomized service latency."""
    dut.m_axi_awready.value = 0
    dut.m_axi_wready.value = 0
    dut.m_axi_arready.value = 0
    dut.m_axi_bvalid.value = 0
    dut.m_axi_rvalid.value = 0
    while True:
        await RisingEdge(dut.clk)
        if dut.m_axi_bvalid.value and dut.m_axi_bready.value:
            dut.m_axi_bvalid.value = 0
        if dut.m_axi_rvalid.value and dut.m_axi_rready.value:
            dut.m_axi_rvalid.value = 0
        if dut.m_axi_arvalid.value and not dut.m_axi_rvalid.value:
            addr = int(dut.m_axi_araddr.value)
            assert addr >= CARVE, f"AR below CARVE_BASE: {addr:#x}"
            dut.m_axi_arready.value = 1
            await RisingEdge(dut.clk)
            dut.m_axi_arready.value = 0
            for _ in range(rng.randrange(0, 6)):
                await RisingEdge(dut.clk)
            dut.m_axi_rdata.value = mem.get(addr - CARVE, 0)
            dut.m_axi_rlast.value = 1
            dut.m_axi_rresp.value = 0
            dut.m_axi_rvalid.value = 1
        elif dut.m_axi_awvalid.value and not dut.m_axi_bvalid.value:
            addr = int(dut.m_axi_awaddr.value)
            assert addr >= CARVE, f"AW below CARVE_BASE: {addr:#x}"
            dut.m_axi_awready.value = 1
            await RisingEdge(dut.clk)
            dut.m_axi_awready.value = 0
            for _ in range(20):
                if dut.m_axi_wvalid.value:
                    break
                await RisingEdge(dut.clk)
            wd = int(dut.m_axi_wdata.value)
            strb = int(dut.m_axi_wstrb.value)
            dut.m_axi_wready.value = 1
            await RisingEdge(dut.clk)
            dut.m_axi_wready.value = 0
            old = mem.get(addr - CARVE, 0)
            new = 0
            for b in range(4):
                sel = wd if (strb >> b) & 1 else old
                new |= sel & (0xFF << (8 * b))
            mem[addr - CARVE] = new
            for _ in range(rng.randrange(0, 6)):
                await RisingEdge(dut.clk)
            dut.m_axi_bresp.value = 0
            dut.m_axi_bvalid.value = 1


async def dmem_pulse(dut, we, addr, wdata=0, be=0xF):
    dut.dmem_req.value = 1
    dut.dmem_we.value = 1 if we else 0
    dut.dmem_addr.value = addr
    dut.dmem_wdata.value = wdata
    dut.dmem_be.value = be
    await RisingEdge(dut.clk)
    dut.dmem_req.value = 0


@cocotb.test()
async def smoke_bridge(dut):
    seed = get_seed()
    nvec = get_nvec(300)
    rng = random.Random(seed)
    start_clock(dut)
    await reset_n(dut, zero_signals=(
        "s_axil_awvalid", "s_axil_wvalid", "s_axil_bready",
        "s_axil_arvalid", "s_axil_rready", "imem_req", "imem_addr",
        "dmem_req", "dmem_we", "dmem_be", "dmem_addr", "dmem_wdata",
        "m_axi_awready", "m_axi_wready", "m_axi_arready",
        "m_axi_bvalid", "m_axi_rvalid", "m_axi_rdata", "m_axi_rlast",
        "m_axi_bresp", "m_axi_rresp"))
    mem = {}
    cocotb.start_soon(axi_mem_model(dut, mem, random.Random(seed ^ 1)))

    # ---- STATUS / CTRL / MCYCLE ----
    st = await axil_read(dut, 0x04)
    assert (st >> 16) == 0x05D0, f"STATUS magic: {st:#x}"
    assert int(dut.soc_rst_n.value) == 0, "SoC out of reset before run"
    lo0 = await axil_read(dut, 0x08)
    lo1 = await axil_read(dut, 0x08)
    assert lo0 == lo1 == 0, "MCYCLE moving while halted"
    await axil_write(dut, 0x00, 1)
    assert int(dut.soc_rst_n.value) == 1, "run did not release reset"
    lo2 = await axil_read(dut, 0x08)
    lo3 = await axil_read(dut, 0x08)
    assert lo3 > lo2 >= 1, "MCYCLE not counting under run"

    # ---- imem load window + 1-cycle read contract ----
    words = [rng.getrandbits(32) for _ in range(32)]
    await axil_write(dut, 0x10, 7)             # IMEM_ADDR = 7
    for wv in words:
        await axil_write(dut, 0x14, wv)        # autoincrement
    ia = await axil_read(dut, 0x10)
    assert ia == 7 + 32, f"IMEM_ADDR autoincrement: {ia}"
    for i, wv in enumerate(words):
        dut.imem_addr.value = (7 + i) * 4
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        got = int(dut.imem_rdata.value)
        assert got == wv, f"imem[{7 + i}]: {got:#x} != {wv:#x}"

    # ---- dmem: an 8-deep write burst, then ordered reads ----
    mirror = {}
    addrs = [rng.randrange(0, 1 << 20) * 4 for _ in range(8)]
    datas = [rng.getrandbits(32) for _ in range(8)]
    acks = 0
    for a, d in zip(addrs, datas):
        await dmem_pulse(dut, True, a, d)
        mirror[a] = d
        acks += int(dut.dmem_ack.value)
    for _ in range(600):
        await RisingEdge(dut.clk)
        acks += int(dut.dmem_ack.value)
        if acks == 8:
            break
    assert acks == 8, f"write acks: {acks}/8"
    for a, d in zip(addrs, datas):
        assert mem.get(a, None) == d, f"mem[{a:#x}] missing/wrong"

    got = []
    exp = []
    for a in addrs:
        await dmem_pulse(dut, False, a)
        exp.append(mirror[a])
        if int(dut.dmem_ack.value):
            got.append(int(dut.dmem_rdata.value))
    for _ in range(800):
        await RisingEdge(dut.clk)
        if int(dut.dmem_ack.value):
            got.append(int(dut.dmem_rdata.value))
        if len(got) == 8:
            break
    assert got == exp, f"read order/data: {got} != {exp}"

    # ---- byte-strobe write ----
    a0 = addrs[0]
    await dmem_pulse(dut, True, a0, 0xAABBCCDD, be=0x5)
    for _ in range(200):
        await RisingEdge(dut.clk)
        if int(dut.dmem_ack.value):
            break
    keep = mirror[a0]
    mirror[a0] = (keep & 0xFF00FF00) | (0xAABBCCDD & 0x00FF00FF)
    assert mem[a0] == mirror[a0], \
        f"strobe write: {mem[a0]:#x} != {mirror[a0]:#x}"

    # ---- randomized soak: mixed ops vs the mirror ----
    outstanding = deque()
    acks = 0
    issued = 0
    for i in range(nvec):
        if outstanding and (len(outstanding) == 8 or rng.random() < 0.5):
            await RisingEdge(dut.clk)
        else:
            a = rng.randrange(0, 1 << 16) * 4
            if rng.random() < 0.5:
                d = rng.getrandbits(32)
                await dmem_pulse(dut, True, a, d)
                mirror[a] = d
                outstanding.append(("w", a, None))
            else:
                await dmem_pulse(dut, False, a)
                outstanding.append(("r", a, mirror.get(a, 0)))
            issued += 1
        if int(dut.dmem_ack.value):
            kind, a, expv = outstanding.popleft()
            acks += 1
            if kind == "r":
                gv = int(dut.dmem_rdata.value)
                assert gv == expv, \
                    f"soak read mem[{a:#x}]: {gv:#x} != {expv:#x} " \
                    f"(seed {seed})"
    for _ in range(2000):
        await RisingEdge(dut.clk)
        if int(dut.dmem_ack.value):
            kind, a, expv = outstanding.popleft()
            acks += 1
            if kind == "r":
                gv = int(dut.dmem_rdata.value)
                assert gv == expv, \
                    f"drain read mem[{a:#x}]: {gv:#x} != {expv:#x} " \
                    f"(seed {seed})"
        if not outstanding:
            break
    assert not outstanding, f"unacked beats: {len(outstanding)}"
    assert acks == issued, f"ack count {acks} != issued {issued}"
    dut._log.info(f"sg0_bridge smoke: {issued} soak beats, seed {seed}")
