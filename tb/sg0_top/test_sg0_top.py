"""SG0 end-to-end integration smoke: the full silicon loop in sim.

A hand-assembled RV32I program is loaded through the AXI-Lite imem
window, CTRL.run releases the SoC, all 64 harts fetch from the imem
BRAM, execute, and store through the memory unit -> dmem credit face
-> bridge FIFO -> AXI4 master -> the bench's AXI memory model. The
referee: the stores land at CARVE_BASE-relative addresses with the
right values. This exercises every contract the bitstream relies on
(reset gating, imem load + 1-cycle fetch, decode/execute across
warps, credit-face pulses, AXI transactions) with zero mocks inside
the chip.
"""
import os
import random
import sys

import cocotb
from cocotb.triggers import RisingEdge

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, MK)
sys.path.insert(0, os.path.join(MK, "tb", "sg0_bridge"))

from mkutil import get_seed, reset_n, start_clock          # noqa: E402
from test_sg0_bridge import axil_read, axil_write, axi_mem_model  # noqa: E402

# all harts run this; stores are idempotent across warps/lanes
PROG = [
    0x123450B7,  # lui  x1, 0x12345      -> x1 = 0x12345000
    0x00010137,  # lui  x2, 0x00010      -> x2 = 0x00010000
    0x00112023,  # sw   x1, 0(x2)
    0x00112223,  # sw   x1, 4(x2)
    0x0000006F,  # jal  x0, 0            (spin)
]


@cocotb.test()
async def smoke_sg0(dut):
    seed = get_seed()
    start_clock(dut)
    await reset_n(dut, zero_signals=(
        "s_axil_awvalid", "s_axil_wvalid", "s_axil_bready",
        "s_axil_arvalid", "s_axil_rready",
        "m_axi_awready", "m_axi_wready", "m_axi_arready",
        "m_axi_bvalid", "m_axi_rvalid", "m_axi_rdata", "m_axi_rlast",
        "m_axi_bresp", "m_axi_rresp"))
    mem = {}
    cocotb.start_soon(axi_mem_model(dut, mem, random.Random(seed ^ 1)))

    st = await axil_read(dut, 0x04)
    assert (st >> 16) == 0x05D0, f"STATUS magic: {st:#x}"

    # load the program at word 0, then release the SoC
    await axil_write(dut, 0x10, 0)
    for w in PROG:
        await axil_write(dut, 0x14, w)
    await axil_write(dut, 0x00, 1)

    # 64 harts x 2 stores = 128 beats through the one-outstanding AXI
    # FSM at model latency — give it room, poll the referee addresses
    for _ in range(60):
        for _ in range(200):
            await RisingEdge(dut.clk)
        if mem.get(0x10000) == 0x12345000 and \
           mem.get(0x10004) == 0x12345000:
            break
    assert mem.get(0x10000) == 0x12345000, \
        f"store 0 missing/wrong: {mem.get(0x10000)}"
    assert mem.get(0x10004) == 0x12345000, \
        f"store 1 missing/wrong: {mem.get(0x10004)}"

    # nothing else should have been written
    stray = {a: v for a, v in mem.items() if a not in (0x10000, 0x10004)}
    assert not stray, f"stray writes: { {hex(a): hex(v) for a, v in stray.items()} }"

    # MCYCLE sanity: counting under run
    lo0 = await axil_read(dut, 0x08)
    lo1 = await axil_read(dut, 0x08)
    assert lo1 > lo0 > 0, "MCYCLE not counting"
    dut._log.info(f"sg0 end-to-end: stores landed, mcycle={lo1}")
