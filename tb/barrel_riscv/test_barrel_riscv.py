"""M15 SIMT-DoD component 1: the riscv-tests rv32ui suite on ALL 8 WARPS
CONCURRENTLY, each warp diffed against its own ISS instance (P1 pattern,
W-way). All warps run the same deterministic ELF, so the shared memory
image remains consistent with every warp's serial view; a warp finishes
when it stores to tohost (1 = pass). Suite selection via MK_ELF env, one
ELF per cocotb invocation (the gate iterates).
"""
import os

import cocotb

from golden import elf as elfmod
from golden.iss import Halt, Iss
from mkutil import check, drive_edge, sample_edge, start_clock

TB = "barrel_riscv"
W = 8
MEM_BASE, MEM_SIZE = 0x8000_0000, 0x80_0000


@cocotb.test()
async def riscv_elf(dut):
    path = os.environ["MK_ELF"]
    name = os.path.basename(path)
    entry, segs, syms = elfmod.load(path)
    tohost = syms.get("tohost")
    assert entry == MEM_BASE and tohost is not None, (hex(entry), tohost)

    mem = bytearray(MEM_SIZE)
    for paddr, blob in segs:
        off = paddr - MEM_BASE
        mem[off:off + len(blob)] = blob

    class WarpIss(Iss):
        def __init__(self, hart):
            super().__init__(MEM_BASE, MEM_SIZE, entry, tohost)
            self._hart = hart

        def _csr_read(self, a):
            return self._hart if a == 0xF14 else super()._csr_read(a)

    iss = []
    for w in range(W):
        i = WarpIss(w)
        for paddr, blob in segs:
            i.load_blob(blob, paddr)
        iss.append(i)
    halted = [None] * W          # None = running, else tohost value
    commits = [0] * W            # riscv-tests p-variants PARK harts != 0 in
    PARK_MIN = 200               # a spin loop; they must stay lockstep-exact

    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    dut.imem_rdata.value = 0
    dut.dmem_rdata.value = 0
    i_req = bool(int(dut.imem_req.value))
    i_addr = (int(dut.imem_addr.value) - MEM_BASE) % MEM_SIZE
    d_req = d_we = False
    d_be = d_addr = d_wdata = 0
    await sample_edge(dut)

    for cyc in range(400_000):
        await drive_edge(dut)
        if i_req:
            dut.imem_rdata.value = int.from_bytes(
                mem[i_addr:i_addr + 4], "little")
        if d_req:
            if d_we:
                for byte in range(4):
                    if (d_be >> byte) & 1:
                        mem[d_addr + byte] = (d_wdata >> (8 * byte)) & 0xFF
                dut.dmem_rdata.value = 0
            else:
                dut.dmem_rdata.value = int.from_bytes(
                    mem[d_addr:d_addr + 4], "little")
        i_req = bool(int(dut.imem_req.value))
        i_addr = (int(dut.imem_addr.value) - MEM_BASE) % MEM_SIZE
        d_req = bool(int(dut.dmem_req.value))
        d_we = bool(int(dut.dmem_we.value))
        d_be = int(dut.dmem_be.value)
        d_addr = (int(dut.dmem_addr.value) - MEM_BASE) % MEM_SIZE
        d_wdata = int(dut.dmem_wdata.value)
        await sample_edge(dut)

        if int(dut.cmt_valid.value):
            w = int(dut.cmt_warp.value)
            commits[w] += 1
            if halted[w] is not None:
                continue                      # spinning past tohost
            try:
                exp = None
                while exp is None:
                    exp = iss[w].step()
            except Halt as h:
                halted[w] = h.value
                check(h.value == 1,
                      f"{name}: warp {w} FAILED tohost={h.value}",
                      cyc, 0, TB, "riscv")
                continue
            ctx = f"{name} warp {w}"
            check(int(dut.cmt_pc.value) == exp["pc"],
                  f"pc got {int(dut.cmt_pc.value):08x} want {exp['pc']:08x}",
                  cyc, 0, TB, "riscv", ctx)
            check(int(dut.cmt_instr.value) == exp["instr"],
                  f"instr got {int(dut.cmt_instr.value):08x} want "
                  f"{exp['instr']:08x}", cyc, 0, TB, "riscv", ctx)
            if exp["rd"]:
                check(int(dut.cmt_wdata.value) == exp["wdata"],
                      f"wdata got {int(dut.cmt_wdata.value):08x} want "
                      f"{exp['wdata']:08x}", cyc, 0, TB, "riscv", ctx)
        if halted[0] is not None and min(commits[1:]) >= PARK_MIN:
            break

    check(halted[0] == 1, f"{name}: warp 0 result {halted[0]}",
          0, 0, TB, "riscv")
    for w in range(1, W):
        check(commits[w] >= PARK_MIN,
              f"{name}: warp {w} only {commits[w]} commits (park not "
              f"proven)", 0, 0, TB, "riscv")
    print(f"[{TB}] {name}: warp0 PASS, warps 1-7 parked lockstep-exact "
          f"({sum(commits)} total commits)")
