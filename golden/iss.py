"""RV32I + Zicsr + Zifencei instruction-set simulator — the CPU golden model.

Lockstep contract (gates/p1_lockstep.py): step() retires at most one
instruction and returns a commit record; trapped instructions do NOT retire
(matching spike --log-commits, which logs no commit line for them). The
canonical log line format shared by ISS / RTL harness / spike-parser is:

    <pc>:<instr>[:xN=V][:S@ADDR=VAL.SZ]     (all hex, SZ in {0,1,2} = B,H,W)

CSR model mirrors `spike --isa=rv32i_zicsr_zifencei --priv=m`:
implemented-with-storage: mstatus mie mtvec mscratch mepc mcause mtval;
read-only: mhartid/mvendorid/marchid/mimpid/mconfigptr = 0, misa = 0x40000100;
write-ignored/read-zero: mstatush mip mcycle(h) minstret(h) mcountinhibit
pmpcfg0-3 pmpaddr0-15; anything else (satp, medeleg, mideleg, s*/u*) raises
an illegal-instruction trap — riscv-tests' prologue relies on exactly this.

Lanes note (M13): all arch state lives in the LaneState object and exec_op()
is a pure function of (op, lane, mem) — the warp ISS replicates LaneState
per lane and adds a mask stack; decode() is already lane-independent.
"""
from __future__ import annotations

M32 = 0xFFFFFFFF


def sext(v: int, bits: int) -> int:
    m = 1 << (bits - 1)
    return ((v & ((1 << bits) - 1)) ^ m) - m


def u32(v: int) -> int:
    return v & M32


def s32(v: int) -> int:
    return sext(v & M32, 32)


CSR_STORE = {0x300: "mstatus", 0x304: "mie", 0x305: "mtvec", 0x340: "mscratch",
             0x341: "mepc", 0x342: "mcause", 0x343: "mtval"}
CSR_RO = {0xF11: 0, 0xF12: 0, 0xF13: 0, 0xF14: 0, 0xF15: 0, 0x301: 0x40000100}
CSR_DUMMY = ({0x310, 0x344, 0xB00, 0xB02, 0xB80, 0xB82, 0x320}
             | set(range(0x3A0, 0x3A4)) | set(range(0x3B0, 0x3C0)))

MSTATUS_WMASK = 0x00001888  # MIE(3) MPIE(7) MPP(12:11)


class Halt(Exception):
    """Raised on the store to tohost; .value carries the stored word."""
    def __init__(self, value: int):
        self.value = value


class LaneState:
    __slots__ = ("x", "pc")

    def __init__(self, reset_pc: int):
        self.x = [0] * 32
        self.pc = reset_pc


class Iss:
    def __init__(self, mem_base=0x8000_0000, mem_size=0x80_0000,
                 reset_pc=0x8000_0000, tohost=None):
        self.base = mem_base
        self.mem = bytearray(mem_size)
        self.lane = LaneState(reset_pc)
        self.csr = {name: 0 for name in CSR_STORE.values()}
        self.tohost = tohost
        self.instret = 0

    # ---------------- memory ----------------
    def _off(self, addr: int, size: int) -> int:
        off = addr - self.base
        if off < 0 or off + size > len(self.mem):
            raise IndexError(f"access outside memory: {addr:#x}")
        return off

    def read(self, addr: int, size: int) -> int:
        off = self._off(addr, size)
        return int.from_bytes(self.mem[off:off + size], "little")

    def write(self, addr: int, size: int, val: int) -> None:
        off = self._off(addr, size)
        self.mem[off:off + size] = (val & ((1 << (8 * size)) - 1)).to_bytes(size, "little")

    def load_blob(self, blob: bytes, addr: int) -> None:
        off = self._off(addr, len(blob) or 1)
        self.mem[off:off + len(blob)] = blob

    # ---------------- traps ----------------
    def _trap(self, cause: int, tval: int, pc: int) -> None:
        self.csr["mepc"] = pc
        self.csr["mcause"] = cause
        self.csr["mtval"] = u32(tval)
        st = self.csr["mstatus"]
        mie = (st >> 3) & 1
        st = (st & ~0x1888) | (mie << 7) | (3 << 11)   # MPIE<-MIE, MIE<-0, MPP<-M
        self.csr["mstatus"] = st & MSTATUS_WMASK
        self.lane.pc = self.csr["mtvec"] & ~3

    # ---------------- CSR access ----------------
    def _csr_read(self, a: int) -> int:
        if a in CSR_STORE:
            return self.csr[CSR_STORE[a]]
        if a in CSR_RO:
            return CSR_RO[a]
        if a in CSR_DUMMY:
            return 0
        raise ValueError("illegal csr")

    def _csr_write(self, a: int, v: int) -> None:
        if a in CSR_STORE:
            n = CSR_STORE[a]
            if n == "mstatus":
                v &= MSTATUS_WMASK
            elif n == "mtvec":
                v &= ~3
            elif n == "mepc":
                v &= ~3
            self.csr[n] = u32(v)
        elif a in CSR_RO:
            raise ValueError("illegal csr write")
        elif a in CSR_DUMMY:
            pass
        else:
            raise ValueError("illegal csr")

    # ---------------- one instruction ----------------
    def step(self):
        """Retire one instruction; returns a commit dict, or None if the
        instruction trapped (no retire). Raises Halt on the tohost store."""
        lane = self.lane
        pc = lane.pc
        if pc & 3:
            self._trap(0, pc, pc)
            return None
        instr = self.read(pc, 4)
        op = instr & 0x7F
        rd = (instr >> 7) & 0x1F
        f3 = (instr >> 12) & 7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        f7 = instr >> 25
        a = lane.x[rs1]
        b = lane.x[rs2]
        wb = None            # (rd, value)
        store = None         # (addr, value, size_log2)
        next_pc = u32(pc + 4)

        try:
            if op == 0x37:                                    # LUI
                wb = (rd, instr & 0xFFFFF000)
            elif op == 0x17:                                  # AUIPC
                wb = (rd, u32(pc + (instr & 0xFFFFF000)))
            elif op == 0x6F:                                  # JAL
                imm = sext(((instr >> 31) << 20) | (((instr >> 12) & 0xFF) << 12)
                           | (((instr >> 20) & 1) << 11) | (((instr >> 21) & 0x3FF) << 1), 21)
                tgt = u32(pc + imm)
                if tgt & 3:
                    self._trap(0, tgt, pc)
                    return None
                wb = (rd, next_pc)
                next_pc = tgt
            elif op == 0x67 and f3 == 0:                      # JALR
                tgt = u32(a + sext(instr >> 20, 12)) & ~1
                if tgt & 3:
                    self._trap(0, tgt, pc)
                    return None
                wb = (rd, next_pc)
                next_pc = tgt
            elif op == 0x63:                                  # branches
                imm = sext(((instr >> 31) << 12) | (((instr >> 7) & 1) << 11)
                           | (((instr >> 25) & 0x3F) << 5) | (((instr >> 8) & 0xF) << 1), 13)
                taken = {0: a == b, 1: a != b, 4: s32(a) < s32(b), 5: s32(a) >= s32(b),
                         6: a < b, 7: a >= b}.get(f3)
                if taken is None:
                    raise ValueError("bad branch f3")
                if taken:
                    tgt = u32(pc + imm)
                    if tgt & 3:
                        self._trap(0, tgt, pc)
                        return None
                    next_pc = tgt
            elif op == 0x03:                                  # loads
                addr = u32(a + sext(instr >> 20, 12))
                size = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}.get(f3)
                if size is None:
                    raise ValueError("bad load f3")
                if addr % size:
                    self._trap(4, addr, pc)
                    return None
                v = self.read(addr, size)
                if f3 in (0, 1):
                    v = u32(sext(v, 8 * size))
                wb = (rd, v)
            elif op == 0x23:                                  # stores
                imm = sext(((instr >> 25) << 5) | ((instr >> 7) & 0x1F), 12)
                addr = u32(a + imm)
                size = {0: 1, 1: 2, 2: 4}.get(f3)
                if size is None:
                    raise ValueError("bad store f3")
                if addr % size:
                    self._trap(6, addr, pc)
                    return None
                val = b & ((1 << (8 * size)) - 1)
                self.write(addr, size, val)
                store = (addr, val, {1: 0, 2: 1, 4: 2}[size])
            elif op == 0x13:                                  # OP-IMM
                imm = sext(instr >> 20, 12)
                sh = rs2
                if f3 == 1 and f7 == 0:
                    r = u32(a << sh)
                elif f3 == 5 and f7 == 0:
                    r = a >> sh
                elif f3 == 5 and f7 == 0x20:
                    r = u32(s32(a) >> sh)
                elif f3 in (1, 5):
                    raise ValueError("bad shift funct7")
                else:
                    r = {0: u32(a + imm), 2: int(s32(a) < imm), 3: int(a < u32(imm)),
                         4: u32(a ^ imm), 6: u32(a | imm), 7: u32(a & imm)}[f3]
                wb = (rd, r)
            elif op == 0x33:                                  # OP
                key = (f3, f7)
                table = {(0, 0): u32(a + b), (0, 0x20): u32(a - b),
                         (1, 0): u32(a << (b & 31)), (2, 0): int(s32(a) < s32(b)),
                         (3, 0): int(a < b), (4, 0): a ^ b,
                         (5, 0): a >> (b & 31), (5, 0x20): u32(s32(a) >> (b & 31)),
                         (6, 0): a | b, (7, 0): a & b}
                if key not in table:
                    raise ValueError("bad OP funct")
                wb = (rd, table[key])
            elif op == 0x0F:                                  # FENCE / FENCE.I
                if f3 not in (0, 1):
                    raise ValueError("bad fence")
            elif op == 0x73:                                  # SYSTEM
                if f3 == 0:
                    if instr == 0x00000073:                   # ECALL (M-mode)
                        self._trap(11, 0, pc)
                        return None
                    if instr == 0x00100073:                   # EBREAK
                        self._trap(3, pc, pc)
                        return None
                    if instr == 0x30200073:                   # MRET
                        st = self.csr["mstatus"]
                        mpie = (st >> 7) & 1
                        self.csr["mstatus"] = ((st & ~0x1888) | (mpie << 3)
                                               | (1 << 7)) & MSTATUS_WMASK
                        next_pc = self.csr["mepc"]
                    else:
                        raise ValueError("bad system")
                else:                                         # Zicsr
                    csr_a = instr >> 20
                    src = rs1 if f3 >= 5 else a
                    old = self._csr_read(csr_a)               # may raise -> illegal
                    kind = f3 & 3
                    if kind == 1:
                        self._csr_write(csr_a, src)
                    elif kind == 2:
                        if rs1 != 0:
                            self._csr_write(csr_a, old | src)
                    elif kind == 3:
                        if rs1 != 0:
                            self._csr_write(csr_a, old & ~src)
                    else:
                        raise ValueError("bad csr f3")
                    wb = (rd, old)
            else:
                raise ValueError("bad opcode")
        except ValueError:
            self._trap(2, instr, pc)
            return None

        if wb is not None and wb[0] != 0:
            lane.x[wb[0]] = u32(wb[1])
        lane.pc = next_pc
        self.instret += 1
        commit = {"pc": pc, "instr": instr,
                  "rd": wb[0] if (wb and wb[0] != 0) else 0,
                  "wdata": u32(wb[1]) if (wb and wb[0] != 0) else 0,
                  "store": store}
        if store is not None and self.tohost is not None and store[0] == self.tohost:
            raise Halt(store[1])
        return commit


def fmt_commit(c: dict) -> str:
    s = f"{c['pc']:08x}:{c['instr']:08x}"
    if c["rd"]:
        s += f":x{c['rd']}={c['wdata']:08x}"
    if c["store"] is not None:
        a, v, sz = c["store"]
        s += f":S@{a:08x}={v:08x}.{sz}"
    return s


def run_to_halt(iss: Iss, max_steps: int = 20_000_000):
    """Returns (lines, tohost_value or None if max_steps hit)."""
    lines = []
    for _ in range(max_steps):
        try:
            c = iss.step()
        except Halt as h:
            # the halting store itself retires and is logged
            return lines + [_last_halt_line(iss, h)], h.value
        if c is not None:
            lines.append(fmt_commit(c))
    return lines, None


def _last_halt_line(iss: Iss, h: Halt) -> str:
    # reconstruct the tohost store commit (step raised after state update)
    pc = u32(iss.lane.pc - 4)
    instr = iss.read(pc, 4)
    return fmt_commit({"pc": pc, "instr": instr, "rd": 0, "wdata": 0,
                       "store": (iss.tohost, h.value, 2)})


if __name__ == "__main__":
    # self-check: hand-assembled snippet — li/add/store/load/branch/jal
    prog = bytes.fromhex(
        "9302a002"   # 80000000 addi x5,x0,42
        "13035000"   # 80000004 addi x6,x0,5
        "b3836200"   # 80000008 add  x7,x5,x6
        "37140080"   # 8000000c lui  x8,0x80001
        "23227400"   # 80000010 sw   x7,4(x8)
        "83424400"   # 80000014 lw   x5,4(x8)
        "63840200"   # 80000018 beq  x5,x0,+8 (not taken)
        "6f00c000"   # 8000001c jal  x0,+12 -> 80000028
        "13000000"   # 80000020 nop (skipped)
        "13000000"   # 80000024 nop (skipped)
        "23200401"   # 80000028 sw   x16,0(x8) -> tohost-ish end
    )
    iss = Iss(tohost=0x80001000)
    iss.load_blob(prog, 0x8000_0000)
    lines, val = run_to_halt(iss, 100)
    assert iss.lane.x[7] == 47 and iss.lane.x[5] == 47, lines
    assert any(":S@80001004=0000002f.2" in ln for ln in lines), lines
    assert val == 0 and lines[-1].endswith(":S@80001000=00000000.2"), (lines, val)
    print("iss self-check OK —", len(lines), "commits")
