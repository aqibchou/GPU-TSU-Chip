#!/usr/bin/env python3
"""M14 golden: lane-parallel SIMT ISS — W-agnostic single-warp model with
L lanes, shared PC, active mask, and the frozen max-PC reconvergence
scheme (docs/HARDWARE_ARCHITECTURE.md#simt-core §5). The RTL simt_core is bit-true to step().

Divergence-stack entry: {other (mask to run later), restart (its start
pc), join (mask after full reconvergence), reconv (pop threshold),
pending (other side not yet run)}. After each executed instruction with
computed next_pc:
    while stack and next_pc >= TOS.reconv:
        if TOS.pending:  # switch: other side runs from its restart
            TOS.pending = False; arrival = next_pc
            next_pc = TOS.restart; TOS.reconv = arrival
            # NO break: re-check so an empty arm (if-without-else)
            # merges immediately and the join runs once, merged (D-016)
        else:            # both sides arrived: merge, pc unchanged
            mask = TOS.join; pop
v1 traps (documented limits): divergent backward branch, divergent JALR,
divergence-stack overflow (depth > 8), misalignment on any enabled lane.
Traps quash the whole warp instruction (per-warp CSR trap state).
"""
import sys

L = 8
STACK_MAX = 8


def _u32(v):
    return v & 0xFFFFFFFF


def _s32(v):
    v &= 0xFFFFFFFF
    return v - 0x1_0000_0000 if v & 0x8000_0000 else v


def _sext(v, bits):
    m = 1 << (bits - 1)
    return (v ^ m) - m


class Halt(Exception):
    def __init__(self, value):
        self.value = value


class SimtWarp:
    """One warp: L lanes over a shared bytearray memory."""

    def __init__(self, mem, mem_base, reset_pc, warp_id, tohost=None):
        self.mem = mem                     # shared bytearray (RTL-visible)
        self.base = mem_base
        self.pc = reset_pc
        self.warp = warp_id
        self.tohost = tohost
        self.x = [[0] * 32 for _ in range(L)]
        self.mask = (1 << L) - 1
        self.stack = []
        self.csr = {0x300: 0, 0x304: 0, 0x305: 0, 0x340: 0, 0x341: 0,
                    0x342: 0, 0x343: 0}

    # ---------------- memory ----------------
    def _off(self, addr, size):
        off = addr - self.base
        assert 0 <= off <= len(self.mem) - size, hex(addr)
        return off

    def read(self, addr, size):
        o = self._off(addr, size)
        return int.from_bytes(self.mem[o:o + size], "little")

    def write(self, addr, size, val):
        o = self._off(addr, size)
        self.mem[o:o + size] = (val & ((1 << (8 * size)) - 1)).to_bytes(
            size, "little")

    def _trap(self, cause, tval):
        self.csr[0x341] = self.pc                       # mepc
        self.csr[0x342] = cause
        self.csr[0x343] = _u32(tval)
        ms = self.csr[0x300]
        self.csr[0x300] = ((ms & ~0x1888) | ((ms & 8) << 4) | 0x1800) \
            & 0xFFFFFFFF
        self.pc = self.csr[0x305] & 0xFFFFFFFC          # mtvec

    # ---------------- one warp instruction ----------------
    def step(self):
        """Execute one shared instruction across enabled lanes. Returns a
        commit dict or None (trap). Raises Halt on tohost store."""
        pc = self.pc
        instr = self.read(pc, 4)
        op = instr & 0x7F
        rd = (instr >> 7) & 0x1F
        f3 = (instr >> 12) & 7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        f7 = instr >> 25
        imm_i = _sext(instr >> 20, 12)
        imm_s = _sext(((instr >> 25) << 5) | ((instr >> 7) & 0x1F), 12)
        imm_b = _sext((((instr >> 31) & 1) << 12) | (((instr >> 7) & 1) << 11)
                      | (((instr >> 25) & 0x3F) << 5)
                      | (((instr >> 8) & 0xF) << 1), 13)
        imm_u = instr & 0xFFFFF000
        imm_j = _sext((((instr >> 31) & 1) << 20)
                      | (((instr >> 12) & 0xFF) << 12)
                      | (((instr >> 20) & 1) << 11)
                      | (((instr >> 21) & 0x3FF) << 1), 21)
        lanes = [ln for ln in range(L) if (self.mask >> ln) & 1]
        wb = {}                       # lane -> (rd, val)
        stores = []                   # (lane, addr, val, size_log2)
        next_pc = _u32(pc + 4)
        new_mask = self.mask

        def a(ln):
            return self.x[ln][rs1]

        def b(ln):
            return self.x[ln][rs2]

        if op == 0x37:                                            # LUI
            for ln in lanes:
                wb[ln] = (rd, imm_u)
        elif op == 0x17:                                          # AUIPC
            for ln in lanes:
                wb[ln] = (rd, _u32(pc + imm_u))
        elif op == 0x6F:                                          # JAL
            tgt = _u32(pc + imm_j)
            if tgt & 3:
                self._trap(0, tgt)
                return None
            for ln in lanes:
                wb[ln] = (rd, _u32(pc + 4))
            next_pc = tgt
        elif op == 0x67:                                          # JALR
            if f3 != 0:
                self._trap(2, instr)
                return None
            tgts = {_u32(a(ln) + imm_i) & 0xFFFFFFFE for ln in lanes}
            if len(tgts) > 1:
                self._trap(2, instr)                    # divergent JALR
                return None
            tgt = tgts.pop()
            if tgt & 3:
                self._trap(0, tgt)
                return None
            for ln in lanes:
                wb[ln] = (rd, _u32(pc + 4))
            next_pc = tgt
        elif op == 0x63:                                          # branches
            if f3 in (2, 3):
                self._trap(2, instr)
                return None
            taken = 0
            for ln in lanes:
                av, bv = a(ln), b(ln)
                t = {0: av == bv, 1: av != bv,
                     4: _s32(av) < _s32(bv), 5: _s32(av) >= _s32(bv),
                     6: av < bv, 7: av >= bv}[f3]
                if t:
                    taken |= 1 << ln
            tgt = _u32(pc + imm_b)
            if taken and tgt & 3:
                self._trap(0, tgt)
                return None
            not_taken = self.mask & ~taken
            if taken == 0:
                pass                                    # uniform not-taken
            elif not_taken == 0:
                next_pc = tgt                           # uniform taken
            else:                                       # DIVERGENT
                if tgt <= pc:
                    self._trap(2, instr)                # backward divergent
                    return None
                if len(self.stack) >= STACK_MAX:
                    self._trap(2, instr)
                    return None
                self.stack.append({"other": taken, "restart": tgt,
                                   "join": self.mask, "reconv": tgt,
                                   "pending": True})
                new_mask = not_taken
        elif op == 0x03:                                          # loads
            if f3 in (3, 6, 7):
                self._trap(2, instr)
                return None
            size = 1 << (f3 & 3)
            addrs = {ln: _u32(a(ln) + imm_i) for ln in lanes}
            for ln in lanes:
                if addrs[ln] % size:
                    self._trap(4, addrs[ln])
                    return None
            mem_lanes = [(ln, False, addrs[ln], 0, (1 << size) - 1)
                         for ln in lanes]
            for ln in lanes:                            # lane order = RTL
                v = self.read(addrs[ln], size)
                if f3 == 0:
                    v = _u32(_sext(v, 8))
                elif f3 == 1:
                    v = _u32(_sext(v, 16))
                wb[ln] = (rd, v)
        elif op == 0x23:                                          # stores
            if f3 > 2:
                self._trap(2, instr)
                return None
            size = 1 << f3
            addrs = {ln: _u32(a(ln) + imm_s) for ln in lanes}
            for ln in lanes:
                if addrs[ln] % size:
                    self._trap(6, addrs[ln])
                    return None
            mem_lanes = [(ln, True, addrs[ln], b(ln), (1 << size) - 1)
                         for ln in lanes]
            for ln in lanes:
                self.write(addrs[ln], size, b(ln))
                stores.append((ln, addrs[ln],
                               b(ln) & ((1 << (8 * size)) - 1), f3))
        elif op == 0x13:                                          # OP-IMM
            for ln in lanes:
                av = a(ln)
                if f3 == 0:
                    v = _u32(av + imm_i)
                elif f3 == 2:
                    v = int(_s32(av) < imm_i)
                elif f3 == 3:
                    v = int(av < _u32(imm_i))
                elif f3 == 4:
                    v = _u32(av ^ imm_i)
                elif f3 == 6:
                    v = _u32(av | imm_i)
                elif f3 == 7:
                    v = _u32(av & imm_i)
                elif f3 == 1:
                    if f7 != 0:
                        self._trap(2, instr)
                        return None
                    v = _u32(av << (rs2 & 31))
                else:
                    if f7 == 0:
                        v = av >> (rs2 & 31)
                    elif f7 == 0x20:
                        v = _u32(_s32(av) >> (rs2 & 31))
                    else:
                        self._trap(2, instr)
                        return None
                wb[ln] = (rd, v)
        elif op == 0x33:                                          # OP
            for ln in lanes:
                av, bv = a(ln), b(ln)
                key = (f7, f3)
                if key == (0, 0):
                    v = _u32(av + bv)
                elif key == (0x20, 0):
                    v = _u32(av - bv)
                elif key == (0, 1):
                    v = _u32(av << (bv & 31))
                elif key == (0, 2):
                    v = int(_s32(av) < _s32(bv))
                elif key == (0, 3):
                    v = int(av < bv)
                elif key == (0, 4):
                    v = _u32(av ^ bv)
                elif key == (0, 5):
                    v = av >> (bv & 31)
                elif key == (0x20, 5):
                    v = _u32(_s32(av) >> (bv & 31))
                elif key == (0, 6):
                    v = _u32(av | bv)
                elif key == (0, 7):
                    v = _u32(av & bv)
                else:
                    self._trap(2, instr)
                    return None
                wb[ln] = (rd, v)
        elif op == 0x0F:                                          # FENCE
            if f3 > 1:
                self._trap(2, instr)
                return None
        elif op == 0x73:                                          # SYSTEM
            if f3 == 0:
                if instr == 0x00000073:
                    self._trap(11, 0)
                    return None
                if instr == 0x00100073:
                    self._trap(3, pc)
                    return None
                if instr == 0x30200073:                 # MRET
                    next_pc = self.csr[0x341]
                    ms = self.csr[0x300]
                    self.csr[0x300] = ((ms >> 4) & 8) | 0x80
                else:
                    self._trap(2, instr)
                    return None
            elif f3 == 4:
                self._trap(2, instr)
                return None
            else:                                       # Zicsr
                csr_a = instr >> 20
                stored = csr_a in self.csr
                ro = csr_a in (0xF11, 0xF12, 0xF13, 0xF14, 0xF15, 0x301)
                dummy = csr_a in (0x310, 0x344, 0xB00, 0xB02, 0xB80, 0xB82,
                                  0x320) or 0x3A0 <= csr_a <= 0x3A3 \
                    or 0x3B0 <= csr_a <= 0x3BF
                if not (stored or ro or dummy):
                    self._trap(2, instr)
                    return None
                wr = (f3 & 3) == 1 or rs1 != 0
                if wr and ro:
                    self._trap(2, instr)
                    return None
                for ln in lanes:
                    if csr_a == 0xF14:
                        old = self.warp * L + ln        # global tid
                    elif csr_a == 0x301:
                        old = 0x40000100
                    elif stored:
                        old = self.csr[csr_a]
                    else:
                        old = 0
                    wb[ln] = (rd, old)
                if wr and stored:
                    # per-warp CSR: lanes agree on src unless rs1-divergent
                    srcs = {(rs1 if f3 & 4 else a(ln)) for ln in lanes}
                    if len(srcs) > 1:
                        self._trap(2, instr)
                        return None
                    src = srcs.pop()
                    old = self.csr[csr_a]
                    kind = f3 & 3
                    val = src if kind == 1 else \
                        (old | src if kind == 2 else old & ~src)
                    if csr_a == 0x300:
                        val &= 0x1888
                    elif csr_a in (0x305, 0x341):
                        val &= 0xFFFFFFFC
                    self.csr[csr_a] = _u32(val)
        else:
            self._trap(2, instr)
            return None

        # apply writebacks (x0 immune)
        for ln, (r, v) in wb.items():
            if r:
                self.x[ln][r] = _u32(v)

        commit = {"pc": pc, "instr": instr, "mask": self.mask,
                  "wb": {ln: wb[ln] for ln in wb if wb[ln][0]},
                  "stores": stores,
                  "mem_lanes": locals().get("mem_lanes", None)}

        # reconvergence (the frozen scheme)
        mask = new_mask
        while self.stack and next_pc >= self.stack[-1]["reconv"]:
            tos = self.stack[-1]
            if tos["pending"]:
                # switch to the other side, then RE-CHECK: if the other
                # side's restart is already >= the (updated) reconv — the
                # if-without-else shape — the loop merges immediately so
                # the join instruction runs once with the merged mask.
                # (The original 'break' here was a SPEC bug found by
                # compiled C kernels at M17: fall-through lanes skipped
                # the join instruction; see D-016.)
                tos["pending"] = False
                arrival = next_pc
                next_pc = tos["restart"]
                tos["reconv"] = arrival
                mask = tos["other"]
            else:
                mask = tos["join"]
                self.stack.pop()
        self.mask = mask
        self.pc = next_pc

        if self.tohost is not None:
            for (_ln, addr, val, _sz) in stores:
                if addr == self.tohost:
                    raise Halt(val)
        return commit
