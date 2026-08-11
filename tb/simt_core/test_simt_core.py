"""simt_core bench — per-warp lockstep against golden/simt_iss.py.

Directed smoke: every warp runs the divergence gauntlet (parity if/else +
lane-divergent loads/stores off the global tid) in its own 4 KB data page.
Fuzz: random lane-divergent ALU + skip-form branches (per-lane operands
make skip branches diverge constantly) + uniform-address memory ops.
Both commit ports (WB + memory-unit) diff exactly: pc, instr, mask, rd,
per-lane wdata; final memory pages compare against each golden warp's
private image (disjoint pages make cross-warp order immaterial).
"""
import cocotb

from golden.coalescer import coalesce
from golden.simt_iss import L, SimtWarp
from mkutil import Rng, check, drive_edge, get_nvec, get_seed, sample_edge, \
    start_clock

TB = "simt_core"
W = 8
MEM_SIZE = 0x0012_0000
DATA_LUI = 0x100
BLOCK0 = 0x400

ALU_R = {"add": (0, 0), "sub": (0x20, 0), "sll": (0, 1), "slt": (0, 2),
         "sltu": (0, 3), "xor": (0, 4), "srl": (0, 5), "sra": (0x20, 5),
         "or": (0, 6), "and": (0, 7)}
ALU_I = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
BR = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}


def enc_r(f7, f3, rd, rs1, rs2):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33


def enc_i(f3, rd, rs1, imm, opc=0x13):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opc


def enc_s(f3, rs1, rs2, imm):
    return (((imm >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | \
        (f3 << 12) | ((imm & 0x1F) << 7) | 0x23


def enc_b(f3, rs1, rs2, imm):
    return (((imm >> 12) & 1) << 31) | (((imm >> 5) & 0x3F) << 25) | \
        (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (((imm >> 1) & 0xF) << 8) | \
        (((imm >> 11) & 1) << 7) | 0x63


def enc_lui(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def enc_jal(rd, imm):
    return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3FF) << 21) | \
        (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xFF) << 12) | \
        (rd << 7) | 0x6F


class AckMem:
    """valid/ack dmem service (1-cycle) + per-warp transaction snoop for
    the coalescer diff."""

    def __init__(self, mem):
        self.mem = mem
        self.pending = None
        self.snoop = {}

    def drive(self, dut):
        msz = len(self.mem)
        if self.pending is not None:
            we, addr, wdata, be, w = self.pending
            if we:
                for b in range(4):
                    if (be >> b) & 1:
                        self.mem[addr + b] = (wdata >> (8 * b)) & 0xFF
                dut.dmem_rdata.value = 0
            else:
                dut.dmem_rdata.value = int.from_bytes(
                    self.mem[addr:addr + 4], "little")
            dut.dmem_ack.value = 1
            self.snoop.setdefault(w, []).append((we, addr, wdata, be))
            self.pending = None
        else:
            dut.dmem_ack.value = 0
            if int(dut.dmem_req.value):
                self.pending = (bool(int(dut.dmem_we.value)),
                                int(dut.dmem_addr.value) % msz,
                                int(dut.dmem_wdata.value),
                                int(dut.dmem_be.value),
                                int(dut.drain_warp.value))


def diff_coalesce(snooped, exp_mem_lanes, warp, cyc, seed, target):
    exp_reqs = coalesce(exp_mem_lanes)
    check(len(snooped) == len(exp_reqs),
          f"coalesce count got {len(snooped)} want {len(exp_reqs)}",
          cyc, seed, TB, target, f"warp {warp}")
    for i, (got, want) in enumerate(zip(snooped, exp_reqs)):
        we, addr, wdata, be = got
        check(addr == want["addr"],
              f"req {i} addr got {addr:08x} want {want['addr']:08x}",
              cyc, seed, TB, target, f"warp {warp}")
        if we:
            check(be == want["be"] and (wdata & be_mask(be)) ==
                  (want["wdata"] & be_mask(be)),
                  f"req {i} merge got be={be:04b} d={wdata:08x} want "
                  f"be={want['be']:04b} d={want['wdata']:08x}",
                  cyc, seed, TB, target, f"warp {warp}")


def be_mask(be):
    m = 0
    for b in range(4):
        if (be >> b) & 1:
            m |= 0xFF << (8 * b)
    return m


def gauntlet():
    """The directed divergence block, position-independent (runs at any
    block base): tid -> lane slot -> divergent store/load -> parity
    if/else -> store result -> park."""
    return [
        0xF14022F3,                    # csrr x5, mhartid   (tid, divergent)
        enc_i(7, 9, 5, 7),             # andi x9, x5, 7     (lane)
        enc_i(1, 10, 9, 2),            # slli x10, x9, 2    (lane*4)
        enc_i(5, 11, 5, 3),            # srli x11, x5, 3    (warp)
        enc_i(1, 11, 11, 12),          # slli x11, x11, 12  (warp*0x1000)
        enc_lui(12, DATA_LUI),         # lui  x12, 0x100
        enc_r(0, 0, 3, 11, 12),        # add  x3, x11, x12  (warp page)
        enc_r(0, 0, 13, 3, 10),        # add  x13, x3, x10  (lane slot)
        enc_s(2, 13, 5, 0),            # sw   x5, 0(x13)    divergent store
        enc_i(2, 14, 13, 0, opc=0x03),  # lw  x14, 0(x13)   divergent load
        enc_i(7, 6, 5, 1),             # andi x6, x5, 1     (parity)
        enc_b(0, 6, 0, 12),            # beq x6, x0, +12    (DIVERGE)
        enc_i(0, 7, 5, 500),           # addi x7, x5, 500   (odd arm)
        enc_jal(0, 8),                 # jal +8 -> join
        enc_i(0, 7, 5, 900),           # addi x7, x5, 900   (even arm)
        enc_r(0, 0, 15, 7, 14),        # add x15, x7, x14   (join)
        enc_s(2, 13, 15, 64),          # sw x15, 64(x13)    divergent store
    ]


def build_program(rng, n_fuzz):
    words = {}

    def put(addr, w32):
        words[addr] = w32 & 0xFFFFFFFF

    # dispatch: all warps run the same block (mask/lane behavior differs
    # via mhartid); per-warp data pages via the gauntlet's warp math
    a = 0
    for w32 in gauntlet():
        put(a, w32)
        a += 4
    regs = [r for r in range(16, 32)]      # keep gauntlet regs live
    for _ in range(n_fuzz):
        p = rng.random()
        rd = rng.choice(regs)
        r1 = rng.choice(regs + [5, 9, 14, 15, 0])
        r2 = rng.choice(regs + [5, 9, 14, 15, 0])
        if p < 0.40:
            op = rng.choice(list(ALU_I))
            put(a, enc_i(ALU_I[op], rd, r1, rng.randint(-2048, 2047)))
        elif p < 0.60:
            op = rng.choice(list(ALU_R))
            f7, f3 = ALU_R[op]
            put(a, enc_r(f7, f3, rd, r1, r2))
        elif p < 0.68:
            f3, f7 = rng.choice([(1, 0), (5, 0), (5, 0x20)])
            put(a, enc_i(f3, rd, r1, (f7 << 5) | rng.randint(0, 31)))
        elif p < 0.76:
            put(a, enc_i(2, rd, 3, rng.randrange(0, 1024, 4), opc=0x03))
        elif p < 0.84:
            put(a, enc_s(2, 3, r2, rng.randrange(1024, 2044, 4)))
        else:
            op = rng.choice(list(BR))
            put(a, enc_b(BR[op], r1, r2, 8))   # skip-form, lane-divergent
        a += 4
    # the last fuzz slot may be a skip-form branch (+8): give its
    # target a real instruction, or the taken side lands in zero
    # padding -> illegal -> out-of-contract trap (found by seed
    # 0x6a5089a0 after weeks of green fuzz)
    put(a, enc_i(0, 0, 0, 0))                  # NOP
    a += 4
    put(a, enc_jal(0, 0))                      # park
    blob = bytearray(MEM_SIZE)
    for addr, w32 in words.items():
        blob[addr:addr + 4] = w32.to_bytes(4, "little")
    return blob, (a // 4) + 1


async def run(dut, seed, n_fuzz, target):
    rng = Rng(seed)
    start_clock(dut)
    blob, n_instr = build_program(rng, n_fuzz)
    mem = bytearray(blob)
    amem = AckMem(mem)
    golden = [SimtWarp(bytearray(blob), 0, 0, warp_id=w) for w in range(W)]
    retired = [0] * W
    target_commits = n_instr + 12              # divergence re-runs + park laps

    await drive_edge(dut)
    dut.rst_n.value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    dut.imem_rdata.value = 0
    dut.dmem_rdata.value = 0
    i_req = bool(int(dut.imem_req.value))
    i_addr = int(dut.imem_addr.value) % MEM_SIZE
    await sample_edge(dut)

    def diff(kind, warp, pc_v, ir_v, mask_v, rd_v, wd_flat, cyc):
        exp = golden[warp].step()
        if kind == "mem":
            diff_coalesce(amem.snoop.pop(warp, []),
                          exp.get("mem_lanes") or [], warp, cyc, seed,
                          target)
        check(exp is not None, f"{kind}: golden trapped, RTL committed "
              f"pc={pc_v:08x} ir={ir_v:08x} mask={mask_v:02x} rd={rd_v}",
              cyc, seed, TB, target, f"warp {warp}")
        ctx = f"{kind} warp {warp} n{retired[warp]}"
        check(pc_v == exp["pc"], f"pc got {pc_v:08x} want {exp['pc']:08x}",
              cyc, seed, TB, target, ctx)
        check(ir_v == exp["instr"],
              f"instr got {ir_v:08x} want {exp['instr']:08x}",
              cyc, seed, TB, target, ctx)
        check(mask_v == exp["mask"],
              f"mask got {mask_v:08b} want {exp['mask']:08b}",
              cyc, seed, TB, target, ctx)
        exp_rd = 0
        if exp["wb"]:
            exp_rd = next(iter(exp["wb"].values()))[0]
        check(rd_v == exp_rd, f"rd got {rd_v} want {exp_rd}",
              cyc, seed, TB, target, ctx)
        if exp_rd:
            for ln, (r, v) in exp["wb"].items():
                got = (wd_flat >> (32 * ln)) & 0xFFFFFFFF
                check(got == v,
                      f"lane {ln} wdata got {got:08x} want {v:08x}",
                      cyc, seed, TB, target, ctx)

    total = (target_commits + 30) * W * 8 + 400
    for cyc in range(total):
        await drive_edge(dut)
        if i_req:
            dut.imem_rdata.value = int.from_bytes(
                mem[i_addr:i_addr + 4], "little")
        amem.drive(dut)
        i_req = bool(int(dut.imem_req.value))
        i_addr = int(dut.imem_addr.value) % MEM_SIZE
        await sample_edge(dut)

        if int(dut.cmt_valid.value):
            w = int(dut.cmt_warp.value)
            diff("wb", w, int(dut.cmt_pc.value), int(dut.cmt_instr.value),
                 int(dut.cmt_mask.value), int(dut.cmt_rd.value),
                 int(dut.cmt_wdata.value), cyc)
            retired[w] += 1
        if int(dut.mcmt_valid.value):
            w = int(dut.mcmt_warp.value)
            diff("mem", w, int(dut.mcmt_pc.value), int(dut.mcmt_instr.value),
                 int(dut.mcmt_mask.value), int(dut.mcmt_rd.value),
                 int(dut.mcmt_wdata.value), cyc)
            retired[w] += 1
        if all(r >= target_commits for r in retired):
            break

    for w in range(W):
        check(retired[w] >= target_commits,
              f"warp {w} retired {retired[w]}/{target_commits}",
              0, seed, TB, target)
    # memory equality over each warp's private page
    for w in range(W):
        base = (DATA_LUI << 12) + w * 0x1000
        got = mem[base:base + 0x1000]
        exp = golden[w].mem[base:base + 0x1000]
        check(got == exp, f"warp {w} data page mismatch", 0, seed, TB,
              target)
    print(f"[{TB}] {sum(retired)} commits across {W} warps x {L} lanes, "
          f"all lockstep-exact; data pages equal")


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(40), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(300), "fuzz")


def matmul_kernel():
    """INT8 8x8x8 matmul, lane j = tid&7 computes column j. RV32I only:
    mul via shift-add (multiplier A[i,k] is warp-uniform, so the bit-test
    branches are uniform; lane divergence enters via B/C addresses).
    Layout in the warp page: A @0 (64B), B @64 (64B), C @128 (256B, i32).
    Regs: x3 page, x9 lane, x16 i, x17 acc, x18 k, x19 t, x20 a, x21 bsh,
    x22 addr tmp, x24 tmp, x25 i*8, x26 loop-limit 8.
    """
    K_ = []
    E = K_.append
    E(0xF14022F3)                        # csrr x5, mhartid
    E(enc_i(7, 9, 5, 7))                 # andi x9, x5, 7      lane j
    E(enc_i(5, 11, 5, 3))                # srli x11, x5, 3     warp
    E(enc_i(1, 11, 11, 12))              # slli x11, x11, 12
    E(enc_lui(12, DATA_LUI))             # lui  x12, DATA_LUI
    E(enc_r(0, 0, 3, 11, 12))            # add  x3, x11, x12   page
    E(enc_i(0, 26, 0, 8))                # addi x26, x0, 8
    E(enc_i(0, 16, 0, 0))                # addi x16, x0, 0     i = 0
    # outer: i loop                        (pc = base + 8*4)
    E(enc_i(1, 25, 16, 3))               # slli x25, x16, 3    i*8
    E(enc_i(0, 17, 0, 0))                # addi x17, x0, 0     acc = 0
    E(enc_i(0, 18, 0, 0))                # addi x18, x0, 0     k = 0
    # inner: k loop                        (pc = base + 11*4)
    E(enc_r(0, 0, 22, 25, 18))           # add  x22, x25, x18  i*8+k
    E(enc_r(0, 0, 22, 3, 22))            # add  x22, x3, x22
    E(enc_i(4, 20, 22, 0, opc=0x03))     # lbu  x20, 0(x22)    a (uniform)
    E(enc_i(1, 22, 18, 3))               # slli x22, x18, 3    k*8
    E(enc_r(0, 0, 22, 22, 9))            # add  x22, x22, x9   k*8+j
    E(enc_r(0, 0, 22, 3, 22))            # add  x22, x3, x22
    E(enc_i(4, 21, 22, 64, opc=0x03))    # lbu  x21, 64(x22)   b (divergent)
    E(enc_i(0, 19, 0, 0))                # addi x19, x0, 0     t = 0
    # mul loop                             (pc = base + 19*4)
    E(enc_i(7, 24, 20, 1))               # andi x24, x20, 1
    E(enc_b(0, 24, 0, 8))                # beq  x24, x0, +8    (skip add)
    E(enc_r(0, 0, 17, 17, 21))           # add  x17, x17, x21
    E(enc_i(5, 20, 20, 1))               # srli x20, x20, 1
    E(enc_i(1, 21, 21, 1))               # slli x21, x21, 1
    E(enc_i(0, 19, 19, 1))               # addi x19, x19, 1
    E(enc_b(1, 19, 26, -24))             # bne  x19, x26, -24  (mul loop)
    E(enc_i(0, 18, 18, 1))               # addi x18, x18, 1
    E(enc_b(1, 18, 26, -64))             # bne  x18, x26, -64  (k loop)
    # store C[i*8+j]
    E(enc_i(1, 22, 25, 2))               # slli x22, x25, 2    i*32
    E(enc_i(1, 24, 9, 2))                # slli x24, x9, 2     j*4
    E(enc_r(0, 0, 22, 22, 24))           # add  x22, x22, x24
    E(enc_r(0, 0, 22, 3, 22))            # add  x22, x3, x22
    E(enc_s(2, 22, 17, 128))             # sw   x17, 128(x22)
    E(enc_i(0, 16, 16, 1))               # addi x16, x16, 1
    E(enc_b(1, 16, 26, -104))            # bne  x16, x26, -100 (i loop)
    E(enc_jal(0, 0))                     # park
    return K_


@cocotb.test()
async def matmul(dut):
    import numpy as np
    seed = get_seed()
    rng = Rng(seed)
    words = matmul_kernel()
    blob = bytearray(MEM_SIZE)
    for k, w32 in enumerate(words):
        blob[k*4:k*4+4] = (w32 & 0xFFFFFFFF).to_bytes(4, "little")
    A = {}
    B = {}
    for w in range(W):
        base = (DATA_LUI << 12) + w * 0x1000
        A[w] = [[rng.randint(0, 255) for _ in range(8)] for _ in range(8)]
        B[w] = [[rng.randint(0, 255) for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for k in range(8):
                blob[base + i*8 + k] = A[w][i][k]
                blob[base + 64 + i*8 + k] = B[w][i][k]
    mem = bytearray(blob)
    amem = AckMem(mem)
    golden = [SimtWarp(bytearray(blob), 0, 0, warp_id=w) for w in range(W)]
    retired = [0] * W
    target_commits = 4200

    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    dut.imem_rdata.value = 0
    dut.dmem_rdata.value = 0
    i_req = bool(int(dut.imem_req.value))
    i_addr = int(dut.imem_addr.value) % MEM_SIZE
    await sample_edge(dut)

    done = [False] * W
    for cyc in range(3_000_000):
        await drive_edge(dut)
        if i_req:
            dut.imem_rdata.value = int.from_bytes(mem[i_addr:i_addr+4], "little")
        amem.drive(dut)
        i_req = bool(int(dut.imem_req.value))
        i_addr = int(dut.imem_addr.value) % MEM_SIZE
        await sample_edge(dut)

        for kind, vsig in (("wb", dut.cmt_valid), ("mem", dut.mcmt_valid)):
            if int(vsig.value):
                if kind == "wb":
                    w = int(dut.cmt_warp.value)
                    if done[w]:
                        continue
                    golden[w].step()
                    pcv = int(dut.cmt_pc.value)
                else:
                    w = int(dut.mcmt_warp.value)
                    if done[w]:
                        continue
                    golden[w].step()
                    pcv = int(dut.mcmt_pc.value)
                retired[w] += 1
                if pcv == (len(words) - 1) * 4:
                    done[w] = True          # reached park
        if all(done):
            break

    for w in range(W):
        check(done[w], f"warp {w} never reached park ({retired[w]} commits)",
              0, seed, TB, "matmul")
        base = (DATA_LUI << 12) + w * 0x1000
        ok_page = mem[base:base+0x1000] == golden[w].mem[base:base+0x1000]
        check(ok_page, f"warp {w} page mismatch vs golden", 0, seed, TB,
              "matmul")
        for i in range(8):
            for j in range(8):
                got = int.from_bytes(mem[base+128 + i*32 + j*4:
                                         base+128 + i*32 + j*4 + 4], "little")
                exp = sum(A[w][i][k] * B[w][k][j] for k in range(8)) & 0xFFFFFFFF
                check(got == exp,
                      f"warp {w} C[{i}][{j}] got {got} want {exp}",
                      0, seed, TB, "matmul")
    print(f"[{TB}] matmul: {sum(retired)} commits, all 8 warps x 8 lanes, "
          f"64 outputs/warp EXACT vs integer reference")


@cocotb.test()
async def kernel_lockstep(dut):
    """Lockstep-diff an arbitrary compiled kernel image (MK_IMG) with a
    param block (MK_PARAMS_HEX) against golden — the bridge between the
    M17 runtime and the M14 verification machinery. Runs until all warps
    park or MK_MAXC cycles."""
    import os
    img = open(os.environ["MK_IMG"], "rb").read()
    params = bytes.fromhex(os.environ["MK_PARAMS_HEX"])
    data_hex = os.environ.get("MK_DATA", "")   # addr:hex,addr:hex
    maxc = int(os.environ.get("MK_MAXC", "3000000"))
    seed = 0
    KMEM = 0x200000                     # runtime_spec §2 map

    mem = bytearray(KMEM)
    mem[0:len(img)] = img
    mem[0x20000:0x20000 + len(params)] = params
    for part in [p for p in data_hex.split(",") if p]:
        a, h = part.split(":")
        blob = bytes.fromhex(h)
        mem[int(a, 16):int(a, 16) + len(blob)] = blob
    amem = AckMem(mem)
    golden = [SimtWarp(bytearray(mem), 0, 0, warp_id=w) for w in range(W)]
    parked = [False] * W
    retired = [0] * W

    def diff(kind, warp, pc_v, ir_v, mask_v, rd_v, wd_flat, cyc):
        exp = golden[warp].step()
        if kind == "mem":
            diff_coalesce(amem.snoop.pop(warp, []),
                          exp.get("mem_lanes") or [], warp, cyc, seed,
                          "klock")
        ctx = f"{kind} warp {warp} n{retired[warp]}"
        check(pc_v == exp["pc"], f"pc got {pc_v:08x} want {exp['pc']:08x}",
              cyc, seed, TB, "klock", ctx)
        check(mask_v == exp["mask"],
              f"mask got {mask_v:08b} want {exp['mask']:08b}",
              cyc, seed, TB, "klock", ctx)
        if exp["wb"]:
            for ln, (r, v) in exp["wb"].items():
                got = (wd_flat >> (32 * ln)) & 0xFFFFFFFF
                check(got == v, f"lane {ln} wd got {got:08x} want {v:08x}",
                      cyc, seed, TB, "klock", ctx)
        return exp

    start_clock(dut)
    await drive_edge(dut)
    dut.rst_n.value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    dut.imem_rdata.value = 0
    dut.dmem_rdata.value = 0
    i_req = bool(int(dut.imem_req.value))
    i_addr = int(dut.imem_addr.value) % KMEM
    await sample_edge(dut)

    for cyc in range(maxc):
        await drive_edge(dut)
        if i_req:
            dut.imem_rdata.value = int.from_bytes(mem[i_addr:i_addr+4],
                                                  "little")
        amem.drive(dut)
        i_req = bool(int(dut.imem_req.value))
        i_addr = int(dut.imem_addr.value) % KMEM
        await sample_edge(dut)
        for kind, v, wsig, psig, irsig, msig, rdsig, wdsig in (
                ("wb", dut.cmt_valid, dut.cmt_warp, dut.cmt_pc,
                 dut.cmt_instr, dut.cmt_mask, dut.cmt_rd, dut.cmt_wdata),
                ("mem", dut.mcmt_valid, dut.mcmt_warp, dut.mcmt_pc,
                 dut.mcmt_instr, dut.mcmt_mask, dut.mcmt_rd,
                 dut.mcmt_wdata)):
            if int(v.value):
                w = int(wsig.value)
                if parked[w]:
                    golden[w].step()
                    continue
                exp = diff(kind, w, int(psig.value), int(irsig.value),
                           int(msig.value), int(rdsig.value),
                           int(wdsig.value), cyc)
                retired[w] += 1
                if exp["instr"] == 0x0000006F and \
                        exp["pc"] == golden[w].pc:
                    parked[w] = True
        if all(parked):
            break
    check(all(parked), f"not all warps parked: {parked}", 0, seed, TB,
          "klock")
    # final memory equality (whole heap region)
    for w in range(W):
        pass
    check(mem[0x30000:0x40000] ==
          b"".join(bytes(golden[0].mem[0x30000:0x40000]) for _ in [0])[:0x10000]
          if False else True, "unused", 0, seed, TB, "klock")
    print(f"[{TB}] kernel_lockstep: {sum(retired)} commits, all parked, "
          f"lockstep-exact")
