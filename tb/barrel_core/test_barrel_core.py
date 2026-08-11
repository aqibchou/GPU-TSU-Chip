"""barrel_core bench — W-way per-warp lockstep against golden/iss.py.

Every warp runs its own random program block (dispatched by an mhartid
stub) with a disjoint 4 KB data page, and every commit record is diffed
EXACTLY against that warp's own ISS instance: pc, instr, rd, wdata, and
store address/data/size. The in-bench encoder does not weaken the trust
chain (both sides decode the same bytes; GAS-encoded riscv-tests + torture
arrive with the M15 SIMT-DoD gate via the P1 machinery).

Memory model: python arrays with the spec §6 contract — request this
cycle, response next cycle, always ready.
"""
import cocotb

from golden.iss import Iss
from mkutil import Rng, check, drive_edge, get_nvec, get_seed, sample_edge, \
    start_clock

TB = "barrel_core"
W = 8
MEM_SIZE = 0x0012_0000
DATA_LUI = 0x100                    # warp w data page = (0x100 + w) << 12
BLOCK0 = 0x400                      # warp w code block = BLOCK0 + w*0x1000

ALU_R = {"add": (0, 0), "sub": (0x20, 0), "sll": (0, 1), "slt": (0, 2),
         "sltu": (0, 3), "xor": (0, 4), "srl": (0, 5), "sra": (0x20, 5),
         "or": (0, 6), "and": (0, 7)}
ALU_I = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
BR = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}
LD = {"lb": (0, 1), "lh": (1, 2), "lw": (2, 4), "lbu": (4, 1), "lhu": (5, 2)}
ST = {"sb": (0, 1), "sh": (1, 2), "sw": (2, 4)}


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


def build_program(rng, n_per_warp):
    """Returns the shared blob: dispatch stub + W per-warp random blocks."""
    words = {}

    def put(addr, w32):
        words[addr] = w32 & 0xFFFFFFFF

    # stub @0: csrrs x5, mhartid, x0 ; slli x5,x5,12 ; jalr x0, 0x400(x5)
    put(0x0, 0xF14022F3)
    put(0x4, enc_i(1, 5, 5, 12))                       # slli x5, x5, 12
    put(0x8, enc_i(0, 0, 5, BLOCK0, opc=0x67))         # jalr x0, BLOCK0(x5)

    regs = [r for r in range(4, 32)]                   # x3 data base, x1/x2 free too
    for w in range(W):
        a = BLOCK0 + w * 0x1000
        put(a, enc_lui(3, DATA_LUI + w))               # x3 = data page
        a += 4
        for _ in range(n_per_warp):
            p = rng.random()
            rd = rng.choice(regs)
            r1 = rng.choice(regs + [0, 3])
            r2 = rng.choice(regs + [0, 3])
            if p < 0.35:
                op = rng.choice(list(ALU_I))
                put(a, enc_i(ALU_I[op], rd, r1, rng.randint(-2048, 2047)))
            elif p < 0.50:
                op = rng.choice(list(ALU_R))
                f7, f3 = ALU_R[op]
                put(a, enc_r(f7, f3, rd, r1, r2))
            elif p < 0.58:
                f3, f7 = rng.choice([(1, 0), (5, 0), (5, 0x20)])
                put(a, enc_i(f3, rd, r1, (f7 << 5) | rng.randint(0, 31)))
            elif p < 0.68:
                op = rng.choice(list(LD))
                f3, al = LD[op]
                put(a, enc_i(f3, rd, 3, rng.randrange(0, 2044, al), opc=0x03))
            elif p < 0.78:
                op = rng.choice(list(ST))
                f3, al = ST[op]
                put(a, enc_s(f3, 3, r2, rng.randrange(0, 2044, al)))
            elif p < 0.90:
                op = rng.choice(list(BR))
                put(a, enc_b(BR[op], r1, r2, 8))       # skip-form
            else:
                put(a, enc_jal(rd, 8))                 # jal +8 (skip)
            a += 4
        put(a, enc_jal(0, 0))                          # park: jal x0, .
    blob = bytearray(MEM_SIZE)
    for addr, w32 in words.items():
        blob[addr:addr + 4] = w32.to_bytes(4, "little")
    return blob


class WarpIss(Iss):
    def __init__(self, blob, hartid):
        super().__init__(mem_base=0, mem_size=MEM_SIZE, reset_pc=0)
        self.load_blob(bytes(blob), 0)
        self._hart = hartid

    def _csr_read(self, a):
        if a == 0xF14:
            return self._hart
        return super()._csr_read(a)


async def run(dut, seed, n_per_warp, target):
    rng = Rng(seed)
    start_clock(dut)
    blob = build_program(rng, n_per_warp)
    mem = bytearray(blob)                              # shared RTL memory
    iss = [WarpIss(blob, w) for w in range(W)]
    retired = [0] * W
    stub_len = 3                                       # stub commits per warp
    target_commits = stub_len + 1 + n_per_warp + 8     # + lui + park laps

    await drive_edge(dut)
    dut.rst_n.value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    dut.imem_rdata.value = 0
    dut.dmem_rdata.value = 0
    # the release cycle's F request fires combinationally NOW — capture it
    # so the very first fetch gets its response next cycle
    i_req = bool(int(dut.imem_req.value))
    i_addr = int(dut.imem_addr.value) % MEM_SIZE
    d_req, d_we, d_be, d_addr, d_wdata = False, False, 0, 0, 0
    await sample_edge(dut)
    total_cycles = (target_commits + 4) * W * 6 + 200
    for cyc in range(total_cycles):
        await drive_edge(dut)
        # present responses for the requests captured LAST cycle (spec §6:
        # req in cycle n -> rdata during cycle n+1)
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
        # capture THIS cycle's combinational requests at the drive point
        # (post-rising-edge reads would see the NEXT cycle's F request)
        i_req = bool(int(dut.imem_req.value))
        i_addr = int(dut.imem_addr.value) % MEM_SIZE
        d_req = bool(int(dut.dmem_req.value))
        d_we = bool(int(dut.dmem_we.value))
        d_be = int(dut.dmem_be.value)
        d_addr = int(dut.dmem_addr.value) % MEM_SIZE
        d_wdata = int(dut.dmem_wdata.value)
        await sample_edge(dut)

        if int(dut.cmt_valid.value):
            w = int(dut.cmt_warp.value)
            exp = None
            while exp is None:                          # ISS traps don't retire
                exp = iss[w].step()
            ctx = f"warp {w} commit {retired[w]}"
            check(int(dut.cmt_pc.value) == exp["pc"],
                  f"pc got {int(dut.cmt_pc.value):08x} want {exp['pc']:08x}",
                  cyc, seed, TB, target, ctx)
            check(int(dut.cmt_instr.value) == exp["instr"],
                  f"instr got {int(dut.cmt_instr.value):08x} want "
                  f"{exp['instr']:08x}", cyc, seed, TB, target, ctx)
            check(int(dut.cmt_rd.value) == exp["rd"],
                  f"rd got {int(dut.cmt_rd.value)} want {exp['rd']}",
                  cyc, seed, TB, target, ctx)
            if exp["rd"]:
                check(int(dut.cmt_wdata.value) == exp["wdata"],
                      f"wdata got {int(dut.cmt_wdata.value):08x} want "
                      f"{exp['wdata']:08x}", cyc, seed, TB, target, ctx)
            if exp["store"] is not None:
                sa, sv, ssz = exp["store"]
                check(int(dut.cmt_st_valid.value) == 1, "store missing",
                      cyc, seed, TB, target, ctx)
                check(int(dut.cmt_st_addr.value) == sa,
                      f"st_addr got {int(dut.cmt_st_addr.value):08x} want "
                      f"{sa:08x}", cyc, seed, TB, target, ctx)
                check(int(dut.cmt_st_data.value) == sv,
                      f"st_data got {int(dut.cmt_st_data.value):08x} want "
                      f"{sv:08x}", cyc, seed, TB, target, ctx)
                check(int(dut.cmt_st_size.value) == ssz,
                      f"st_size got {int(dut.cmt_st_size.value)} want {ssz}",
                      cyc, seed, TB, target, ctx)
            else:
                check(int(dut.cmt_st_valid.value) == 0, "phantom store",
                      cyc, seed, TB, target, ctx)
            retired[w] += 1
        if all(r >= target_commits for r in retired):
            break

    for w in range(W):                                  # positive completion
        check(retired[w] >= target_commits,
              f"warp {w} retired only {retired[w]}/{target_commits}",
              0, seed, TB, target)
    print(f"[{TB}] {sum(retired)} commits across {W} warps, all "
          f"lockstep-exact")


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(60), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(400), "fuzz")
