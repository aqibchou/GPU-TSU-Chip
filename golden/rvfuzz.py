"""rvfuzz — emit a random legal RV32I program as GAS assembly.

Design (M2): straight-line torture, no loops, every path bounded:
- all registers zeroed first; x3 is reserved as the data-region base
- ops: ALU imm/reg (incl. all shifts), lui/auipc, aligned loads/stores of
  every width, and *skip-form* control flow — branches/jal/jalr whose taken
  target is exactly pc+8, so both outcomes are legal and loop-free
- letting GAS encode from mnemonics keeps my encoder out of the trust chain:
  the ISS/RTL decode gets tested against binutils' encoder, and spike runs
  the identical bytes
- no CSR instructions here: the riscv-tests suite covers the CSR/trap paths

Usage: python rvfuzz.py SEED N > prog.S   (build with sw/rv/link.ld)
"""
import random
import sys

REGS = [r for r in range(1, 32) if r != 3]          # x3 = data base, reserved
BRANCHES = ["beq", "bne", "blt", "bge", "bltu", "bgeu"]
ALU_R = ["add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"]
ALU_I = ["addi", "slti", "sltiu", "xori", "ori", "andi"]
DATA_BYTES = 4096
MAX_OFF = 2044   # S/I-type immediates are signed 12-bit: offsets stay < 2048


def gen(seed: int, n: int, out) -> None:
    rng = random.Random(seed)
    w = out.write
    w(".globl _start\n_start:\n")
    for r in range(1, 32):
        w(f"  li x{r}, 0\n")
    w("  la x3, fuzzdata\n")

    def rd():
        return f"x{rng.choice(REGS) if rng.random() > 0.05 else 0}"

    def rs():
        return f"x{rng.choice(REGS + [0, 3])}"

    for _ in range(n):
        p = rng.random()
        if p < 0.40:                                   # ALU immediate
            op = rng.choice(ALU_I)
            w(f"  {op} {rd()}, {rs()}, {rng.randint(-2048, 2047)}\n")
        elif p < 0.55:                                 # ALU register
            w(f"  {rng.choice(ALU_R)} {rd()}, {rs()}, {rs()}\n")
        elif p < 0.62:                                 # shifts by immediate
            op = rng.choice(["slli", "srli", "srai"])
            w(f"  {op} {rd()}, {rs()}, {rng.randint(0, 31)}\n")
        elif p < 0.70:                                 # load
            op, align = rng.choice([("lw", 4), ("lh", 2), ("lhu", 2),
                                    ("lb", 1), ("lbu", 1)])
            off = rng.randrange(0, MAX_OFF, align)
            w(f"  {op} {rd()}, {off}(x3)\n")
        elif p < 0.78:                                 # store
            op, align = rng.choice([("sw", 4), ("sh", 2), ("sb", 1)])
            off = rng.randrange(0, MAX_OFF, align)
            w(f"  {op} {rs()}, {off}(x3)\n")
        elif p < 0.86:                                 # branch skipping one op
            w(f"  {rng.choice(BRANCHES)} {rs()}, {rs()}, . + 8\n")
            w(f"  addi {rd()}, {rs()}, {rng.randint(-2048, 2047)}\n")
        elif p < 0.90:                                 # jal skipping one op
            w(f"  jal {rd()}, . + 8\n")
            w(f"  xori {rd()}, {rs()}, {rng.randint(-2048, 2047)}\n")
        elif p < 0.94:                                 # jalr skipping one op
            t = f"x{rng.choice(REGS)}"
            w(f"  auipc {t}, 0\n")
            w(f"  jalr {rd()}, {t}, 12\n")
            w(f"  ori {rd()}, {rs()}, {rng.randint(-2048, 2047)}\n")
        elif p < 0.97:                                 # lui
            w(f"  lui {rd()}, {rng.randint(0, 0xFFFFF)}\n")
        else:                                          # auipc
            w(f"  auipc {rd()}, {rng.randint(0, 0xFFFFF)}\n")

    w("  la x5, tohost\n  li x6, 1\n  sw x6, 0(x5)\n1:\n  j 1b\n")
    w('\n.section .tohost, "aw", @progbits\n')
    w(".globl tohost\ntohost: .word 0\n.word 0\n")
    w(".globl fromhost\nfromhost: .word 0\n.word 0\n")
    w(f"\n.data\nfuzzdata: .space {DATA_BYTES}\n")


if __name__ == "__main__":
    gen(int(sys.argv[1], 0), int(sys.argv[2], 0), sys.stdout)
