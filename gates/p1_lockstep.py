#!/usr/bin/env python3
"""P1 gate: RV32I lockstep — RTL vs ISS vs spike, zero commit diffs.

Three commit-log producers, one canonical format (golden/iss.py docstring):
  ISS   : golden/iss.py executing the ELF directly
  spike : `spike -l --log-commits` parsed by golden/spikelog.py
  RTL   : tb/rv32i_core Verilator harness (once built) writing the format natively

Modes:
  --suite            all riscv-tests rv32ui (builds them first)
  --fuzz N --size M  N rvfuzz chunks of M instructions each
  --single ELF       one program
  --nightly          suite + 10 x 100k fuzz (the 1M-instr P1 bar)
  --no-rtl           golden-vs-spike only (pre-RTL bring-up / triage)

Pass bar (P1-DoD): zero diffs, instruction by instruction, all rv32ui + 1M fuzz.
"""
import argparse
import os
import pathlib
import subprocess
import sys
import time

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))

from golden import elf as elfmod          # noqa: E402
from golden import rvfuzz, spikelog       # noqa: E402
from golden.iss import Halt, Iss, fmt_commit  # noqa: E402

# Toolchains default to this checkout and can be shared via
# GPU_TSU_TOOL_ROOT; generated binaries remain local to this checkout.
TOOL_ROOT = pathlib.Path(os.environ.get("GPU_TSU_TOOL_ROOT", MK))
GCC = TOOL_ROOT / "tools/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-gcc"
SPIKE = TOOL_ROOT / "tools/spike/bin/spike"
RTL_SIM = MK / "tb/rv32i_core/obj_dir/Vrv32i_core"
RVTESTS = TOOL_ROOT / "tools/src/riscv-tests"
BUILD = MK / "data/rvtests"
MARCH = "rv32i_zicsr_zifencei"
MEM_BASE, MEM_SIZE = 0x8000_0000, 0x80_0000
SPIKE_ISA = ["--isa=" + MARCH, "--priv=m", f"-m{MEM_BASE:#x}:{MEM_SIZE:#x}"]


def sh(cmd, timeout=120, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout, **kw)


# D-010: ma_data needs hardware misaligned-access support; spike itself fails
# it under --priv=m (verified 2026-07-04, tohost=668-fail), and this core
# traps on misalignment by design, matching spike. Lockstep on it was clean
# (85/85 commits identical) — excluded from the pass bar, not from honesty.
EXCLUDE = {"ma_data"}


def build_suite():
    BUILD.mkdir(parents=True, exist_ok=True)
    elfs = []
    for src in sorted((RVTESTS / "isa/rv32ui").glob("*.S")):
        if src.stem in EXCLUDE:
            continue
        out = BUILD / f"rv32ui-p-{src.stem}"
        r = sh([GCC, f"-march={MARCH}", "-mabi=ilp32", "-static", "-mcmodel=medany",
                "-fvisibility=hidden", "-nostdlib", "-nostartfiles",
                "-I", RVTESTS / "env/p", "-I", RVTESTS / "isa/macros/scalar",
                "-T", RVTESTS / "env/p/link.ld", src, "-o", out])
        if r.returncode:
            print(f"BUILD FAIL {src.stem}:\n{r.stderr[:800]}")
            return None
        elfs.append(out)
    return elfs


def run_iss(elf_path, max_steps):
    entry, segs, syms = elfmod.load(str(elf_path))
    tohost = syms.get("tohost")
    iss = Iss(MEM_BASE, MEM_SIZE, entry, tohost)
    for pa, blob in segs:
        iss.load_blob(blob, pa)
    lines = []
    for _ in range(max_steps):
        try:
            c = iss.step()
        except Halt as h:
            pc = (iss.lane.pc - 4) & 0xFFFFFFFF
            lines.append(fmt_commit({"pc": pc, "instr": iss.read(pc, 4), "rd": 0,
                                     "wdata": 0, "store": (tohost, h.value, 2)}))
            return lines, h.value
        if c is not None:
            lines.append(fmt_commit(c))
    return lines, None


def run_spike(elf_path, timeout=180):
    entry, _, syms = elfmod.load(str(elf_path))
    r = sh([SPIKE, *SPIKE_ISA, "-l", "--log-commits", elf_path], timeout=timeout)
    return spikelog.parse(r.stderr, entry, syms.get("tohost"))


def run_rtl(elf_path, max_cycles, log_path):
    entry, segs, syms = elfmod.load(str(elf_path))
    img = bytearray(MEM_SIZE)
    for pa, blob in segs:
        img[pa - MEM_BASE:pa - MEM_BASE + len(blob)] = blob
    bin_path = pathlib.Path(str(log_path) + ".bin")
    bin_path.write_bytes(img)
    r = sh([RTL_SIM, f"+bin={bin_path}", f"+entry={entry:#x}",
            f"+tohost={syms.get('tohost', 0):#x}", f"+log={log_path}",
            f"+max={max_cycles}"], timeout=600)
    bin_path.unlink(missing_ok=True)
    lines = pathlib.Path(log_path).read_text().splitlines() if pathlib.Path(log_path).exists() else []
    val = None
    if lines and lines[-1].startswith("HALT"):
        val = int(lines.pop().split()[1], 16)
    return lines, val, r


def diff(name, a_name, a, b_name, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            print(f"DIFF {name} at commit {i}:")
            for j in range(max(0, i - 3), i + 1):
                print(f"  {a_name}[{j}] {a[j]}")
                print(f"  {b_name}[{j}] {b[j]}")
            return False
    if len(a) != len(b):
        print(f"DIFF {name}: length {a_name}={len(a)} {b_name}={len(b)}")
        tail_src, tail = (a_name, a) if len(a) > len(b) else (b_name, b)
        for j in range(n, min(n + 3, len(tail))):
            print(f"  extra {tail_src}[{j}] {tail[j]}")
        return False
    return True


def check_program(elf_path, name, max_steps, use_rtl, scratch):
    iss_lines, iss_val = run_iss(elf_path, max_steps)
    spk_lines, spk_val = run_spike(elf_path)
    ok = diff(name, "iss", iss_lines, "spike", spk_lines)
    if iss_val != spk_val:
        print(f"DIFF {name}: tohost iss={iss_val} spike={spk_val}")
        ok = False
    if ok and use_rtl:
        rtl_lines, rtl_val, r = run_rtl(elf_path, max_steps * 8 + 20000,
                                        scratch / f"{name}.rtlog")
        ok = diff(name, "rtl", rtl_lines, "iss", iss_lines)
        if rtl_val != iss_val:
            print(f"DIFF {name}: tohost rtl={rtl_val} iss={iss_val} (rtl rc={r.returncode})")
            ok = False
    if ok and iss_val != 1:
        print(f"FAIL {name}: test reported failure tohost={iss_val}")
        ok = False
    print(f"{'PASS' if ok else 'FAIL'} {name} ({len(iss_lines)} commits)")
    return ok


def fuzz_chunk(seed, size, use_rtl, scratch):
    src = scratch / f"fuzz_{seed:x}.S"
    elf = scratch / f"fuzz_{seed:x}.elf"
    with open(src, "w") as f:
        rvfuzz.gen(seed, size, f)
    r = sh([GCC, f"-march={MARCH}", "-mabi=ilp32", "-nostdlib", "-nostartfiles",
            "-T", MK / "sw/rv/link.ld", src, "-o", elf])
    if r.returncode:
        print(f"FUZZ BUILD FAIL seed={seed:#x}:\n{r.stderr[:500]}")
        return False
    ok = check_program(elf, f"fuzz[{seed:#x},{size}]", size * 3 + 1000, use_rtl, scratch)
    if ok:
        src.unlink()
        elf.unlink()
    else:
        print(f"repro: python gates/p1_lockstep.py --fuzz 1 --size {size} --seed {seed:#x}"
              + ("" if use_rtl else " --no-rtl"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--single", type=str, default=None)
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--no-rtl", action="store_true")
    a = ap.parse_args()
    use_rtl = not a.no_rtl
    if use_rtl and not RTL_SIM.exists():
        print(f"RTL sim missing ({RTL_SIM}) — build tb/rv32i_core first"
              " or pass --no-rtl")
        return 1
    scratch = MK / "ci/logs/p1_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    ok = True
    if a.nightly:
        a.suite = True
        a.fuzz, a.size = 10, 100_000
    if a.single:
        ok &= check_program(pathlib.Path(a.single), pathlib.Path(a.single).name,
                            5_000_000, use_rtl, scratch)
    if a.suite:
        elfs = build_suite()
        if elfs is None:
            return 1
        for e in elfs:
            ok &= check_program(e, e.name, 2_000_000, use_rtl, scratch)
    if a.fuzz:
        base = a.seed if a.seed is not None else int(time.time())
        for i in range(a.fuzz):
            ok &= fuzz_chunk(base + i if a.fuzz > 1 or a.seed is None else base,
                             a.size, use_rtl, scratch)
    print("P1 LOCKSTEP:", "GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
