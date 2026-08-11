#!/usr/bin/env python3
"""M4 gate: single p-bit cell — LUT accuracy + bit-true equivalence +
Bernoulli rates through the REAL hardware randomness path.

════════ FROZEN DECISION RULES (R3, set 2026-07-04 before data collection) ══
1. LUT accuracy: max |p17/65536 − σ_ideal| ≤ 7.0e-4 over the full reachable
   (acc, beta) grid (golden/pbit.py lut_accuracy_report), and the checked-in
   sigmoid_lut.mem must equal the golden generator's output byte-for-byte.
2. Equivalence: tb/pbit_cell cocotb fuzz green at seed 0xC0FFEE (exact
   per-update diff of s_out AND p17 across random mode/bias/degree/beta/rnd).
3. Bernoulli sweep on pbit_mc_top (xoshiro stream -> cell, the true HW path):
   grid = beta_raw {16, 64, 128, 224} x bias_raw {-48,-24,-12,-4,-1,0,
   +1,+4,+12,+24,+48}, n = 200,000 samples per config, each config on its own
   jump-spaced stream. Test: exact two-sided binomial test of ones against
   the BIT-TRUE p = p17(bias,beta)/65536 (not the ideal sigmoid — accuracy is
   rule 1's job). Family-wise alpha = 0.01, Bonferroni over the 44 configs.
Pass = 1 ∧ 2 ∧ 3. Evidence: ci/logs/m4/.
════════════════════════════════════════════════════════════════════════════
"""
import json
import pathlib
import subprocess
import sys
import time

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))

from scipy.stats import binomtest                      # noqa: E402

from golden.pbit import lut_accuracy_report, lut_table, p17  # noqa: E402
from golden.xoshiro import stream_states               # noqa: E402

RES = MK / "ci/logs/m4"
TB = MK / "tb/pbit_cell"
MC = TB / "mc/pbit_mc"
BETAS = [16, 64, 128, 224]
BIASES = [-48, -24, -12, -4, -1, 0, 1, 4, 12, 24, 48]
NSAMP = 200_000
ALPHA = 0.01
SEED_M4 = 0xC0FFEE


def build_mc() -> bool:
    r = subprocess.run(
        ["verilator", "--cc", MK / "rtl/pbit/pbit_mc_top.sv",
         MK / "rtl/pbit/pbit_cell.sv", MK / "rtl/pbit/xoshiro128pp.sv",
         "--exe", TB / "cell_mc.cpp", "--build", "-j", "4",
         "--top-module", "pbit_mc_top", "-O3", "-CFLAGS", "-O2", "-Wall",
         "--Mdir", TB / "mc", "-o", "pbit_mc",
         "-GLUT_FILE=" + f'"{MK}/rtl/pbit/sigmoid_lut.mem"'],
        capture_output=True, text=True, cwd=TB)
    if r.returncode:
        print(r.stderr[-1500:])
    return r.returncode == 0


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    ok = True

    # rule 1 — LUT accuracy + mem sync
    rep = lut_accuracy_report()
    mem = (MK / "rtl/pbit/sigmoid_lut.mem").read_text().splitlines()
    mem_vals = [int(x, 16) for x in mem if not x.startswith("//")]
    sync = mem_vals == [int(v) for v in lut_table()]
    print(f"[lut] max_err={rep['max_err']:.2e} bound={rep['bound']:.1e} "
          f"mem_sync={sync}")
    ok &= rep["ok"] and sync

    # rule 2 — exact equivalence fuzz
    r = subprocess.run(["make", "-s", "-C", TB, "fuzz", "MK_FUZZ_SEED=0xC0FFEE"],
                       capture_output=True, text=True)
    print(f"[equiv] cocotb fuzz {'PASS' if r.returncode == 0 else 'FAIL'}")
    ok &= r.returncode == 0

    # rule 3 — Bernoulli sweep on the real randomness path
    if not build_mc():
        print("[mc] build FAILED")
        ok = False
    else:
        cfgs = [(b, h) for b in BETAS for h in BIASES]
        streams = stream_states(SEED_M4, len(cfgs))
        thr = ALPHA / len(cfgs)
        worst = (1.0, None)
        for idx, (beta, bias) in enumerate(cfgs):
            st = streams[idx]
            r = subprocess.run(
                [MC, f"+s0={st[0]}", f"+s1={st[1]}", f"+s2={st[2]}",
                 f"+s3={st[3]}", f"+bias={bias}", f"+beta={beta}",
                 f"+n={NSAMP}"], capture_output=True, text=True)
            ones = int(r.stdout.split("ones=")[1].split()[0])
            p_true = p17(bias, beta) / 65536.0
            pv = binomtest(ones, NSAMP, p_true).pvalue if 0.0 < p_true < 1.0 \
                else (1.0 if ones == NSAMP * p_true else 0.0)
            if pv < worst[0]:
                worst = (pv, (beta, bias, ones))
            if pv < thr:
                print(f"[mc] FAIL beta={beta} bias={bias}: ones={ones} "
                      f"p_true={p_true:.6f} pval={pv:.2e} < {thr:.2e}")
                ok = False
        print(f"[mc] {len(cfgs)} configs x {NSAMP} samples; worst pval "
              f"{worst[0]:.3e} at {worst[1]} (threshold {thr:.2e})")

    (RES / "m4.json").write_text(json.dumps(
        {"ok": ok, "when": time.strftime("%F %T"), "lut_max_err": rep["max_err"],
         "mem_sync": sync, "worst_pval": worst[0] if 'worst' in dir() else None},
        indent=1))
    print("M4 CELL GATE:", "GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
