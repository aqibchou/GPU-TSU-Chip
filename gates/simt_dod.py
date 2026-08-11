#!/usr/bin/env python3
"""SIMT-DoD gate (M15) — the barrel core's definition-of-done battery.

════════ FROZEN DECISION RULES (R3, set 2026-07-07 before the suite ran;
only rv32ui-p-add had been executed as the bench smoke) ════
1. riscv-tests: every rv32ui ELF under data/rvtests (ma_data excluded per
   D-010, same as P1) runs on the 8-warp barrel: warp 0 must reach
   tohost=1; warps 1-7 must provably park (>= 200 lockstep-exact commits
   each); every retirement on every warp diffs exactly against that
   warp's ISS. Positive completion: the bench's PASS line parsed per ELF.
2. barrel fuzz: tb/barrel_core fuzz (per-warp random programs, disjoint
   data pages) green at seed 0xC0FFEE, >= 1M cumulative cycles when run
   at nightly scale (FUZZ_N scaled); the nightly stage re-proves it.
3. matmul-on-lanes: the INT8 8x8x8 kernel (shift-add mul, lane = output
   column, lane-divergent B/C addresses through the lane-serial memory
   unit) on simt_core: all 64 outputs per warp exact vs the integer
   reference AND full lockstep vs golden simt_iss throughout.
COMPOSITION NOTE (documented v1 architecture decision): rule 1 (ISA
compliance) runs on the barrel scalar configuration — the riscv-tests
hart-park idiom (bnez mhartid, self) is a divergent BACKWARD branch,
inherently thread-parallel and inexpressible in any SIMT reconvergence
scheme (real GPUs cannot run it either). Rules 2-3 run on simt_core with
lanes. Both cores share the scheduler, regfile discipline, and the
invariant catalog.
Evidence: ci/logs/simt_dod/.
════════════════════════════════════════════════════════════════════════════
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

MK = pathlib.Path(__file__).resolve().parent.parent
RES = MK / "ci/logs/simt_dod"
EXCLUDE = {"rv32ui-p-ma_data"}          # D-010


def run_suite():
    elfs = sorted(p for p in (MK / "data/rvtests").iterdir()
                  if p.name not in EXCLUDE and not p.name.endswith(".dump"))
    results = {}
    for e in elfs:
        env = {**os.environ, "MK_ELF": str(e),
               "COCOTB_TEST_MODULES": "test_barrel_riscv",
               "MODULE": "test_barrel_riscv"}
        r = subprocess.run(["make", "-s", "-C", MK / "tb/barrel_riscv",
                            "sim"], capture_output=True, text=True,
                           env=env, timeout=1800)
        ok = r.returncode == 0 and \
            re.search(rf"{re.escape(e.name)}: warp0 PASS", r.stdout)
        results[e.name] = bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {e.name}", flush=True)
        if not ok:
            (RES / f"fail_{e.name}.log").write_text(r.stdout[-4000:])
    return results


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = run_suite()
    n_pass = sum(results.values())
    rule1 = n_pass == len(results) and len(results) >= 38
    # rule 3: matmul on the lane core
    env = {**os.environ, "COCOTB_TEST_MODULES": "test_simt_core",
           "MODULE": "test_simt_core", "COCOTB_TEST_FILTER": "matmul"}
    r3 = subprocess.run(["make", "-s", "-C", MK / "tb/simt_core", "sim"],
                        capture_output=True, text=True, env=env,
                        timeout=3600)
    rule3 = r3.returncode == 0 and \
        re.search(r"matmul: \d+ commits.*EXACT", r3.stdout) is not None
    report = {"rule1_riscv": {"pass": rule1, "n_pass": n_pass,
                              "n_total": len(results),
                              "fails": [k for k, v in results.items()
                                        if not v]},
              "rule2_fuzz": "nightly stages (barrel_core + simt_core fuzz)",
              "rule3_matmul": {"pass": bool(rule3)},
              "wall_s": round(time.time() - t0, 1)}
    report["verdict"] = "PASS" if (rule1 and rule3) else "FAIL"
    with open(RES / "simt_dod.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in
                      ("rule1_riscv", "rule3_matmul", "verdict")}))
    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
