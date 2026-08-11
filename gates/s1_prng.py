#!/usr/bin/env python3
"""S1 gate: PRNG farm quality + cross-stream independence (D16, Σ.4c).

════════════════ FROZEN DECISION RULES (R3 — set before data collection,
2026-07-04; changes require a written argument here + a devlog note) ═══════
Stage A — equivalence: cocotb fuzz green AND the C++ dump path's first 8
  words of all 16 streams match golden exactly (also proves distinctness).
Stage B — PractRand, per config in {s0 single-stream, il16 interleaved-16}:
  RNG_test stdin32 to -tlmax 64GB (= 2^36 bytes in PractRand units).
  COMPLETION IS POSITIVE EVIDENCE: the log must contain the final
  "(2^36 bytes)" checkpoint — absence of failure lines alone proves nothing
  (first run of this gate was "passed" by an argv typo that made RNG_test
  exit instantly; never again). Then: HARD FAIL on any line containing
  "FAIL" or "SUSPICIOUS" (any case, covers "very/mildly suspicious").
  Lines containing "unusual": at most 2 per run — PractRand flags ~p<1e-3
  events as 'unusual' and a 64 GiB run evaluates thousands of subtests, so
  0-2 by chance is expected. This is the pre-registered reading of the
  unified doc's "zero anomalies" (D-011).
Stage C — TestU01 (secondary battery replacing dead dieharder, D-001):
  SmallCrush on BOTH configs and full Crush on s0.
  AMENDED 2026-07-04 (written argument, R3): the original rule — the literal
  "All tests were passed" string — false-alarms ~25% of CLEAN Crush runs:
  TestU01 flags any p outside [1e-3, 0.999], so a 144-statistic Crush expects
  ~0.29 chance flags per run (SmallCrush, 15 stats: ~0.03). Amended rule:
  * EXTREME p (< 1e-10 or > 1-1e-10, incl. eps/eps1 markers) -> FAIL, always.
  * borderline flags (listed but not extreme): allowed up to 1 (SmallCrush)
    or 2 (Crush) per run, BUT any borderline triggers one RERUN of the same
    battery on a different jump-spaced stream; the rerun must be clean by the
    same rule AND must not re-flag any test name from the first run.
    Independent replication refuting a marginal p is L'Ecuyer's own practice.
  First Crush run (stream 0): 1 borderline, WeightDistrib r=0 at p=0.9995 —
  triggered the rerun path (stream 1).
Stage D — cross-stream battery (golden/prng_check.py, its own frozen
  Bonferroni family-alpha 0.01): (a) 64 golden streams x 2^20 words;
  (b) RTL il16 dump, 512 MiB, de-interleaved to 16 x 8M words.
S1 = A ∧ B ∧ C ∧ D. Evidence: ci/logs/s1/*.json + raw logs.
════════════════════════════════════════════════════════════════════════════

Stages run independently (long ones in background shells):
  --stage {equiv,pr-s0,pr-il16,sc-s0,sc-il16,crush-s0,bat-gold,bat-rtl}
  --verdict     read all stage JSONs, apply the rules, print S1 GREEN/RED
  --quick       nightly regression: equiv + 1 GiB PractRand(s0) + golden
                battery 16 x 2^18 (a drift alarm, not the gate)
"""
import argparse
import json
import pathlib
import struct
import subprocess
import sys
import time

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))

import numpy as np                      # noqa: E402

from golden import prng_check           # noqa: E402
from golden.xoshiro import FarmVec, Xoshiro128pp, stream_states  # noqa: E402

RES = MK / "ci/logs/s1"
DUMP = MK / "tb/pbit_prng/dump/prng_dump"
RNG_TEST = MK / "tools/bin/RNG_test"
TU01 = MK / "tools/bin/tu01_stdin"
SEED = 0xC0FFEE          # frozen project seed for S1 streams
NSTREAMS = 16
TLMAX_FULL = ("64GB", "(2^36 bytes)")    # arg, required completion marker
TLMAX_QUICK = ("1GB", "(2^30 bytes)")


def seeds_file() -> pathlib.Path:
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / "seeds.bin"
    states = stream_states(SEED, NSTREAMS)
    p.write_bytes(b"".join(struct.pack("<4I", *s) for s in states))
    return p


def save(stage: str, ok: bool, detail: dict):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{stage}.json").write_text(json.dumps(
        {"stage": stage, "ok": ok, "when": time.strftime("%F %T"), **detail}, indent=1))
    print(f"[{stage}] {'PASS' if ok else 'FAIL'} {detail}")
    return ok


def stage_equiv() -> bool:
    r = subprocess.run(["make", "-s", "-C", MK / "tb/pbit_prng", "fuzz",
                        "MK_FUZZ_SEED=0xC0FFEE"], capture_output=True, text=True)
    cocotb_ok = r.returncode == 0
    sf = seeds_file()
    rr = subprocess.run([DUMP, f"+seeds={sf}", "+selftest"],
                        capture_output=True, text=True)
    states = stream_states(SEED, NSTREAMS)
    gold = {}
    for i, s in enumerate(states):
        g = Xoshiro128pp(list(s))
        gold[i] = [g.next() for _ in range(8)]
    head_ok, firsts = True, set()
    for ln in rr.stdout.strip().splitlines():
        i, w, v = ln.split()
        if gold[int(i)][int(w)] != int(v, 16):
            head_ok = False
        if int(w) == 0:
            firsts.add(v)
    distinct_ok = len(firsts) == NSTREAMS
    return save("equiv", cocotb_ok and head_ok and distinct_ok,
                {"cocotb": cocotb_ok, "head": head_ok, "distinct": distinct_ok})


def practrand(tag: str, mode: str, stream: int, tlmax: tuple) -> bool:
    arg, marker = tlmax
    sf = seeds_file()
    log = RES / f"practrand_{tag}.log"
    with open(log, "w") as lf:
        p_dump = subprocess.Popen(
            [DUMP, f"+seeds={sf}", f"+mode={mode}", f"+stream={stream}", "+bytes=0"],
            stdout=subprocess.PIPE)
        p_test = subprocess.Popen([RNG_TEST, "stdin32", "-tlmax", arg,
                                   "-multithreaded"],
                                  stdin=p_dump.stdout, stdout=lf,
                                  stderr=subprocess.STDOUT)
        p_dump.stdout.close()
        p_test.wait()
        p_dump.wait()
    text = log.read_text()
    completed = marker in text          # positive evidence, never absence
    hard = [ln for ln in text.splitlines()
            if "FAIL" in ln.upper() or "SUSPICIOUS" in ln.upper()]
    unusual = [ln for ln in text.splitlines() if "unusual" in ln]
    ok = completed and p_test.returncode == 0 and not hard and len(unusual) <= 2
    return save(f"pr-{tag}", ok, {"completed": completed, "hard": len(hard),
                                  "unusual": len(unusual),
                                  "rc": p_test.returncode, "tlmax": arg})


def _tu01_run(mode: str, stream: int, battery: str, log: pathlib.Path):
    sf = seeds_file()
    with open(log, "w") as lf:
        p_dump = subprocess.Popen(
            [DUMP, f"+seeds={sf}", f"+mode={mode}", f"+stream={stream}", "+bytes=0"],
            stdout=subprocess.PIPE)
        p_test = subprocess.Popen([TU01, battery], stdin=p_dump.stdout,
                                  stdout=lf, stderr=subprocess.STDOUT)
        p_dump.stdout.close()
        p_test.wait()
        p_dump.wait()
    return _tu01_flags(log.read_text())


def _tu01_flags(text: str):
    """Returns (clean_string, extreme[names], borderline[names]) from the
    battery summary. Extreme = eps/eps1 markers or p outside [1e-10, 1-1e-10]."""
    import re
    extreme, borderline = [], []
    in_tbl = False
    for ln in text.splitlines():
        if "p-value" in ln and "Test" in ln:
            in_tbl = True
            continue
        if in_tbl:
            if ln.strip().startswith("----") or not ln.strip():
                continue
            if "All other tests were passed" in ln or "=====" in ln:
                break
            m = re.match(r"\s*\d+\s+(.+?)\s+([0-9.eE+-]+|eps1?|1 - eps1)\s*$", ln)
            if m:
                name, pv = m.group(1).strip(), m.group(2)
                if pv in ("eps", "eps1", "1 - eps1"):
                    extreme.append(name)
                else:
                    p = float(pv)
                    (extreme if (p < 1e-10 or p > 1 - 1e-10) else borderline).append(name)
    return ("All tests were passed" in text), extreme, borderline


def testu01(tag: str, mode: str, stream: int, battery: str) -> bool:
    allowed = {"small": 1, "crush": 2}[battery]
    log = RES / f"tu01_{battery}_{tag}.log"
    clean, extreme, borderline = _tu01_run(mode, stream, battery, log)
    if clean:
        return save(f"{battery}-{tag}", True, {"log": log.name, "flags": 0})
    if extreme or len(borderline) > allowed:
        return save(f"{battery}-{tag}", False,
                    {"log": log.name, "extreme": extreme, "borderline": borderline})
    # borderline within allowance -> one rerun on a different stream
    log2 = RES / f"tu01_{battery}_{tag}_rerun.log"
    alt = (stream + 1) % NSTREAMS
    clean2, ext2, bord2 = _tu01_run(mode, alt, battery, log2)
    repeat = set(borderline) & set(bord2)
    ok = (not ext2) and len(bord2) <= allowed and not repeat and not extreme
    return save(f"{battery}-{tag}", ok,
                {"log": log.name, "rerun": log2.name, "first": borderline,
                 "rerun_flags": bord2, "repeated": sorted(repeat)})


def battery_golden(nstreams=64, nwords=1 << 20, tag="bat-gold") -> bool:
    fv = FarmVec(stream_states(SEED, nstreams))
    r = prng_check.battery(fv.next_block(nwords))
    return save(tag, r["ok"], {"n_stats": r["n_stats"], "worst": r["worst"],
                               "n_fail": len(r["failures"])})


def battery_rtl(mib=512) -> bool:
    sf = seeds_file()
    nbytes = mib << 20
    raw = subprocess.run([DUMP, f"+seeds={sf}", "+mode=interleave",
                          f"+bytes={nbytes}"], capture_output=True).stdout
    w = np.frombuffer(raw, dtype="<u4")
    w = w[: (len(w) // NSTREAMS) * NSTREAMS].reshape(-1, NSTREAMS).T.copy()
    r = prng_check.battery(w)
    return save("bat-rtl", r["ok"], {"n_stats": r["n_stats"], "worst": r["worst"],
                                     "n_fail": len(r["failures"]),
                                     "words_per_stream": int(w.shape[1])})


REQUIRED = ["equiv", "pr-s0", "pr-il16", "small-s0", "small-il16",
            "crush-s0", "bat-gold", "bat-rtl"]


def verdict() -> int:
    missing, red = [], []
    for s in REQUIRED:
        p = RES / f"{s}.json"
        if not p.exists():
            missing.append(s)
        elif not json.loads(p.read_text())["ok"]:
            red.append(s)
    if missing or red:
        print(f"S1: NOT GREEN — missing={missing} red={red}")
        return 1
    print("S1 GREEN — all frozen stages pass (evidence: ci/logs/s1/)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.verdict:
        return verdict()
    if a.quick:
        ok = stage_equiv()
        ok &= practrand("quick", "single", 0, TLMAX_QUICK)
        ok &= battery_golden(16, 1 << 18, tag="bat-quick")
        print("S1-QUICK:", "GREEN" if ok else "RED")
        return 0 if ok else 1
    st = a.stage
    ok = {
        "equiv":    lambda: stage_equiv(),
        "pr-s0":    lambda: practrand("s0", "single", 0, TLMAX_FULL),
        "pr-il16":  lambda: practrand("il16", "interleave", 0, TLMAX_FULL),
        "sc-s0":    lambda: testu01("s0", "single", 0, "small"),
        "sc-il16":  lambda: testu01("il16", "interleave", 0, "small"),
        "crush-s0": lambda: testu01("s0", "single", 0, "crush"),
        "bat-gold": lambda: battery_golden(),
        "bat-rtl":  lambda: battery_rtl(),
    }[st]()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
