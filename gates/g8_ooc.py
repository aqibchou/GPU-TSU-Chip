#!/usr/bin/env python3
"""G8σ gate (M20) — parse the OOC reports in ci/logs/g8/ against the
frozen bars in docs/FPGA_IMPLEMENTATION.md#timing-and-resource-requirements:

G8.1 post-route WNS >= 0 at the 8.000 ns constraint, EVERY module
G8.2 resource sums fit xck26 with >= 20% headroom on every class
G8.3 check_timing clean (no_clock / constant_clock / unconstrained)

Evidence in: <module>_route.rpt, <module>_util.rpt, <module>_check.rpt
(pulled from the DP-2 host's ~/g8_out). Verdict: ci/logs/g8/g8_report.json.
"""
import json
import pathlib
import re
import sys

MK = pathlib.Path(__file__).resolve().parent.parent
G8 = MK / "ci/logs/g8"

MODULES = ["simt_core", "barrel_sched", "simt_regfile", "tensor_sidecar",
           "port_arbiter", "dcache", "fabric_grid", "prng_farm",
           "s_cluster"]
DEVICE = {"lut": 117_120, "ff": 234_240, "bram": 144, "dsp": 1_248}
HEADROOM = 0.20

CHECKS = ("no_clock", "constant_clock", "unconstrained_internal_endpoints")


def wns(mod):
    p = G8 / f"{mod}_route.rpt"
    if not p.exists():
        return None
    txt = p.read_text()
    m = re.search(r"WNS\(ns\).*?\n\s*-[-\s]*\n\s*(-?[\d.]+)", txt, re.S)
    return float(m.group(1)) if m else None


def util(mod):
    p = G8 / f"{mod}_util.rpt"
    if not p.exists():
        return None
    txt = p.read_text()

    def grab(pat):
        m = re.search(rf"\|\s*{pat}\s*\|\s*(\d+)", txt)
        return int(m.group(1)) if m else 0
    return {"lut": grab("CLB LUTs"), "ff": grab("CLB Registers"),
            "bram": grab("Block RAM Tile"), "dsp": grab("DSPs")}


def check_clean(mod):
    p = G8 / f"{mod}_check.rpt"
    if not p.exists():
        return None
    txt = p.read_text()
    bad = []
    for c in CHECKS:
        m = re.search(rf"checking {c} \((\d+)\)", txt)
        if m and int(m.group(1)) != 0:
            bad.append(f"{c}={m.group(1)}")
    return bad


def main():
    rep = {"modules": {}, "missing": []}
    sums = {k: 0 for k in DEVICE}
    g81 = g83 = True
    for mod in MODULES:
        w, u, ck = wns(mod), util(mod), check_clean(mod)
        if u is None or ck is None:
            rep["missing"].append(mod)
            g81 = False
            continue
        # no reg-to-reg max paths (I/O-bounded module): vacuous is a
        # PASS iff check_timing is clean (the D-022 rule)
        vac = w is None
        rep["modules"][mod] = {"wns": w, "vacuous": vac, **u,
                               "check": ck or "clean"}
        if (w is not None and w < 0) or (vac and ck):
            g81 = False
        if ck:
            g83 = False
        for k in sums:
            sums[k] += u[k]
    caps = {k: int(v * (1 - HEADROOM)) for k, v in DEVICE.items()}
    g82 = all(sums[k] <= caps[k] for k in sums)
    rep["sums"] = sums
    rep["caps_after_headroom"] = caps
    rep["G8.1"] = g81
    rep["G8.2"] = g82
    rep["G8.3"] = g83
    rep["pass"] = g81 and g82 and g83
    (G8 / "g8_report.json").write_text(json.dumps(rep, indent=1))
    for mod, r in rep["modules"].items():
        wtxt = "vacuous" if r["vacuous"] else f"{r['wns']:+7.3f}"
        print(f"  {mod:15s} WNS {wtxt:>8s}  LUT {r['lut']:6d}  "
              f"FF {r['ff']:6d}  BRAM {r['bram']:3d}  DSP {r['dsp']:4d}  "
              f"{r['check']}")
    if rep["missing"]:
        print(f"  MISSING reports: {rep['missing']}")
    print(f"  sums: {sums}  caps(-20%): {caps}")
    for b in ("G8.1", "G8.2", "G8.3"):
        print(f"  {b}: {'PASS' if rep[b] else 'FAIL'}")
    print(f"G8σ: {'ALL PASS' if rep['pass'] else 'RED'} -> "
          f"{G8 / 'g8_report.json'}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
