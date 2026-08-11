#!/usr/bin/env python3
"""FASTPATH leg A measurement pilot (registered in
docs/FPGA_IMPLEMENTATION.md#fast-path-optimizations, 2026-07-15). A mechanism-selection measurement,
NOT a bar: quantify ready-set occupancy and warp0 issue duty on frozen
workloads, classify what "ready" warps are actually doing (done-park
spin vs engine-poll vs work) from their PCs, and compute the predicted
warp0 duty under each candidate mechanism:

  - compaction only (rotate the ready set — leg A's registered RTL)
  - + engine-wait parking (T_STATUS pollers park while the engine is
    busy — new spec surface)
  - + done-halt (harts park after the DONE store — new spec surface)

each capped by the interlock issue period (5 = in_pipe clears at
WB-exit; 4 = WB-entry). Mixed-workload predictions use a first-order
mean-field estimate (documented); the decision-driving scalar
workload is exact (all non-w0 warps sit in one class).

Instrumentation: the harness SCHEDCLR/SCHEDCLASS/SCHED sampler over
verilator public-read taps (busy, phase, pc) — logic-free, off unless
armed. PC classes come from the kernel's own disassembly: class 1 =
crt0 self-jump parks, class 2 = engine-poll csrr sites (hull, may
over-cover a few rarely-executed wrapper instructions — disclosed).

Evidence: ci/logs/fastpath/a_pilot.json.
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

import mkcuda                                                    # noqa: E402
from mkcuda import Runtime                                       # noqa: E402

from golden.gibbs_grid import Graph                              # noqa: E402
from golden.sampling_isa import (SF_IMM, SF_RECORD,              # noqa: E402
                                 SF_STATS_RESET, build_image,
                                 image_from_graph)

LATS = (1, 4)
N = 1024


def rnz(rng, m=16):
    v = 0
    while v == 0:
        v = int(rng.integers(-m, m + 1))
    return v


def torus(nx, ny, rng):
    g = Graph(nx * ny)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for j in (y * nx + (x + 1) % nx, ((y + 1) % ny) * nx + x):
                g.add_edge(i, j, rnz(rng))
    return g


def pc_classes(dis_path):
    """(park_ranges, poll_hull) from the disassembly: self-jumps and
    csrr-0x8c9 sites."""
    text = pathlib.Path(dis_path).read_text()
    parks, polls = [], []
    for m in re.finditer(r"^\s*([0-9a-f]+):\s+\S+\s+j\s+([0-9a-f]+)\b",
                         text, re.M):
        addr, tgt = int(m.group(1), 16), int(m.group(2), 16)
        if addr == tgt:
            parks.append(addr)
    for m in re.finditer(
            r"^\s*([0-9a-f]+):\s+\S+\s+csrr\s+\S+,\s*(?:0x8c9|2249)",
            text, re.M):
        polls.append(int(m.group(1), 16))
    park_ranges = [(a, a + 4) for a in parks[:2]]
    poll_hull = (min(polls), max(polls) + 16) if polls else None
    return park_ranges, poll_hull


def arm(rt, kern):
    """Set PC classes for this kernel and clear the sampler."""
    parks, poll = pc_classes(kern.disasm_path)
    for i in range(4):
        rt._cmd(f"SCHEDCLASS {i} 0 0")
    for i, (lo, hi) in enumerate(parks[:2]):
        rt._cmd(f"SCHEDCLASS {i} {lo:x} {hi:x}")
    if poll:
        rt._cmd(f"SCHEDCLASS 2 {poll[0]:x} {poll[1]:x}")
    assert rt._cmd("SCHEDCLR") == "OK"
    return {"parks": parks, "poll": poll}


def collect(rt):
    toks = rt._cmd("SCHED").split()
    assert toks[0] == "SCHED"
    cyc = int(toks[2])
    hist = [int(x) for x in toks[4:13]]
    w0k = [int(x) for x in toks[14:23]]
    warps = {}
    i = 23
    while i < len(toks):
        w = int(toks[i][1:])
        warps[w] = {"rdy": int(toks[i + 1]), "iss": int(toks[i + 2]),
                    "cls": [int(x) for x in toks[i + 3:i + 8]]}
        i += 8
    return {"cyc": cyc, "hist": hist, "w0k": w0k, "warps": warps}


def predictions(st):
    """Measured duty + per-mechanism predictions (first-order)."""
    cyc = max(1, st["cyc"])
    warps = st["warps"]
    meas = warps[0]["iss"] / cyc
    w0_rdy = warps[0]["rdy"] / cyc
    # compaction only: round-robin over the ready set = sum 1/k over
    # cycles where w0 is ready with ready-count k (exact from w0k)
    compact = sum((st["w0k"][k] / k) for k in range(1, 9)) / cyc
    # mean OTHER-ready count and its class composition (warps 1..7)
    oth_rdy = sum(warps[w]["rdy"] for w in range(1, 8)) / cyc
    oth_park = sum(warps[w]["cls"][1] + warps[w]["cls"][2]
                   for w in range(1, 8)) / cyc
    oth_poll = sum(warps[w]["cls"][3] for w in range(1, 8)) / cyc
    def mean_field(removed):
        r_eff = max(0.0, oth_rdy - removed)
        return w0_rdy / (1.0 + r_eff)
    out = {
        "duty_measured": meas,
        "w0_ready_frac": w0_rdy,
        "ready_mean": sum(k * h for k, h in enumerate(st["hist"])) / cyc,
        "other_ready_mean": oth_rdy,
        "other_parked_in_done_spin": oth_park,
        "other_in_engine_poll": oth_poll,
        "pred_compact_only": compact,
        "pred_plus_engine_park": mean_field(oth_poll),
        "pred_plus_done_halt": mean_field(oth_park),
        "pred_both": mean_field(oth_poll + oth_park),
    }
    for cap_name, cap in (("period5", 1 / 5), ("period4", 1 / 4)):
        out[f"speedup_both_{cap_name}"] = min(out["pred_both"], cap) / max(
            1e-9, meas)
        out[f"speedup_compact_{cap_name}"] = min(
            out["pred_compact_only"], cap) / max(1e-9, meas)
    return out


def main():
    rt = Runtime()
    rep = {"workloads": {}}
    try:
        # ---- (a) the scalar worst case ----
        kern = rt.compile(MK / "sw/kernels/fastpath_a_scalar.c")
        for lat in LATS:
            rt.set_latency(lat)
            out_b = rt.alloc(4 * 4)
            meta = arm(rt, kern)
            rt.launch(kern, grid_n=64, args=[4000, out_b],
                      max_cycles=3_000_000, chunk=100_000)
            st = collect(rt)
            stamps = rt.from_device(out_b, np.uint32, 3)
            p = predictions(st)
            p["tid0_loop_cycles"] = int(stamps[1]) - int(stamps[0])
            rep["workloads"][f"scalar_lat{lat}"] = {
                "classes": meta, "raw": st, "analysis": p}
            print(f"[a pilot] scalar LAT={lat}: duty {p['duty_measured']:.4f}"
                  f" ready_mean {p['ready_mean']:.2f} "
                  f"pred(compact {p['pred_compact_only']:.3f} "
                  f"+park {p['pred_plus_engine_park']:.3f} "
                  f"+halt {p['pred_plus_done_halt']:.3f} "
                  f"both {p['pred_both']:.3f}) "
                  f"speedup both@p5 {p['speedup_both_period5']:.2f}x "
                  f"@p4 {p['speedup_both_period4']:.2f}x")
            rt.free_all()

        # ---- (b) the C* mixed ISA shape ----
        rng = np.random.default_rng(0x5337C0DE)
        g = torus(32, 32, rng)
        bias = np.array([rnz(rng, 8) for _ in range(N)], np.int64)
        full, ffl = image_from_graph(g, bias, [(128, 1)], seed=0x5337C0DE)
        rows = [(g.adj[i], int(bias[i])) for i in range(N)]
        rimg, rfl = build_image(N, rows=rows)
        kern = rt.compile(MK / "sw/kernels/s7_ctax.c")
        for lat in LATS:
            rt.set_latency(lat)
            fb = rt.to_device(np.array(full, dtype="<u4"))
            rb = rt.to_device(np.array(rimg, dtype="<u4"))
            dest = rt.alloc(4 * (1 + N + 8 * N))
            st_b = rt.alloc(4 * 4 * 2)
            meta = arm(rt, kern)
            rt.launch(kern, grid_n=64,
                      args=[fb, len(full), ffl, rb, len(rimg), rfl,
                            8, 128, SF_IMM | SF_RECORD | SF_STATS_RESET,
                            dest, st_b, 2],
                      max_cycles=60_000_000, chunk=500_000)
            st = collect(rt)
            p = predictions(st)
            rep["workloads"][f"cstar_lat{lat}"] = {
                "classes": meta, "raw": st, "analysis": p}
            print(f"[a pilot] cstar  LAT={lat}: duty {p['duty_measured']:.4f}"
                  f" ready_mean {p['ready_mean']:.2f} "
                  f"poll_frac {p['other_in_engine_poll']:.2f} "
                  f"park_frac {p['other_parked_in_done_spin']:.2f} "
                  f"pred both {p['pred_both']:.3f}")
            rt.free_all()

        # ---- (c) the engine-wait GEMM shape ----
        rng = np.random.default_rng(0xC2C2)
        A = rng.integers(-128, 128, size=64 * 64, dtype=np.int8)
        B = rng.integers(-128, 128, size=64 * 8, dtype=np.int8)
        kern = rt.compile(MK / "sw/kernels/fastpath_gemm.c")
        for lat in LATS:
            rt.set_latency(lat)
            ab = rt.to_device(A.view(np.uint8))
            bb = rt.to_device(B.view(np.uint8))
            cb = rt.alloc(4 * 64 * 8)
            st_b = rt.alloc(4 * 3 * 2)
            meta = arm(rt, kern)
            rt.launch(kern, grid_n=64,
                      args=[ab, bb, cb, 64, 1, 64, 64, 8, 64, st_b, 2],
                      max_cycles=20_000_000, chunk=200_000)
            st = collect(rt)
            p = predictions(st)
            rep["workloads"][f"gemm_lat{lat}"] = {
                "classes": meta, "raw": st, "analysis": p}
            print(f"[a pilot] gemm   LAT={lat}: duty {p['duty_measured']:.4f}"
                  f" ready_mean {p['ready_mean']:.2f} "
                  f"poll_frac {p['other_in_engine_poll']:.2f} "
                  f"park_frac {p['other_parked_in_done_spin']:.2f} "
                  f"pred both {p['pred_both']:.3f}")
            rt.free_all()
    finally:
        rt.close()

    ev = MK / "ci" / "logs" / "fastpath"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "a_pilot.json").write_text(json.dumps(rep, indent=1))
    print(f"[a pilot] evidence: {ev / 'a_pilot.json'}")
    # the decision line (scalar @ LAT=1 is the frozen decision driver)
    key = "scalar_lat1"
    p = rep["workloads"][key]["analysis"]
    print(f"[a pilot] DECISION DRIVER {key}: measured duty "
          f"{p['duty_measured']:.4f}; compact-only speedup capped @p5 "
          f"{p['speedup_compact_period5']:.2f}x / @p4 "
          f"{p['speedup_compact_period4']:.2f}x; with done-halt "
          f"{p['speedup_both_period5']:.2f}x / "
          f"{p['speedup_both_period4']:.2f}x (bar 2.0x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
