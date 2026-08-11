#!/usr/bin/env python3
"""FASTPATH leg D bar gate (D-032d; sub-bars frozen in
docs/FPGA_IMPLEMENTATION.md#fast-path-optimizations 2026-07-15 BEFORE the RTL was written).

D.1  the chain-differential gap bar: serving chain (GEMV 64x1x64 /
     GELU 64 alternating) issued in the QUEUED discipline (mk_t_post
     x n, one mk_t_wait), chain lengths 21 vs 1 (20 op BOUNDARIES = n-1 gaps):
       gaps := [wall(21) - wall(1)] - [dTBUSY(21) - dTBUSY(1)]
     must be <= 4 cycles PER BOUNDARY at LAT in {1, 4}. (Instrument
     note, disclosed in the card: the originally-written 3-vs-1 form
     carries +-16..45 cycles of rotation-phase quantization per
     boundary — first-contact even produced a negative 2-pair gap —
     so the pair count amortizes it; the frozen <= 4/pair bound is
     unchanged.)
CTRL the POLLED discipline (mode 0, the pre-D per-op wait) re-run on
     this build must stay within [0.8, 1.25] of the banked PER-BOUNDARY
     baseline (780 cyc @ LAT=1 / 943 @ LAT=4, ci/logs/fastpath/
     d_baseline.json, measured pre-RTL) — the queue must not change
     the polled path's timing shape.
D.3b queued-vs-polled OUTPUT EQUIVALENCE: identical inputs through
     both disciplines produce byte-identical C and LUT result
     buffers (absolute correctness of the ops rides G15/S7sigma in
     the battery; this pins the queue path specifically).
(D.3a — GO at FULL traps mcause 2 — rides the flipped s7_gowb kernel
in the S7.5 battery leg.)

Evidence: ci/logs/fastpath/d_bar.json.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

from mkcuda import Runtime                                       # noqa: E402

LATS = (1, 4)
N_LONG = 21                   # 20 boundaries (jitter amortization)
BASELINE_PAIR = {1: 780, 4: 943}  # banked pre-RTL polled gap/boundary


def run_chain(rt, kern, A, B, n_ops, mode):
    ab = rt.to_device(A.view(np.uint8))
    bb = rt.to_device(B.view(np.uint8))
    cb = rt.alloc(4 * 64)
    ldb = rt.alloc(64)
    ob = rt.alloc(16)
    c0 = rt._counters()
    rt.launch(kern, grid_n=64, args=[ab, bb, cb, ldb, n_ops, ob, mode],
              max_cycles=8_000_000, chunk=100_000)
    c1 = rt._counters()
    st = rt.from_device(ob, np.uint32, 2)
    c_img = rt.from_device(cb, np.uint32, 64).tobytes()
    l_img = rt.from_device(ldb, np.uint8, 64).tobytes()
    out = {"wall": int(st[1]) - int(st[0]),
           "tbusy": c1["TBUSY"] - c0["TBUSY"],
           "c_sha": hashlib.sha256(c_img).hexdigest(),
           "l_sha": hashlib.sha256(l_img).hexdigest()}
    rt.free_all()
    return out


def main():
    rt = Runtime()
    rep = {"lats": {}}
    ok = True
    try:
        kern = rt.compile(MK / "sw/kernels/fastpath_d_chain.c")
        rng = np.random.default_rng(0xD032)
        A = rng.integers(-128, 128, size=64 * 64, dtype=np.int8)
        B = rng.integers(-128, 128, size=64, dtype=np.int8)
        pairs = N_LONG - 1     # boundaries in the differential
        for lat in LATS:
            rt.set_latency(lat)
            r = {m: {n: run_chain(rt, kern, A, B, n, m)
                     for n in (1, N_LONG)} for m in (0, 1)}
            gq = ((r[1][N_LONG]["wall"] - r[1][1]["wall"])
                  - (r[1][N_LONG]["tbusy"] - r[1][1]["tbusy"])) / pairs
            gp = ((r[0][N_LONG]["wall"] - r[0][1]["wall"])
                  - (r[0][N_LONG]["tbusy"] - r[0][1]["tbusy"])) / pairs
            ctrl = gp / BASELINE_PAIR[lat]
            eq = (r[0][N_LONG]["c_sha"] == r[1][N_LONG]["c_sha"]
                  and r[0][N_LONG]["l_sha"] == r[1][N_LONG]["l_sha"])
            d1 = gq <= 4.0
            dc = 0.8 <= ctrl <= 1.25
            ok &= d1 and dc and eq
            rep["lats"][str(lat)] = {
                "gap_per_boundary_queued": gq, "gap_per_boundary_polled": gp,
                "polled_vs_baseline": ctrl, "outputs_equal": eq,
                "raw": r,
                "D.1": d1, "CTRL": dc, "D.3b": eq}
            print(f"[d bar] LAT={lat}: queued gap {gq:.1f} cyc/boundary "
                  f"(bar <= 4) {'PASS' if d1 else 'FAIL'}; polled "
                  f"{gp:.0f}/boundary = {ctrl:.2f}x baseline "
                  f"{'ok' if dc else 'MOVED'}; outputs "
                  f"{'identical' if eq else 'DIFFER'}")
    finally:
        rt.close()
    ev = MK / "ci" / "logs" / "fastpath"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "d_bar.json").write_text(json.dumps(rep, indent=1))
    print(f"[d bar] evidence: {ev / 'd_bar.json'}")
    print(f"FASTPATH-D: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
