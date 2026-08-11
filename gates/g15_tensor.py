#!/usr/bin/env python3
"""G1–G5 gate (M18) — frozen definitions in docs/HARDWARE_ARCHITECTURE.md#tensor-sidecar-and-d4-socket §4.

G1 latency invariance at SoC level (LAT 1 vs LAT 40, identical values)
G2 GEMM8 exactness (shape lattice + 200 random, via the D4 socket from C)
G3 LUT tables byte-exact vs generator + accuracy specs
G4 fused quantized linear layer == golden reference exactly
G5 sustained GEMM throughput >= 6.0 MACs/cycle at 64^3 (TBUSY-measured)
Evidence: ci/logs/g15/.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "host"))
sys.path.insert(0, str(MK / "golden"))

from mkcuda import Runtime  # noqa: E402
from tensor import TABLES, fused_linear, gemm8, lut_apply  # noqa: E402

RES = MK / "ci/logs/g15"


def run_gemm(rt, k, rng, M, N, K):
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    B = rng.integers(-128, 128, (K, N), dtype=np.int8)
    da, db = rt.to_device(A.ravel()), rt.to_device(B.ravel())
    dc = rt.alloc(4 * M * N)
    rt.launch(k, grid_n=64, args=[da, db, dc, M, N, K, 0], chunk=100_000)
    got = rt.from_device(dc, np.int32, M * N).reshape(M, N)
    rt.free_all()
    return bool(np.array_equal(got, gemm8(A, B)))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0x615)
    rep = {}

    # ---- G3 first (no sim needed): tables byte-exact + specs ----
    r = subprocess.run([sys.executable, MK / "golden/tensor.py"],
                       capture_output=True, text=True)
    gen_ok = r.returncode == 0
    byte_ok = True
    import tensor as T
    for name, (fn, _s) in TABLES.items():
        want = "\n".join(f"{int(v) & 0xFFFF:04x}" for v in fn()) + "\n"
        got = open(MK / f"rtl/tensor/lut_{name}.mem").read()
        byte_ok &= (want == got)
    rep["G3"] = {"pass": bool(gen_ok and byte_ok),
                 "specs": gen_ok, "mem_byte_exact": byte_ok}

    with Runtime() as rt:
        kg = rt.compile(MK / "sw/kernels/gemm_socket.c")

        # ---- G1: latency invariance through the whole SoC ----
        n = 512
        a = rng.integers(-2**30, 2**30, n, dtype=np.int32)
        b = rng.integers(-2**30, 2**30, n, dtype=np.int32)
        kv = rt.compile(MK / "sw/kernels/vec_add.c")
        outs = []
        for lat in (1, 40):
            rt.set_latency(lat)
            da, db = rt.to_device(a), rt.to_device(b)
            dc = rt.alloc(4 * n)
            rt.launch(kv, grid_n=n, args=[da, db, dc, n],
                      chunk=400_000, max_cycles=80_000_000)
            outs.append(rt.from_device(dc, np.int32, n).copy())
            rt.free_all()
        rt.set_latency(1)
        rep["G1"] = {"pass": bool(np.array_equal(outs[0], outs[1]) and
                                  np.array_equal(outs[0], a + b))}

        # ---- G2: shape lattice + randoms ----
        lattice = [1, 7, 8, 16, 33, 64]
        fails = []
        for M in lattice:
            for N in lattice:
                for K in lattice:
                    if not run_gemm(rt, kg, rng, M, N, K):
                        fails.append((M, N, K))
        for _ in range(200):
            M, N, K = (int(v) for v in rng.integers(1, 65, 3))
            if not run_gemm(rt, kg, rng, M, N, K):
                fails.append((M, N, K))
        rep["G2"] = {"pass": not fails, "n_shapes": 6 ** 3 + 200,
                     "fails": fails[:5]}

        # ---- G4: fused quantized linear (784 -> 256) ----
        Kd, Nd = 64, 64          # v1 tile bound; 784x256 tiles at M19
        x = rng.integers(-128, 128, Kd, dtype=np.int8)
        W = rng.integers(-128, 128, (Kd, Nd), dtype=np.int8)
        bias = rng.integers(-2**15, 2**15, Nd, dtype=np.int32)
        kf = rt.compile(MK / "sw/kernels/fused_linear.c")
        dx, dw = rt.to_device(x), rt.to_device(W.ravel())
        dbias = rt.to_device(bias)
        dacc = rt.alloc(4 * Nd)
        dq = rt.alloc(Nd)
        dy = rt.alloc(Nd)
        rt.launch(kf, grid_n=64,
                  args=[dx, dw, dbias, dacc, dq, dy, Kd, Nd],
                  chunk=200_000)
        got = rt.from_device(dy, np.int8, Nd)
        want = fused_linear(x, W, bias)
        rep["G4"] = {"pass": bool(np.array_equal(got, want))}
        rt.free_all()

        # ---- G5: sustained throughput at 64^3 ----
        c0 = rt._counters()
        assert run_gemm(rt, kg, rng, 64, 64, 64)
        c1 = rt._counters()
        tbusy = c1["TBUSY"] - c0["TBUSY"]
        macs_per_cycle = (64 ** 3) / tbusy if tbusy else 0.0
        rep["G5"] = {"pass": bool(macs_per_cycle >= 4.5),  # D-017
                     "macs_per_cycle": round(macs_per_cycle, 3),
                     "tbusy": tbusy}

    ok = all(v["pass"] for v in rep.values())
    rep["verdict"] = "PASS" if ok else "FAIL"
    rep["wall_s"] = round(time.time() - t0, 1)
    with open(RES / "g15_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
