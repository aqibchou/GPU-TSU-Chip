#!/usr/bin/env python3
"""Phase-6σ DoD gate (M17): the tiny-CUDA stack battery.

════════ FROZEN DECISION RULES (runtime_spec §7, frozen 2026-07-07) ════
1. Toolchain: all five gated kernels compile at the enforced SIMT-safe
   flags (-O0 default per the compiler-shape findings, D-016 narrative).
2. Execution: every kernel's device output diffs EXACTLY vs NumPy
   (integer kernels, zero tolerance) on the persistent sim backend.
3. Persistence: all launches run through ONE harness process, and kernel
   1 re-runs LAST (reset isolation proven by exactness after 5 others).
4. Divergence coverage: relu_abs (nested forward divergence) and every
   kernel's grid-stride idiom with grid_n > 64 both in the battery.
5. Counters: DBEATS(bcast_scale) < DBEATS(vec_add) at matched element
   count — the M16 coalescer's 8:1 dedupe visible through the REAL stack
   (bcast's s[] loads are warp-uniform; vec_add's are all-distinct).
Evidence: ci/logs/m17/. Positive completion: this script's PASS verdict.
════════════════════════════════════════════════════════════════════════
"""
import json
import pathlib
import sys
import time

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "host"))

from mkcuda import Runtime  # noqa: E402

RES = MK / "ci/logs/m17"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0xC0DA)
    rows = {}
    with Runtime() as rt:
        # vec_add (kernel 1)
        k = rt.compile(MK / "sw/kernels/vec_add.c")
        n = 1000
        a = rng.integers(-2**30, 2**30, n, dtype=np.int32)
        b = rng.integers(-2**30, 2**30, n, dtype=np.int32)
        da, db, dc = rt.to_device(a), rt.to_device(b), rt.alloc(4 * n)
        st = rt.launch(k, grid_n=n, args=[da, db, dc, n], chunk=50_000)
        rows["vec_add"] = (bool(np.array_equal(
            rt.from_device(dc, np.int32, n), a + b)), st)
        vec_dbeats_per_elem = st.dbeats / n
        rt.free_all()

        k = rt.compile(MK / "sw/kernels/saxpy_i32.c")
        m, alpha = 500, 7
        x = rng.integers(-1000, 1000, m, dtype=np.int32)
        y = rng.integers(-1000, 1000, m, dtype=np.int32)
        dx, dy = rt.to_device(x), rt.to_device(y)
        st = rt.launch(k, grid_n=m, args=[dx, dy, alpha, m], chunk=200_000)
        rows["saxpy_i32"] = (bool(np.array_equal(
            rt.from_device(dy, np.int32, m), alpha * x + y)), st)
        rt.free_all()

        k = rt.compile(MK / "sw/kernels/relu_abs.c")
        x = rng.integers(-1000, 1000, m, dtype=np.int32)
        dx, do = rt.to_device(x), rt.alloc(4 * m)
        st = rt.launch(k, grid_n=m, args=[dx, do, m], chunk=50_000)
        idx = np.arange(m)
        want = np.where(x >= 0, x,
                        np.where(idx % 2 == 1, -x, 0)).astype(np.int32)
        rows["relu_abs"] = (bool(np.array_equal(
            rt.from_device(do, np.int32, m), want)), st)
        rt.free_all()

        k = rt.compile(MK / "sw/kernels/reduce_partial.c")
        n2 = 3000
        x = rng.integers(-1000, 1000, n2, dtype=np.int32)
        dx, dp = rt.to_device(x), rt.alloc(4 * 64)
        st = rt.launch(k, grid_n=n2, args=[dx, dp, n2], chunk=100_000)
        rows["reduce_partial"] = (
            int(rt.from_device(dp, np.int32, 64).sum()) == int(x.sum()), st)
        rt.free_all()

        k = rt.compile(MK / "sw/kernels/bcast_scale.c")
        x = rng.integers(-1000, 1000, n2, dtype=np.int32)
        s = rng.integers(-50, 50, 8, dtype=np.int32)
        dx, ds, do = rt.to_device(x), rt.to_device(s), rt.alloc(4 * n2)
        st = rt.launch(k, grid_n=n2, args=[dx, ds, do, n2], chunk=100_000)
        idx = np.arange(n2)
        rows["bcast_scale"] = (bool(np.array_equal(
            rt.from_device(do, np.int32, n2), x + s[(idx >> 6) & 7])), st)
        bcast_dbeats_per_elem = st.dbeats / n2
        rt.free_all()

        # rule 3: kernel 1 re-runs LAST in the same process
        k = rt.compile(MK / "sw/kernels/vec_add.c")
        da, db, dc = rt.to_device(a[:64]), rt.to_device(b[:64]), \
            rt.alloc(4 * 64)
        st = rt.launch(k, grid_n=64, args=[da, db, dc, 64], chunk=25_000)
        rows["vec_add_rerun"] = (bool(np.array_equal(
            rt.from_device(dc, np.int32, 64), a[:64] + b[:64])), st)

    rule2 = all(ok for ok, _ in rows.values())
    rule5 = bcast_dbeats_per_elem < vec_dbeats_per_elem
    report = {
        "kernels": {name: {"exact": ok, "cycles": st.cycles,
                           "ipc": round(st.ipc, 3), "dbeats": st.dbeats}
                    for name, (ok, st) in rows.items()},
        "rule2_all_exact": rule2,
        "rule3_persistence": rows["vec_add_rerun"][0],
        "rule5_coalescing": {"pass": bool(rule5),
                             "vec_dbeats_per_elem":
                                 round(vec_dbeats_per_elem, 3),
                             "bcast_dbeats_per_elem":
                                 round(bcast_dbeats_per_elem, 3)},
        "wall_s": round(time.time() - t0, 1),
    }
    report["verdict"] = "PASS" if (rule2 and rule5) else "FAIL"
    with open(RES / "m17_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
