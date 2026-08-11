#!/usr/bin/env python3
"""run_kernel.py — the mk tiny-CUDA CLI (M17, runtime_spec §6).

  run_kernel.py run KERNEL.c --grid N [--args SPEC ...] [--backend sim|hw]
                [--profile] [--log] [--dump OUT.npy:BUF]
  run_kernel.py disasm KERNEL.c
  run_kernel.py gate                 # the Phase-6sigma DoD battery

arg SPECs (ordered):  in:FILE.npy | out:NBYTES | int:VALUE
Outputs declared with out: are downloaded after the launch and printed
(or saved via --dump idx:FILE.npy).
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "host"))
from mkcuda import Runtime  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("kernel")
    r.add_argument("--grid", type=int, required=True)
    r.add_argument("--args", nargs="*", default=[])
    r.add_argument("--backend", default="sim")
    r.add_argument("--profile", action="store_true")
    r.add_argument("--log", action="store_true")
    r.add_argument("--chunk", type=int, default=100_000)
    d = sub.add_parser("disasm")
    d.add_argument("kernel")
    sub.add_parser("gate")
    a = ap.parse_args()

    if a.cmd == "gate":
        sys.exit(subprocess.run([sys.executable,
                                 MK / "gates/m17_runtime.py"]).returncode)
    if a.cmd == "disasm":
        with Runtime() as rt:
            k = rt.compile(a.kernel)
            print(open(k.disasm_path).read())
        return

    with Runtime(backend=a.backend) as rt:
        k = rt.compile(a.kernel)
        launch_args, outs = [], []
        for spec in a.args:
            kind, _, val = spec.partition(":")
            if kind == "in":
                arr = np.load(val)
                launch_args.append(rt.to_device(arr))
            elif kind == "out":
                b = rt.alloc(int(val))
                launch_args.append(b)
                outs.append(b)
            elif kind == "int":
                launch_args.append(int(val, 0))
            else:
                sys.exit(f"bad arg spec {spec}")
        st = rt.launch(k, grid_n=a.grid, args=launch_args, chunk=a.chunk)
        for i, b in enumerate(outs):
            arr = rt.from_device(b, np.int32, b.nbytes // 4)
            print(f"out[{i}] ({b.nbytes} B): {arr[:16]}"
                  f"{' ...' if len(arr) > 16 else ''}")
        if a.profile:
            print(f"cycles={st.cycles} commits={st.commits} "
                  f"ipc={st.ipc:.3f} wb={st.wb_commits} "
                  f"mem={st.mem_commits} dbeats={st.dbeats} "
                  f"wall={st.wall_s:.2f}s")
        if a.log:
            for tid, vals in sorted(rt.device_log().items()):
                print(f"log tid {tid}: {[hex(v) for v in vals]}")


if __name__ == "__main__":
    main()
