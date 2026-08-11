#!/usr/bin/env python3
"""Cross-hart store->load coherence probe (HANDOFF V6.4, H2).

Runs sw/kernels/coherence_probe.c: tid0 (warp 0, single active lane)
ping-pongs 4 rounds against warp 1 (tids 8-15, warp-uniform) through
plain scratch-memory words — no fence, no special ops. The kernel
always completes (both sides bounded-spin and write a verdict), so
the outcome is data, not a timeout.

  PASS    -> cross-hart visibility works; a warp-granular worker
             pool (t5 lever #1) is hardware-possible. sc[4]/rounds
             is the visibility-latency proxy in spin iterations.
  TIMEOUT -> stores never became visible cross-hart; lever #1 is off
             the table on this SoC without RTL (ISA-v2 wishlist).

Either verdict is a valid banked measurement; the exit code (0 only
on PASS) is for scripting. Evidence: ci/logs/t5/coherence_probe.json
"""
import json
import pathlib
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

from mkcuda import Runtime                                       # noqa: E402

OUT = MK / "ci/logs/t5"
ROUNDS = 4


def main():
    rt = Runtime()
    kern = rt.compile(MK / "sw/kernels/coherence_probe.c")
    sc = rt.alloc(64)
    rt.memset(sc)
    st = rt.launch(kern, grid_n=64, args=[sc.addr], max_cycles=50_000_000)
    w = np.frombuffer(rt.read(sc, 28), dtype="<u4")
    verdict = {1: "PASS", 2: "TIMEOUT"}.get(int(w[2]), f"UNSET({int(w[2])})")
    res = {
        "verdict": verdict,
        "rounds_acked": int(w[3]),
        "rounds_asked": ROUNDS,
        "tid0_spins_total": int(w[4]),
        "warp1_pings_seen": int(w[5]),
        "warp1_timeout": int(w[6]),
        "cycles": int(st.cycles),
    }
    print(json.dumps(res, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coherence_probe.json").write_text(json.dumps(res, indent=1))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
