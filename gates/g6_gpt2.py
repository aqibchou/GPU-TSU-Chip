#!/usr/bin/env python3
"""G6 gate (M19) — frozen in docs/SOFTWARE_AND_VALIDATION.md#gpt-2-integer-pipeline §4.

20 greedy tokens generated ON the simt_soc (sw/kernels/gpt2_seq.c, the
64-hart cooperative sequencer) must equal the golden integer chain's 20
tokens EXACTLY — token IDs identical; both sides integer-deterministic,
so the bar is bit-exactness, not closeness. Frozen prompt: "The capital
of France is". Evidence: ci/logs/g6/. Runs nightly (hours of sim wall
at -O0 — the v1 reconvergence contract forbids optimizing compilers;
docs/SOFTWARE_AND_VALIDATION.md#gpt-2-device-sequencer-notes).
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "host"))
sys.path.insert(0, str(MK / "golden"))

from mkcuda import Runtime  # noqa: E402
from gpt2_device import serialize  # noqa: E402
from gpt2_int import IntGpt2, calib_windows  # noqa: E402
from gpt2_load import Bpe, load_weights  # noqa: E402

RES = MK / "ci/logs/g6"
N_GEN = int(os.environ.get("MK_NGEN", "20"))
MAXCYC = int(os.environ.get("MK_MAXCYC", "60000000000"))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    W = load_weights()
    bpe = Bpe()
    m = IntGpt2(W, calib_windows(bpe))
    prompt = bpe.encode("The capital of France is")
    want = [int(v) for v in m.greedy(list(prompt), N_GEN)[len(prompt):]]

    rep = {"n_gen": N_GEN, "golden_trail": want}
    with Runtime() as rt:
        k = rt.compile(MK / "sw/kernels/gpt2_seq.c",
                       extra_cflags=("-O1", "-fno-reorder-blocks",
                                     "-fno-crossjumping"))
        desc, trail, _dbg = serialize(rt, m, prompt, N_GEN)
        rt.launch(k, grid_n=64, args=[desc],
                  chunk=50_000_000, max_cycles=MAXCYC)
        got = [int(v) for v in rt.from_device(trail, np.uint32, N_GEN)]
        rep["counters"] = rt._counters()
    rep["device_trail"] = got
    rep["G6"] = bool(got == want)
    rep["decoded"] = bpe.decode(list(prompt) + got)
    rep["verdict"] = "PASS" if rep["G6"] else "FAIL"
    rep["wall_s"] = round(time.time() - t0, 1)
    with open(RES / "g6_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["G6"] else 1)


if __name__ == "__main__":
    main()
