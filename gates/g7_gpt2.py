#!/usr/bin/env python3
"""G7 gate (M19) — frozen in docs/SOFTWARE_AND_VALIDATION.md#gpt-2-integer-pipeline §4.

The Q8 golden integer chain's perplexity on the frozen 2048-token eval
slice must be within 5% relative of the FP32 numpy forward. The FP32
forward itself is sanity-bounded (finite, ppl in [15, 60]) to catch
broken loads, not as a leaderboard claim. Calibration: the frozen
4x1024-token train-region protocol (golden/gpt2_int.calib_windows).
Evidence: ci/logs/g7/.
"""
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "golden"))

from gpt2_int import IntGpt2, calib_windows, eval_slice  # noqa: E402
from gpt2_load import Bpe, load_weights, ppl  # noqa: E402

RES = MK / "ci/logs/g7"
EVAL_SHA = "386e62ab532d2ab2"          # frozen at first materialization
BAR_REL = 0.05


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    W = load_weights()
    bpe = Bpe()
    eids = eval_slice(bpe)
    sha = hashlib.sha256(bytes(str(eids), "utf8")).hexdigest()[:16]
    rep = {"eval_sha": sha, "eval_sha_ok": sha == EVAL_SHA,
           "n_eval": len(eids)}

    p_fp = ppl(W, eids)
    rep["fp32_ppl"] = round(p_fp, 3)
    rep["fp32_sane"] = bool(np.isfinite(p_fp) and 15 < p_fp < 60)

    m = IntGpt2(W, calib_windows(bpe))
    p_q8 = m.ppl_int(eids)
    rel = (p_q8 - p_fp) / p_fp
    rep["q8_ppl"] = round(p_q8, 3)
    rep["rel_delta"] = round(rel, 5)
    rep["bar"] = BAR_REL
    rep["G7"] = bool(rep["eval_sha_ok"] and rep["fp32_sane"]
                     and abs(rel) <= BAR_REL)

    # greedy agreement is reported as evidence, not a bar
    prompt = bpe.encode("The capital of France is")
    gi = m.greedy(list(prompt), 6)
    rep["greedy_int"] = bpe.decode(gi)

    rep["verdict"] = "PASS" if rep["G7"] else "FAIL"
    rep["wall_s"] = round(time.time() - t0, 1)
    with open(RES / "g7_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["G7"] else 1)


if __name__ == "__main__":
    main()
