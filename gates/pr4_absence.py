#!/usr/bin/env python3
"""PR4 red path (tensor_spec §1c, D-036; PROFILES card): on the build
selected by MK_PROFILE, T_PROFILE reads back the build's id, and GO at
an op absent from the profile traps mcause 2 (assert code 0x80000002)
before anything dispatches. Union (0) probes nothing (all present).
Evidence: ci/logs/profiles/pr4_p<N>.json
"""
import json
import os
import pathlib
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK))
sys.path.insert(0, str(MK / "host"))

from mkcuda import Buffer, KernelAssert, Runtime                 # noqa: E402

PROF = int(os.environ.get("MK_PROFILE", "0"))
ABSENT = {0: None, 1: 9, 2: 7, 3: 7}[PROF]
OUT = MK / "ci/logs/profiles"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rt = Runtime()
    kern = rt.compile(MK / "sw/kernels/pr4_probe.c")
    res = rt.alloc(16)
    rt.memset(res)
    rt.launch(kern, grid_n=64, args=[res.addr, 0xFFFF],
              max_cycles=2_000_000)
    rid = int(np.frombuffer(rt.read(res, 4), dtype="<u4")[0])
    assert rid == PROF, f"T_PROFILE read {rid}, build is {PROF}"
    print(f"  T_PROFILE == {rid} OK", flush=True)

    trapped, code = False, 0
    if ABSENT is not None:
        rt.memset(res)
        try:
            rt.launch(kern, grid_n=64, args=[res.addr, ABSENT],
                      max_cycles=2_000_000)
        except KernelAssert as e:
            fails = e.args[0]
            trapped = True
            code = int(fails[0][1])
        assert trapped, f"GO at absent op {ABSENT} did not trap"
        assert code == 0x80000002, f"trap code {code:#x} != mcause 2"
        reached = int(np.frombuffer(rt.read(res, 8), dtype="<u4")[1])
        assert reached != 0xD0, "post-GO write reached despite trap"
        print(f"  GO at absent op {ABSENT}: trapped mcause 2 OK",
              flush=True)
    rep = {"profile": PROF, "t_profile_read": rid,
           "absent_op_probed": ABSENT, "trapped": trapped,
           "trap_code": code, "verdict": "PASS"}
    (OUT / f"pr4_p{PROF}.json").write_text(json.dumps(rep, indent=1))
    print(f"PR4 (profile {PROF}): PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
