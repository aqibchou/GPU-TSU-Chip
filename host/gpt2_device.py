#!/usr/bin/env python3
"""M19 device GPT-2 driver: serialize the FROZEN golden integer chain's
constants into device memory and run sw/kernels/gpt2_seq.c (the
on-device sequencer). G6's harness: device trail == golden greedy.

Layout must mirror gpt2_seq.c's descriptor/layer-table word indices.
"""
import os
import pathlib
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "host"))
sys.path.insert(0, str(MK / "golden"))

from mkcuda import Runtime  # noqa: E402
from gpt2_int import IntGpt2, calib_windows  # noqa: E402
from gpt2_load import Bpe, load_weights  # noqa: E402
from tensor import exp_table, gelud_table, rsqrt_table  # noqa: E402

CTX = 32
NVOCAB = 50257
NVPAD = 50304
L_WORDS = 26


def tile_pack(Wq):
    """(K, N) int8 -> [kb][nb][64][64] contiguous bytes (K, N % 64 == 0)."""
    K, N = Wq.shape
    t = Wq.reshape(K // 64, 64, N // 64, 64).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t).astype(np.int8).tobytes()


class Serializer:
    def __init__(self, rt):
        self.rt = rt

    def blob(self, data: bytes, align=64):
        b = self.rt.alloc(len(data), align=align)
        self.rt.write(b, data)
        return int(b)

    def words(self, arr):
        return self.blob(np.asarray(arr, dtype="<u4").tobytes())

    def iwords(self, arr):
        return self.blob(np.asarray(arr, dtype="<i4").tobytes())

    def bytes8(self, arr):
        return self.blob(np.asarray(arr, dtype=np.int8).tobytes())


def serialize(rt, m, prompt, n_gen):
    """Write every frozen constant + scratch; return desc pointer."""
    s = Serializer(rt)
    q, z = m.q, m.iz
    desc = rt.alloc(4 * 24)
    dbg = rt.alloc(4 * 7744)
    rt.write(dbg, b"\x00" * (4 * 7744))

    def pack_lin(key, zkey, npad=None):
        bacc, Ms, shs = z[zkey]
        Wq = q[key + ".w"]
        if npad and Wq.shape[1] < npad:
            Wq = np.pad(Wq, ((0, 0), (0, npad - Wq.shape[1])))
            bacc = np.pad(bacc, (0, npad - len(bacc)))
            Ms = np.pad(Ms, (0, npad - len(Ms)), constant_values=1)
            shs = np.pad(shs, (0, npad - len(shs)), constant_values=1)
        return (s.blob(tile_pack(Wq)), s.iwords(bacc),
                s.words(Ms), s.words(shs))

    layers = []
    for li in range(12):
        pp = f"h.{li}."
        g1, b1 = z[pp + "ln1"]
        g2, b2 = z[pp + "ln2"]
        wq, bq_, mq, shq = pack_lin(pp + "attn.c_attn", pp + "qkv")
        wp, bp, mp, shp = pack_lin(pp + "attn.c_proj", pp + "cproj")
        wf, bf, mf, shf = pack_lin(pp + "mlp.c_fc", pp + "cfc")
        wm, bm, mm, shm = pack_lin(pp + "mlp.c_proj", pp + "mproj")
        L = [s.iwords(g1), s.iwords(b1),
             wq, bq_, mq, shq,
             s.words(z[pp + "sM"]), s.words(z[pp + "ssh"]),
             s.words(z[pp + "oM"]), s.words(z[pp + "osh"]),
             wp, bp, mp, shp,
             s.iwords(g2), s.iwords(b2),
             wf, bf, mf, shf,
             s.words(z[pp + "gM"]), s.words(z[pp + "gsh"]),
             wm, bm, mm, shm]
        assert len(L) == L_WORDS
        layers.extend(L)
    layers_ptr = s.words(layers)

    # head: pad vocab to 50304, requant consts padded with sh=1
    hW = np.pad(q["head.w"], ((0, 0), (0, NVPAD - NVOCAB)))
    hM = np.pad(z["hM"], (0, NVPAD - NVOCAB), constant_values=1)
    hsh = np.pad(z["hsh"], (0, NVPAD - NVOCAB), constant_values=1)
    gf, bf_ = z["lnf"]

    # scratch: bar/phase/rows/logits + KV cache
    scratch_words = (128 + 768 * 2 + 768 * 2 + NVPAD * 2 + 3072
                     + 2304 + 768 + 192 + 8 + NVPAD)
    kv_bytes = 2 * 12 * 12 * 64 * CTX
    scratch = rt.alloc(4 * scratch_words + kv_bytes, align=64)
    rt.write(scratch, b"\x00" * (4 * scratch_words + kv_bytes))

    trail = rt.alloc(4 * n_gen)
    rt.write(trail, b"\x00" * 4 * n_gen)

    dwords = [0x6D6B6732, len(prompt), n_gen, CTX,
              s.words(prompt),
              s.bytes8(q["emb.w"].T),          # [50257][768] row-major
              s.words(z["emb_M"]), s.words(z["emb_sh"]),
              s.iwords(z["wpe"][:CTX]),
              layers_ptr,
              s.iwords(gf), s.iwords(bf_),
              s.blob(tile_pack(hW)), s.words(hM), s.words(hsh),
              NVOCAB, NVPAD,
              s.words(exp_table()),
              s.iwords(gelud_table()),
              s.words(rsqrt_table()),
              int(scratch), int(trail), int(dbg)]
    rt.write(desc, np.asarray(dwords, dtype="<u4").tobytes())
    return desc, trail, dbg


def main():
    n_gen = int(os.environ.get("MK_NGEN", "20"))
    W = load_weights()
    bpe = Bpe()
    print("[gpt2_device] building frozen golden chain…", flush=True)
    m = IntGpt2(W, calib_windows(bpe))
    prompt = bpe.encode("The capital of France is")
    want = m.greedy(list(prompt), n_gen)[len(prompt):]
    print("[gpt2_device] golden trail:", want, flush=True)

    with Runtime() as rt:
        k = rt.compile(MK / "sw/kernels/gpt2_seq.c",
                       extra_cflags=("-O1", "-fno-reorder-blocks",
                                     "-fno-crossjumping"))
        # -O1 is legal for THIS kernel because it obeys the divergence
        # construction contract (docs/SOFTWARE_AND_VALIDATION.md#gpt-2-device-sequencer-notes): lane-
        # varying guards only in -O0 .text.main structure functions
        # (linked LAST so divergent-region calls target lower pcs),
        # hot math in branchless -O1 leaves. ISS-validated.
        print("[gpt2_device] serializing…", flush=True)
        desc, trail = serialize(rt, m, prompt, n_gen)
        print(f"[gpt2_device] heap used: "
              f"{(rt._heap - 0x200000) / 1e6:.1f} MB", flush=True)
        try:
            rt.launch(k, grid_n=64, args=[desc],
                      chunk=20_000_000,
                      max_cycles=int(os.environ.get("MK_MAXCYC",
                                                    "6000000000")))
        except Exception as e:
            print("TIMEOUT:", e)
            print("counters:", rt._counters())
            print("asserts:", rt._assert_failures()[:8])
            logs = rt.device_log()
            for t in (0, 1, 8, 63):
                print(f"log[{t}]:", [hex(v) for v in logs.get(t, [])][:34])
            # barrier + phase words
            scr_base = int(desc) and None
        got = list(rt.from_device(trail, np.uint32, n_gen))
    print("[gpt2_device] device trail:", got, flush=True)
    print("[gpt2_device] golden trail:", want, flush=True)
    ok = got == [int(v) for v in want]
    print("MATCH" if ok else "MISMATCH")
    print("decoded:", repr(bpe.decode(list(prompt) + got)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
