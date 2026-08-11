#!/usr/bin/env python3
"""G6 debug: run the 2-token job on the RTL, then PEEK device scratch
and diff against golden step-5 references. Localizes the second-token
divergence (RTL [262,262] vs golden [262,3139])."""
import pathlib
import sys

import numpy as np

MK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK / "host"))
sys.path.insert(0, str(MK / "golden"))

from mkcuda import Runtime  # noqa: E402
from gpt2_device import serialize  # noqa: E402
from gpt2_int import IntGpt2, calib_windows  # noqa: E402
from gpt2_load import Bpe, load_weights  # noqa: E402

# kernel scratch word offsets (gpt2_seq.c S_* map)
S_X, S_H, S_QKV, S_LOGIT = 128, 896, 106880, 110152
S_KV = 160456
CTX, NVPAD = 32, 50304


def main():
    W = load_weights()
    bpe = Bpe()
    m = IntGpt2(W, calib_windows(bpe))
    prompt = bpe.encode("The capital of France is")
    n = len(prompt)

    # golden references
    g_trail = m.greedy(list(prompt), 2)[n:]
    ids6 = list(prompt) + [g_trail[0]]          # 6 tokens, step-5 input
    tr = {}
    lg_g = m.forward_int(ids6, trace=tr)        # (6, 50257) q8 logits
    print("golden trail:", g_trail, "| step5 argmax:",
          int(np.argmax(lg_g[-1])), flush=True)

    with Runtime() as rt:
        k = rt.compile(MK / "sw/kernels/gpt2_seq.c",
                       extra_cflags=("-O1", "-fno-reorder-blocks",
                                     "-fno-crossjumping"))
        desc, trail, dbgbuf = serialize(rt, m, prompt, 2)
        rt.launch(k, grid_n=64, args=[desc],
                  chunk=50_000_000, max_cycles=16_000_000_000)
        got_trail = list(rt.from_device(trail, np.uint32, 2))
        scr = int(rt.from_device(desc, np.uint32, 22)[20])

        class B:                                 # ad-hoc buffer handle
            def __init__(self, a): self.addr, self.nbytes = a, 0
            def __index__(self): return self.addr

        def peek(off_words, count, dtype=np.int32):
            return rt.from_device(B(scr + 4 * off_words), dtype, count)

        dbg = rt.from_device(B(int(dbgbuf)), np.int32, 7744)
        # GELUD blob on device vs generator
        from tensor import gelud_table
        gaddr = int(rt.from_device(desc, np.uint32, 22)[18])
        gdev = rt.from_device(B(gaddr), np.int32, 256)
        okg = np.array_equal(gdev.astype(np.int64), gelud_table())
        print("GELUD blob:", "EXACT" if okg else "MISMATCH")
        # pre-gelu g16 vs golden cfc output
        bacc_, Ms_, shs_ = m.iz["h.0.cfc"]
        from gpt2_int import sat16
        h1g = tr_h1 = None
        from tensor import gelud_table as _gt
        gt = _gt()
        print("kernel LUT probes: geld[0]", dbg[7680], "(want", int(gt[0]),
              ") geld[120]", dbg[7681], "(want", int(gt[120]),
              ") geld[128]", dbg[7682], "(want", int(gt[128]),
              ") ptr", hex(int(dbg[7683]) & 0xFFFFFFFF), "(want", hex(gaddr), ")")
        print("requant window i=0..7 (y16, gM, gsh, requant, sat15):")
        from gpt2_int import mkq
        gMg, gshg = m.iz["h.0.gM"], m.iz["h.0.gsh"]
        g16g0 = tr["g16_0"][0]
        g0g = tr["g0"][0]
        for i in range(8):
            b6 = dbg[7688 + 6 * i: 7694 + 6 * i]
            print(f"  i={i}: dev", list(b6), "want",
                  [None, int(gMg[i]), int(gshg[i]), int(g0g[i]),
                   int(g0g[i])])
        g16_g = tr.get("g16_0")
        if g16_g is not None:
            d16 = dbg[4608:7680]
            ok16 = np.array_equal(d16.astype(np.int64), g16_g[0])
            print("L0s0 cfc g16 (pre-gelu):", "EXACT" if ok16 else "MISMATCH")
            if not ok16:
                nz = np.nonzero(d16.astype(np.int64) != g16_g[0])[0]
                print(f"  {len(nz)} diffs, first [{nz[0]}] dev {d16[nz[0]]} "
                      f"gold {int(g16_g[0][nz[0]])}")
        for nm, sl, ref in (("L0s0 attn-o", slice(0, 768), tr["o0"][0]),
                            ("L0s0 x+attn", slice(768, 1536),
                             tr["x_attn0"][0]),
                            ("L0s0 gelu g15", slice(1536, 4608),
                             tr["g0"][0])):
            d = dbg[sl]
            ok = np.array_equal(d.astype(np.int64), ref)
            print(nm + ":", "EXACT" if ok else "MISMATCH")
            if not ok:
                nz = np.nonzero(d.astype(np.int64) != ref)[0]
                print(f"  {len(nz)} diffs, first [{nz[0]}] dev {d[nz[0]]} "
                      f"gold {int(ref[nz[0]])}")
        lg_d = peek(S_LOGIT, NVPAD)[:50257]
        x_d = peek(S_X, 768)
        print("device trail:", got_trail, flush=True)
        ok_l = np.array_equal(lg_d, lg_g[-1].astype(np.int32))
        print("step5 logits vs golden:", "EXACT" if ok_l else "MISMATCH")
        if not ok_l:
            nz = np.nonzero(lg_d != lg_g[-1])[0]
            print(f"  {len(nz)} diffs; device argmax {int(np.argmax(lg_d))}"
                  f" golden argmax {int(np.argmax(lg_g[-1]))}")
            for i in nz[:5]:
                print(f"  [{i}] dev {lg_d[i]} gold {int(lg_g[-1][i])}")
        x_g5 = tr["x"][-1]
        h_g5 = tr["h_lnf"][-1]
        ok_x = np.array_equal(x_d.astype(np.int64), x_g5)
        print("step5 residual x:", "EXACT" if ok_x else "MISMATCH")
        if not ok_x:
            nz = np.nonzero(x_d.astype(np.int64) != x_g5)[0]
            print(f"  {len(nz)} diffs, first [{nz[0]}] dev {x_d[nz[0]]} "
                  f"gold {int(x_g5[nz[0]])}")
        h_d = peek(S_H, 768)
        ok_h = np.array_equal(h_d.astype(np.int64), h_g5)
        print("step5 lnf h15:", "EXACT" if ok_h else "MISMATCH")
        if not ok_h:
            nz = np.nonzero(h_d.astype(np.int64) != h_g5)[0]
            print(f"  {len(nz)} diffs, first [{nz[0]}] dev {h_d[nz[0]]} "
                  f"gold {int(h_g5[nz[0]])}")
        # per-layer K-cache sweep: first (layer, pos) mismatch localizes
        print("--- KV sweep (first bad layer/pos) ---", flush=True)
        bad = None
        for li in range(12):
            k_gl = tr["k8"][li]                      # (6, 768) golden k8
            base = scr + 4 * S_KV + (li * 12) * 64 * CTX
            kT = rt.from_device(B(base), np.int8, 768 * CTX)
            kT = np.frombuffer(kT.tobytes(), np.int8).reshape(12, 64, CTX)
            dev = np.concatenate([kT[hd, :, :6].T for hd in range(12)],
                                 axis=1)             # (6, 768)
            for pos in range(6):
                if not np.array_equal(dev[pos].astype(np.int64),
                                      k_gl[pos]):
                    nz = np.nonzero(dev[pos].astype(np.int64)
                                    != k_gl[pos])[0]
                    print(f"L{li} pos{pos}: {len(nz)} diffs "
                          f"(first j={nz[0]} dev {dev[pos][nz[0]]} "
                          f"gold {int(k_gl[pos][nz[0]])})")
                    if bad is None:
                        bad = (li, pos)
                    break
        print("first bad (layer,pos):", bad, flush=True)
        # residual after step 5's 12 layers: golden equivalent
        # (recompute: forward_int internals — reuse the embed+layers by
        # calling forward_int and capturing x is invasive; instead check
        # KV row for layer 0, position 4 and 5, head 0)
        kv = peek(S_KV, 0, np.int32)  # placeholder
        kT0 = rt.from_device(B(scr + 4 * S_KV), np.int8, 64 * CTX)
        kT0 = np.frombuffer(kT0.tobytes(), dtype=np.int8).reshape(64, CTX)
        # golden k8 for layer 0 all positions (full-context forward)
        from gpt2_int import requant_vec, sat8, imm, int_layernorm
        q, z = m.q, m.iz
        x = requant_vec(q["emb.w"][:, ids6].T,
                        z["emb_M"][ids6][:, None],
                        z["emb_sh"][ids6][:, None]) + z["wpe"][:6]
        gq, bq = z["h.0.ln1"]
        h = int_layernorm(x, gq, bq)
        bacc, Ms, shs = z["h.0.qkv"]
        qkv = sat8(requant_vec(imm(h, q["h.0.attn.c_attn.w"]) + bacc,
                               Ms, shs))
        k_g = qkv[:, 768:768 + 64]              # (6, 64) head-0 k8
        dev_k = kT0[:, :6].T                    # (6, 64)
        ok_k = np.array_equal(dev_k.astype(np.int64), k_g)
        print("L0 head0 K cache (pos 0-5):", "EXACT" if ok_k else "MISMATCH")
        if not ok_k:
            for p in range(6):
                d = np.nonzero(dev_k[p].astype(np.int64) != k_g[p])[0]
                if len(d):
                    print(f"  pos {p}: {len(d)} diffs, first j={d[0]} "
                          f"dev {dev_k[p][d[0]]} gold {int(k_g[p][d[0]])}")


if __name__ == "__main__":
    main()
