#!/usr/bin/env python3
"""M19: GPT-2 small loaded from raw safetensors + BPE, pure numpy — no
torch in the trust chain. Provides the FP32 forward (the G7 baseline) and
the tokenizer both gates share.

HF GPT-2 stores linear layers as Conv1D: weight shape (in, out), so the
forward is x @ W + b throughout.
"""
import json
import os

import numpy as np

MK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(MK, "data/gpt2")
N_LAYER, N_HEAD, D_MODEL, N_CTX = 12, 12, 768, 1024


def load_weights():
    path = os.path.join(D, "model.safetensors")
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        base = 8 + n
        buf = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        assert meta["dtype"] == "F32", (name, meta["dtype"])
        b0, b1 = meta["data_offsets"]
        out[name] = np.frombuffer(
            buf[base + b0:base + b1], dtype="<f4").reshape(meta["shape"])
    return out


# ---------------- BPE tokenizer (GPT-2 byte-level) ----------------
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Bpe:
    def __init__(self):
        self.enc = json.load(open(os.path.join(D, "vocab.json")))
        self.dec = {v: k for k, v in self.enc.items()}
        merges = open(os.path.join(D, "merges.txt"),
                      encoding="utf-8").read().split("\n")[1:]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges) if m}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def _bpe(self, tok):
        word = list(tok)
        while len(word) > 1:
            pairs = [(self.ranks.get((word[i], word[i + 1]), 1 << 30), i)
                     for i in range(len(word) - 1)]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            word = word[:i] + [word[i] + word[i + 1]] + word[i + 2:]
        return word

    def encode(self, text):
        import re
        pat = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+"
            r"| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+")
        ids = []
        for tok in pat.findall(text):
            u = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            ids.extend(self.enc[p] for p in self._bpe(u))
        return ids

    def decode(self, ids):
        u = "".join(self.dec[i] for i in ids)
        return bytes(self.u2b[c] for c in u).decode("utf-8",
                                                    errors="replace")


# ---------------- FP32 forward ----------------
def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                  (x + 0.044715 * x ** 3)))


def forward(W, ids):
    """ids -> logits (T, vocab). Full-context fp32 reference."""
    T = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][:T]
    mask = np.triu(np.full((T, T), -1e10, dtype=np.float32), 1)
    for li in range(N_LAYER):
        p = f"h.{li}."
        h = _ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        hd = D_MODEL // N_HEAD
        q = q.reshape(T, N_HEAD, hd).transpose(1, 0, 2)
        k = k.reshape(T, N_HEAD, hd).transpose(1, 0, 2)
        v = v.reshape(T, N_HEAD, hd).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        att = att - att.max(-1, keepdims=True)
        e = np.exp(att)
        att = e / e.sum(-1, keepdims=True)
        o = (att @ v).transpose(1, 0, 2).reshape(T, D_MODEL)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = _ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h = _gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = _ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return x @ W["wte.weight"].T


def ppl(W, ids, window=1024):
    """Non-overlapping windowed perplexity (ids may exceed n_ctx)."""
    nlls = []
    for s0 in range(0, len(ids) - 1, window):
        chunk = ids[s0:s0 + window]
        if len(chunk) < 2:
            break
        logits = forward(W, chunk)
        lp = logits[:-1] - logits[:-1].max(-1, keepdims=True)
        lse = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        nlls.extend(-lse[np.arange(len(chunk) - 1),
                         np.asarray(chunk[1:])])
    return float(np.exp(np.mean(nlls)))


if __name__ == "__main__":
    W = load_weights()
    bpe = Bpe()
    prompt = "The capital of France is"
    ids = bpe.encode(prompt)
    for _ in range(6):
        nxt = int(np.argmax(forward(W, ids)[-1]))
        ids.append(nxt)
    print("greedy:", repr(bpe.decode(ids)))
    # eval slice: frozen from text8's held-out region (G7's slice; the
    # bar is Q8-vs-FP32 DELTA so any frozen text works — documented)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text8 import splits
    _, _, test = splits()
    raw = "".join(" " if c == 0 else chr(96 + c) for c in test[:12000])
    eids = bpe.encode(raw)[:2048]
    import hashlib
    print("eval slice tokens:", len(eids), "sha:",
          hashlib.sha256(bytes(str(eids), 'utf8')).hexdigest()[:16])
    p = ppl(W, eids)
    print(f"FP32 ppl on frozen slice: {p:.2f}")
    assert np.isfinite(p) and 5 < p < 200, p
