#!/usr/bin/env python3
"""SV2 W4 golden (docs/SOFTWARE_AND_VALIDATION.md#serving-engine, frozen format).

Packed little-endian nibbles, zero-point 8: w = nibble - 8, range
[-8, +7]. The engine accumulates RAW int32 (no scales anywhere in the
datapath — scaling is model-side at requant, per the v1 gemm8
contract). K pads to 128 nibbles / 64 W8 bytes (one 512b beat), rows pad
likewise.

Self-check: python -m golden.quant4
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pad_to(x, mult, axis, fill=0):
    n = x.shape[axis]
    r = (-n) % mult
    if r == 0:
        return x
    pw = [(0, 0)] * x.ndim
    pw[axis] = (0, r)
    return np.pad(x, pw, constant_values=fill)


def pack_w4(w):
    """(M, K) ints in [-8, 7] -> (M, ceil(K/2)) uint8, little-endian
    nibbles (element k even -> low nibble). K pads to 128 (one 512b
    beat of nibbles) first."""
    w = pad_to(np.asarray(w, np.int64), 128, axis=1)
    assert w.min() >= -8 and w.max() <= 7, "W4 range"
    nib = (w + 8).astype(np.uint8)
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_w4(packed, k):
    """(M, K/2) uint8 -> (M, k) ints in [-8, 7]."""
    lo = (packed & 0xF).astype(np.int64) - 8
    hi = (packed >> 4).astype(np.int64) - 8
    out = np.empty((packed.shape[0], packed.shape[1] * 2), np.int64)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :k]


def gemv_w4(packed, x, k):
    """y[M] int32 = unpack(W)[M x k] . x[k] (x int8), raw accumulate."""
    w = unpack_w4(packed, k)
    x = np.asarray(x, np.int64)[:k]
    y = w @ x
    assert np.abs(y).max() < (1 << 31), "int32 overflow out of contract"
    return y.astype(np.int64)


def gemv_w8(w, x):
    """y[M] int32 = W[M x K] int8 . x[K] int8 (the gemm8 N=1 shape)."""
    w = np.asarray(w, np.int64)
    x = np.asarray(x, np.int64)
    y = w @ x
    assert np.abs(y).max() < (1 << 31)
    return y.astype(np.int64)


def x_image(X, w4):
    """T_OP 7 X image (spec 'T_OP 7 implementation freeze'): X[K x B]
    int8 -> flat COLUMN-MAJOR bytes, each column zero-padded to
    KPAD = ceil(K/kstep)*kstep (kstep 64 W8 / 128 W4). Returns
    (bytes u8 array, kpad). v1 capacity: B*KPAD <= 4096."""
    X = np.asarray(X, np.int64)
    k, b = X.shape
    kstep = 128 if w4 else 64
    kpad = ((k + kstep - 1) // kstep) * kstep
    assert 1 <= b <= 8 and b * kpad <= 4096, "v1 x-SRAM capacity"
    img = np.zeros(b * kpad, np.int64)
    for col in range(b):
        img[col * kpad:col * kpad + k] = X[:, col]
    return img.astype(np.int8).view(np.uint8), kpad


def gemm_verify_w8(w, X):
    """Y[M x B] int32 = W[M x K] int8 . X[K x B] int8 (T_OP 7, row-
    major Y — gemm8's C layout with N=B)."""
    w = np.asarray(w, np.int64)
    X = np.asarray(X, np.int64)
    y = w @ X
    assert np.abs(y).max() < (1 << 31)
    return y.astype(np.int64)


def gemm_verify_w4(packed, X, k):
    """Y[M x B] int32 = unpack(W)[M x k] . X[k x B], raw accumulate."""
    w = unpack_w4(packed, k)
    X = np.asarray(X, np.int64)[:k]
    y = w @ X
    assert np.abs(y).max() < (1 << 31)
    return y.astype(np.int64)


def quantize_w4_rowwise(wf):
    """Model-side helper (M23 uses it; engine never sees scales):
    per-row symmetric scale = max|row|/7, round-half-away, clip."""
    wf = np.asarray(wf, np.float64)
    s = np.abs(wf).max(axis=1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    q = np.clip(np.floor(np.abs(wf) / s + 0.5) * np.sign(wf), -8, 7)
    return q.astype(np.int64), s.squeeze(1)


def self_check():
    rng = np.random.default_rng(0x54A0)
    for m, k in ((1, 1), (3, 31), (16, 64), (33, 129), (64, 2048)):
        w = rng.integers(-8, 8, (m, k))
        x = rng.integers(-128, 128, ((k + 127) // 128) * 128)
        p = pack_w4(w)
        assert np.array_equal(unpack_w4(p, k), w), "pack/unpack"
        want = w @ x[:k].astype(np.int64)
        assert np.array_equal(gemv_w4(p, x, k), want), "gemv_w4"
    wf = rng.standard_normal((8, 256)) * 0.1
    q, s = quantize_w4_rowwise(wf)
    err = np.abs(q * s[:, None] - wf).max()
    assert q.min() >= -8 and q.max() <= 7 and err <= s.max() * 0.5 + 1e-12
    # T_OP 7 batched-verify references + the frozen X image layout
    for m, k, b in ((1, 1, 2), (5, 31, 3), (16, 64, 8), (33, 200, 5),
                    (64, 500, 8)):
        w8 = rng.integers(-128, 128, (m, k))
        w4 = rng.integers(-8, 8, (m, k))
        X = rng.integers(-128, 128, (k, b))
        assert np.array_equal(gemm_verify_w8(w8, X),
                              np.asarray(w8, np.int64) @ X)
        assert np.array_equal(gemm_verify_w4(pack_w4(w4), X, k),
                              np.asarray(w4, np.int64) @ X)
        for is4 in (False, True):
            img, kpad = x_image(X, is4)
            kstep = 128 if is4 else 64
            assert kpad % kstep == 0 and kpad >= k
            cols = img.view(np.int8).reshape(b, kpad)
            assert np.array_equal(cols[:, :k].T, X)
            assert not cols[:, k:].any()
        # B=1 column-major image == the padded gemv x vector
        img1, kp1 = x_image(X[:, :1], False)
        assert np.array_equal(img1.view(np.int8)[:k], X[:, 0])
    print("quant4 self-check: pack/unpack + gemv exact on 5 shapes, "
          "verify refs + x_image layout exact on 5 (m,k,b) shapes, "
          "row-quant sane — ALL OK")


if __name__ == "__main__":
    self_check()
