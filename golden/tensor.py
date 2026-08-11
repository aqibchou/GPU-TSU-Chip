#!/usr/bin/env python3
"""M18 golden: tensor sidecar reference semantics (docs/HARDWARE_ARCHITECTURE.md#tensor-sidecar-and-d4-socket).

gemm8      — the bit-frozen INT8 GEMM (int32 accumulate, no saturation)
lut_*      — the four frozen 256-entry tables + their float references
fused_linear — GEMM8 + bias + GELU-LUT, the G4 reference
write_mems — emit the checked-in .mem files (G3 diffs byte-exact)
"""
import os

import numpy as np

MK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gemm8(A, B, C=None, acc=False):
    """A (M,K) int8, B (K,N) int8 -> C (M,N) int32 exact."""
    A = np.asarray(A, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    R = A.astype(np.int32) @ B.astype(np.int32)
    if acc:
        assert C is not None
        return (np.asarray(C, dtype=np.int32) + R).astype(np.int32)
    return R.astype(np.int32)


# ---------------- LUT tables (frozen scale conventions, spec §3) --------
def _i8_domain():
    return np.arange(-128, 128, dtype=np.int64)


def gelu_table():
    """int8 -> int8: gelu(x/32)*32, round-half-away, clamp int8."""
    x = _i8_domain() / 32.0
    g = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) *
                                 (x + 0.044715 * x ** 3)))
    v = np.sign(g) * np.floor(np.abs(g) * 32.0 + 0.5)
    return np.clip(v, -128, 127).astype(np.int64)


def gelud_table():
    """int8 -> int8: (gelu(x/32) - relu(x/32)) * 512, round-half-away.
    The delta form (M19/D-018): gelu-relu is bounded in [-0.17, 0], so
    q9 output resolution fits int8 — 16x finer than the direct q5 GELU
    table, and the >=4 / <=-4 passthrough becomes exact 0 at the domain
    edges (no piecewise special-casing in kernels)."""
    x = _i8_domain() / 32.0
    g = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) *
                                 (x + 0.044715 * x ** 3)))
    d = g - np.maximum(x, 0.0)
    v = np.sign(d) * np.floor(np.abs(d) * 512.0 + 0.5)
    return np.clip(v, -128, 127).astype(np.int64)


def exp_table():
    """int8 -> uint16: exp(x/16)*256, clamped to 65535."""
    x = _i8_domain() / 16.0
    return np.minimum(np.floor(np.exp(x) * 256.0 + 0.5), 65535
                      ).astype(np.int64)


def sigmoid_table():
    """int8 -> uint16: sigmoid(x/16)*2^15."""
    x = _i8_domain() / 16.0
    return np.minimum(np.floor(1 / (1 + np.exp(-x)) * 32768.0 + 0.5),
                      65535).astype(np.int64)


def rsqrt_table():
    """uint8 -> uint16: 1/sqrt(x/256 + 2^-8) * 2^12."""
    x = np.arange(256, dtype=np.float64) / 256.0 + 1.0 / 256.0
    return np.minimum(np.floor(x ** -0.5 * 4096.0 + 0.5), 65535
                      ).astype(np.int64)


TABLES = {"gelu": (gelu_table, True), "exp": (exp_table, True),
          "sigmoid": (sigmoid_table, True), "rsqrt": (rsqrt_table, False),
          "gelud": (gelud_table, True)}


def lut_apply(name, v):
    """Apply a frozen table to an int8 (or uint8 for rsqrt) vector."""
    tab, signed = TABLES[name][0](), TABLES[name][1]
    v = np.asarray(v)
    idx = (v.astype(np.int64) + 128) if signed else v.astype(np.int64)
    return tab[idx]


def fused_linear(x_q, W_q, bias_q):
    """G4 reference: y = GELU_LUT(sat8((x W + b) >> 8)).
    x_q (K,) int8, W_q (K,N) int8, bias_q (N,) int32."""
    acc = gemm8(x_q[None, :], W_q)[0] + np.asarray(bias_q, np.int32)
    scaled = np.clip(acc >> 8, -128, 127).astype(np.int8)
    return lut_apply("gelu", scaled).astype(np.int8)


def write_mems(outdir=os.path.join(MK, "rtl/tensor")):
    os.makedirs(outdir, exist_ok=True)
    for name, (fn, _s) in TABLES.items():
        tab = fn()
        with open(os.path.join(outdir, f"lut_{name}.mem"), "w") as f:
            f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in tab) + "\n")
    return sorted(os.listdir(outdir))


if __name__ == "__main__":
    # accuracy specs (spec §3, frozen)
    x = _i8_domain()
    g_ref = 0.5 * (x / 32) * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                          ((x / 32) + 0.044715 * (x / 32) ** 3)))
    g_err = np.abs(gelu_table() / 32.0 - g_ref).max()
    assert g_err <= 1.5 / 32, g_err          # <= 1 LSB (+rounding half)
    e_ref = np.exp(x / 16.0) * 256
    # rel-err spec applies where the format can express it:
    # a rounded integer has abs err <= 0.5, so 2^-7 relative
    # precision requires table value >= 64 (spec wording fixed)
    mask = (e_ref >= 64) & (e_ref <= 65535)
    e_rel = (np.abs(exp_table() - e_ref)[mask] / e_ref[mask]).max()
    assert e_rel <= 2 ** -7, e_rel
    s_ref = 1 / (1 + np.exp(-x / 16.0))
    s_err = np.abs(sigmoid_table() / 32768.0 - s_ref).max()
    assert s_err <= 2 ** -8, s_err
    r = np.arange(256) / 256.0 + 1 / 256.0
    r_rel = (np.abs(rsqrt_table() / 4096.0 - r ** -0.5) / r ** -0.5).max()
    assert r_rel <= 2 ** -6, r_rel
    # gelud: table vs float64 delta ref, <= 1 LSB at q9
    xs = np.arange(-128, 128) / 32.0
    gf = 0.5 * xs * (1.0 + np.tanh(np.sqrt(2 / np.pi) *
                                   (xs + 0.044715 * xs ** 3)))
    df = gf - np.maximum(xs, 0.0)
    d_err = np.abs(gelud_table() / 512.0 - df).max()
    assert d_err <= 1.5 / 512, d_err
    assert gelud_table().min() >= -128 and gelud_table().max() <= 0
    # gemm sanity vs float
    rng = np.random.default_rng(5)
    for _ in range(200):
        M, K, N = rng.integers(1, 65, 3)
        A = rng.integers(-128, 128, (M, K), dtype=np.int8)
        B = rng.integers(-128, 128, (K, N), dtype=np.int8)
        assert np.array_equal(gemm8(A, B),
                              A.astype(np.int64) @ B.astype(np.int64))
    files = write_mems()
    print(f"golden tensor self-check OK: gelud_err={d_err:.5f} "
          f"gelu_err={g_err:.4f} "
          f"exp_rel={e_rel:.5f} sig_err={s_err:.6f} rsqrt_rel={r_rel:.5f}; "
          f"200 gemm shapes exact; mems: {files}")
