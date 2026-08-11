"""SV2 exact-diff bench: rtl/tensor/tensor_array.sv vs the bit-frozen
goldens (gemm8 N=1 for W8; golden/quant4.py for W4), plus the frozen
SV2.2 streaming-efficiency bound. Wide face: one 512b beat per cycle
after a burst request; narrow face: the house one-outstanding AckMem.
"""
import os
import sys

import cocotb
import numpy as np

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, MK)

from golden.quant4 import (gemm_verify_w4, gemm_verify_w8, gemv_w4,  # noqa: E402
                           gemv_w8, pack_w4, pad_to, x_image)
from mkutil import drive_edge, get_nvec, get_seed, reset_n, sample_edge, \
    start_clock  # noqa: E402

X_AT, Y_AT, W_AT = 0x1000, 0x8000, 0x100000


class Mem:
    def __init__(self, size=1 << 23):
        self.b = bytearray(size)

    def word(self, addr):
        return int.from_bytes(self.b[addr:addr + 4], "little")

    def store(self, addr, data):
        self.b[addr:addr + 4] = int(data).to_bytes(4, "little")

    def put(self, addr, blob):
        self.b[addr:addr + len(blob)] = blob

    def beat(self, addr):
        return int.from_bytes(self.b[addr:addr + 64], "little")


async def narrow_server(dut, mem):
    while True:
        await sample_edge(dut)
        if int(dut.m_valid.value):
            we = int(dut.m_we.value)
            a = int(dut.m_addr.value)
            wd = int(dut.m_wdata.value) if we else 0
            await drive_edge(dut)
            if we:
                mem.store(a, wd)
            else:
                dut.m_rdata.value = mem.word(a)
            dut.m_ack.value = 1
        else:
            await drive_edge(dut)
            dut.m_ack.value = 0


async def wide_server(dut, mem):
    """One 512b beat per cycle after a request. The engine may pulse
    the NEXT request on the same edge it consumes a last beat (its
    back-to-back prefetch), so every sample checks wr_req — including
    the ones inside a burst."""
    req = None
    while True:
        if req is None:
            await sample_edge(dut)
            if int(dut.wr_req.value):
                req = (int(dut.wr_addr.value), int(dut.wr_beats.value))
                continue
            await drive_edge(dut)
            dut.wb_valid.value = 0
        else:
            a, n = req
            req = None
            for i in range(n):
                await drive_edge(dut)
                dut.wb_valid.value = 1
                dut.wb_data.value = mem.beat(a + 64 * i)
                await sample_edge(dut)
                if int(dut.wr_req.value):
                    req = (int(dut.wr_addr.value),
                           int(dut.wr_beats.value))
            if req is None:
                await drive_edge(dut)
                dut.wb_valid.value = 0


async def boot(dut):
    start_clock(dut)
    mem = Mem()
    await reset_n(dut, zero_signals=("go", "m_ack", "wb_valid"), cycles=4)
    cocotb.start_soon(narrow_server(dut, mem))
    cocotb.start_soon(wide_server(dut, mem))
    return mem


async def gemv(dut, m, k, w4, resident, b=1, timeout=600_000):
    """Issue one T_OP-6/7 command (B = b; op 6 IS b=1); returns busy
    cycle count."""
    await drive_edge(dut)
    dut.t_a.value = W_AT
    dut.t_b.value = X_AT
    dut.t_c.value = Y_AT
    dut.t_m.value = (k << 12) | m
    dut.t_n.value = b - 1
    dut.t_flags.value = (1 if w4 else 0) | (2 if resident else 0)
    dut.go.value = 1
    await drive_edge(dut)
    dut.go.value = 0
    cycles = 0
    for _ in range(timeout):
        await sample_edge(dut)
        cycles += 1
        if not int(dut.busy.value):
            return cycles
    raise AssertionError(f"gemv timeout m={m} k={k} w4={w4} b={b}")


def load_case(mem, rng, m, k, w4):
    x = rng.integers(-128, 128, ((k + 127) // 128) * 128, dtype=np.int64)
    mem.put(X_AT, bytes((int(v) & 0xFF) for v in x))
    if w4:
        w = rng.integers(-8, 8, (m, k), dtype=np.int64)
        mem.put(W_AT, pack_w4(w).tobytes())
        want = gemv_w4(pack_w4(w), x, k)
    else:
        w = rng.integers(-128, 128, (m, k), dtype=np.int64)
        wp = pad_to(w, 64, axis=1).astype(np.int8)
        mem.put(W_AT, wp.tobytes())
        want = gemv_w8(w, x[:k])
    return want


async def run_case(dut, mem, rng, m, k, w4, resident=False):
    want = load_case(mem, rng, m, k, w4)
    cyc = await gemv(dut, m, k, w4, resident)
    got = np.array([mem.word(Y_AT + 4 * i) for i in range(m)],
                   dtype=np.uint32).astype(np.int32)
    assert np.array_equal(got.astype(np.int64),
                          want.astype(np.int32).astype(np.int64)), \
        f"gemv mismatch m={m} k={k} w4={w4} at " \
        f"{np.nonzero(got.astype(np.int64) != want)[0][:4]}"
    return cyc


def load_case_b(mem, rng, m, k, b, w4):
    X = rng.integers(-128, 128, (k, b), dtype=np.int64)
    img, _ = x_image(X, w4)
    mem.put(X_AT, img.tobytes())
    if w4:
        w = rng.integers(-8, 8, (m, k), dtype=np.int64)
        mem.put(W_AT, pack_w4(w).tobytes())
        want = gemm_verify_w4(pack_w4(w), X, k)
    else:
        w = rng.integers(-128, 128, (m, k), dtype=np.int64)
        mem.put(W_AT, pad_to(w, 64, axis=1).astype(np.int8).tobytes())
        want = gemm_verify_w8(w, X)
    return X, want


async def run_case_b(dut, mem, rng, m, k, b, w4, resident=False):
    """T_OP 7 exact-diff: Y[M x B] row-major vs the batched golden."""
    _, want = load_case_b(mem, rng, m, k, b, w4)
    cyc = await gemv(dut, m, k, w4, resident, b=b)
    got = np.array([mem.word(Y_AT + 4 * i) for i in range(m * b)],
                   dtype=np.uint32).astype(np.int32).reshape(m, b)
    assert np.array_equal(got.astype(np.int64),
                          want.astype(np.int32).astype(np.int64)), \
        f"verify mismatch m={m} k={k} b={b} w4={w4} rows " \
        f"{np.nonzero((got.astype(np.int64) != want).any(1))[0][:4]}"
    return cyc


@cocotb.test()
async def smoke(dut):
    """Directed SV2.1 crosses + the x-resident leg."""
    mem = await boot(dut)
    rng = np.random.default_rng(0x54A1)
    for w4 in (False, True):
        for m, k in ((1, 1), (1, 31), (31, 32), (32, 33), (33, 512),
                     (512, 32), (64, 64)):
            await run_case(dut, mem, rng, m, k, w4)
    # x-resident: second command reuses the loaded x with fresh W
    k = 96
    x = rng.integers(-128, 128, 128, dtype=np.int64)
    mem.put(X_AT, bytes((int(v) & 0xFF) for v in x))
    w1 = rng.integers(-128, 128, (40, k), dtype=np.int64)
    mem.put(W_AT, pad_to(w1, 64, axis=1).astype(np.int8).tobytes())
    await gemv(dut, 40, k, False, resident=False)
    w2 = rng.integers(-128, 128, (40, k), dtype=np.int64)
    mem.put(W_AT, pad_to(w2, 64, axis=1).astype(np.int8).tobytes())
    await gemv(dut, 40, k, False, resident=True)
    got = np.array([mem.word(Y_AT + 4 * i) for i in range(40)],
                   dtype=np.uint32).astype(np.int32).astype(np.int64)
    assert np.array_equal(got, gemv_w8(w2, x[:k])), "x-resident leg"


@cocotb.test()
async def smoke_b(dut):
    """T_OP 7 (SV2.5) directed crosses: B in 2..8, both formats, plus
    the X_RESIDENT leg at B>1."""
    mem = await boot(dut)
    rng = np.random.default_rng(0x54A7)
    for w4 in (False, True):
        for m, k, b in ((1, 1, 2), (3, 17, 3), (8, 64, 8), (33, 100, 5),
                        (64, 256, 8), (128, 300, 4), (5, 500, 7)):
            await run_case_b(dut, mem, rng, m, k, b, w4)
    # X_RESIDENT at B=4: second command reuses the loaded X, fresh W
    m, k, b = 24, 96, 4
    X = rng.integers(-128, 128, (k, b), dtype=np.int64)
    img, _ = x_image(X, False)
    mem.put(X_AT, img.tobytes())
    w1 = rng.integers(-128, 128, (m, k), dtype=np.int64)
    mem.put(W_AT, pad_to(w1, 64, axis=1).astype(np.int8).tobytes())
    await gemv(dut, m, k, False, resident=False, b=b)
    w2 = rng.integers(-128, 128, (m, k), dtype=np.int64)
    mem.put(W_AT, pad_to(w2, 64, axis=1).astype(np.int8).tobytes())
    await gemv(dut, m, k, False, resident=True, b=b)
    got = np.array([mem.word(Y_AT + 4 * i) for i in range(m * b)],
                   dtype=np.uint32).astype(np.int32).reshape(m, b)
    assert np.array_equal(got.astype(np.int64), gemm_verify_w8(w2, X)), \
        "X-resident B leg"


@cocotb.test()
async def smoke_eff(dut):
    """SV2.2: total cycles <= 1.10x ideal on 2048x2048, both formats.
    Measured counts land in ci/logs/sv2/eff.json for the SV2.4 gate."""
    import json
    mem = await boot(dut)
    rng = np.random.default_rng(0x54A2)
    m = k = 2048
    rep = {"m": m, "k": k}
    for w4 in (False, True):
        ideal = ((k + 127) // 128 if w4 else (k + 63) // 64) * m
        cyc = await run_case(dut, mem, rng, m, k, w4)
        bound = int(1.10 * ideal)
        print(f"[tensor_array] eff {'W4' if w4 else 'W8'} {m}x{k}: "
              f"{cyc} cycles vs ideal {ideal} (bound {bound})")
        assert cyc <= bound, f"efficiency: {cyc} > {bound}"
        rep["w4" if w4 else "w8"] = {"cycles": cyc, "ideal": ideal,
                                     "eff": ideal / cyc}
    # SV2.5 efficiency legs at the feasible drain-law frontier
    # (spec "T_OP 7 implementation freeze"): beats/row >= 2B
    for tag, w4, kb, bb in (("w8_b4", False, 1024, 4),
                            ("w4_b2", True, 2048, 2)):
        ideal = ((kb + 127) // 128 if w4 else (kb + 63) // 64) * m
        cyc = await run_case_b(dut, mem, rng, m, kb, bb, w4)
        bound = int(1.10 * ideal)
        print(f"[tensor_array] eff {tag} {m}x{kb}: {cyc} cycles vs "
              f"ideal {ideal} (bound {bound})")
        assert cyc <= bound, f"efficiency {tag}: {cyc} > {bound}"
        rep[tag] = {"cycles": cyc, "ideal": ideal, "eff": ideal / cyc}
    out = os.path.join(MK, "ci/logs/sv2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eff.json"), "w") as f:
        json.dump(rep, f, indent=1)


@cocotb.test()
async def fuzz(dut):
    """Seed-keyed random shapes, both formats, B in 1..8 (k <= 300
    keeps every B inside the v1 x-SRAM capacity)."""
    mem = await boot(dut)
    seed = get_seed()
    n = get_nvec(200)
    rng = np.random.default_rng(seed)
    for i in range(n):
        m = int(rng.integers(1, 301))
        k = int(rng.integers(1, 301))
        w4 = bool(rng.random() < 0.5)
        b = int(rng.integers(1, 9))
        if b == 1:
            await run_case(dut, mem, rng, m, k, w4)
        else:
            await run_case_b(dut, mem, rng, m, k, b, w4)
