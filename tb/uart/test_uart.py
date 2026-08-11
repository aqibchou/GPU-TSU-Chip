"""UART bench — Σ.3 per-transaction compare for a handshaked unit, with
nothing about the timing left unchecked:

TX: random bytes pushed through the tx_valid/tx_ready handshake with random
    idle gaps; txd is sampled EVERY cycle and the full trace is run through
    golden decode_strict, which enforces exact 8N1 framing and per-bit line
    stability — so the transaction compare is cycle-exact in effect.
RX: golden encode() renders bytes into an exact per-cycle waveform (random
    inter-frame gaps); the bench drives it cycle by cycle and the sequence of
    rx_valid pulses must reproduce the byte list exactly — no misses, no
    spurious pulses.

FUZZ_N counts transactions (frames), not cycles: one frame = 10 bit periods.
"""
import cocotb

from golden.uart import decode_strict, encode
from mkutil import (Rng, check, drive_edge, get_nvec, get_seed, reset_n,
                    sample_edge, start_clock)

TB = "uart"
CPB = 4  # must match rtl/common/uart.sv CLKS_PER_BIT default


async def run(dut, seed: int, n_frames: int, target: str):
    rng = Rng(seed)
    start_clock(dut)
    dut.rxd.value = 1
    await reset_n(dut, ("tx_valid", "tx_data"))

    n_tx = n_frames // 2
    n_rx = n_frames - n_tx

    # ---------------- TX half ----------------
    tx_bytes = [rng.bits(8) for _ in range(n_tx)]
    trace = []
    sent = 0
    pending = None          # byte currently offered on tx_valid
    idle = 0
    # run until all bytes sent plus drain: frame = 10*CPB cycles + slack
    budget = (n_tx + 2) * 12 * CPB + 100
    for cyc in range(budget):
        await drive_edge(dut)
        ready = int(dut.tx_ready.value)
        if pending is None and sent < n_tx and idle == 0:
            pending = tx_bytes[sent]
            dut.tx_valid.value = 1
            dut.tx_data.value = pending
        elif pending is None:
            dut.tx_valid.value = 0
            if idle > 0:
                idle -= 1
        if pending is not None and ready:
            # accepted at the coming edge; drop valid after it
            pass
        await sample_edge(dut)
        trace.append(int(dut.txd.value))
        if pending is not None and ready:
            sent += 1
            pending = None
            idle = rng.randrange(0, 8)
            # deassert at next drive point
            await drive_edge(dut)
            dut.tx_valid.value = 0
            await sample_edge(dut)
            trace.append(int(dut.txd.value))
        if sent == n_tx and int(dut.tx_ready.value) and cyc > 12 * CPB:
            break
    # drain a few idle cycles
    for _ in range(3 * CPB):
        await sample_edge(dut)
        trace.append(int(dut.txd.value))

    got = decode_strict(trace, CPB)
    check(got == tx_bytes,
          f"TX decode mismatch: got {len(got)} frames {[hex(x) for x in got[:8]]}… "
          f"want {len(tx_bytes)} {[hex(x) for x in tx_bytes[:8]]}…",
          0, seed, TB, target)

    # ---------------- RX half ----------------
    rx_bytes = [rng.bits(8) for _ in range(n_rx)]
    gaps = [rng.randrange(0, 3 * CPB) for _ in range(n_rx)]
    wave = encode(rx_bytes, gaps, CPB)
    got_rx = []
    for bit in wave:
        await drive_edge(dut)
        dut.rxd.value = bit
        await sample_edge(dut)
        if int(dut.rx_valid.value):
            got_rx.append(int(dut.rx_data.value))
    # settle: no trailing pulses expected
    for _ in range(2 * CPB):
        await sample_edge(dut)
        check(int(dut.rx_valid.value) == 0, "spurious rx_valid after idle",
              len(wave), seed, TB, target)

    check(got_rx == rx_bytes,
          f"RX mismatch: got {len(got_rx)} {[hex(x) for x in got_rx[:8]]}… "
          f"want {len(rx_bytes)} {[hex(x) for x in rx_bytes[:8]]}…",
          0, seed, TB, target)


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(20), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(400), "fuzz")
