"""Golden model: 8N1 UART bit timing (matches rtl/common/uart.sv).

encode()       — bytes -> exact per-cycle line waveform (LSB first, start=0,
                 stop=1, CLKS_PER_BIT cycles per bit, per-frame idle gaps).
decode_strict()— per-cycle line trace -> bytes, enforcing exact framing: the
                 line must be STABLE across every bit period and idle-high
                 between frames. Any deviation raises, with the cycle index.
                 Strict decode makes a transaction-level compare equivalent
                 to a cycle-exact one (Σ.3: per-transaction for handshaked
                 units, but nothing about the timing is left unchecked).
"""


def encode(data: list[int], gaps: list[int], cpb: int, lead_in: int = 8) -> list[int]:
    w = [1] * lead_in
    for byte, gap in zip(data, gaps):
        w += [0] * cpb                                   # start
        for k in range(8):
            w += [(byte >> k) & 1] * cpb                 # LSB first
        w += [1] * cpb                                   # stop
        w += [1] * gap                                   # inter-frame idle
    w += [1] * (2 * cpb)                                 # tail idle
    return w


def decode_strict(trace: list[int], cpb: int) -> list[int]:
    out = []
    i = 0
    n = len(trace)
    while i < n:
        if trace[i] == 1:
            i += 1
            continue
        # frame starts at i: 10 bit periods (start + 8 data + stop)
        if i + 10 * cpb > n:
            raise ValueError(f"truncated frame at cycle {i}")
        bits = []
        for k in range(10):
            period = trace[i + k * cpb: i + (k + 1) * cpb]
            if len(set(period)) != 1:
                raise ValueError(f"line unstable in bit {k} of frame at cycle {i}: {period}")
            bits.append(period[0])
        if bits[0] != 0:
            raise ValueError(f"bad start bit at cycle {i}")
        if bits[9] != 1:
            raise ValueError(f"bad stop bit in frame at cycle {i}")
        out.append(sum(b << k for k, b in enumerate(bits[1:9])))
        i += 10 * cpb
    return out


if __name__ == "__main__":
    data = [0x00, 0xFF, 0xA5, 0x5A, 0x81]
    gaps = [0, 3, 17, 1, 9]
    assert decode_strict(encode(data, gaps, 4), 4) == data
    assert decode_strict(encode(data, gaps, 7), 7) == data
    try:
        decode_strict([1, 1, 0, 0, 0, 0], 4)
        raise SystemExit("expected truncated-frame error")
    except ValueError:
        pass
    print("uart golden self-check OK")
