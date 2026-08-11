"""xoshiro128++ golden model + jump-ahead stream seeding (D16).

The farm gives every p-bit its own stream: stream i's initial state is the
mother state advanced by i * 2^64 steps. Rather than trusting published jump
constants, the jump is computed here from first principles: the xoshiro128
state update is linear over GF(2), so we build the 128x128 transition matrix
M directly from step() (feeding basis vectors through it), raise it to 2^64
by repeated squaring, and apply it. self_check() proves M reproduces step()
on random states — the jump inherits correctness by construction. Streams
2^64 apart in a 2^128-1 cycle cannot overlap for any realistic run length.

The ++ output scrambler (rotl(s0+s3,7)+s0) is NOT linear — only the state
update is, which is all the jump needs.

Seeding: splitmix32 expansion of a 32-bit project seed (never all-zero).
Vectorized multi-stream generation (numpy) powers the S1 correlation battery.
"""
from __future__ import annotations

import numpy as np

M32 = 0xFFFFFFFF


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & M32


def splitmix32(seed: int):
    """Standard splitmix32 stream — used only to expand seeds."""
    s = seed & M32
    while True:
        s = (s + 0x9E3779B9) & M32
        z = s
        z = ((z ^ (z >> 16)) * 0x21F0AAAD) & M32
        z = ((z ^ (z >> 15)) * 0x735A2D97) & M32
        z ^= z >> 15
        yield z


def seed_state(seed: int) -> list[int]:
    g = splitmix32(seed)
    while True:
        st = [next(g) for _ in range(4)]
        if any(st):
            return st


def step_state(s: list[int]) -> list[int]:
    """One xoshiro128 state transition (linear over GF(2))."""
    s0, s1, s2, s3 = s
    t = (s1 << 9) & M32
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 11)
    return [s0, s1, s2, s3]


def output_pp(s: list[int]) -> int:
    """xoshiro128++ scrambler on the CURRENT state (pre-update)."""
    return (_rotl((s[0] + s[3]) & M32, 7) + s[0]) & M32


class Xoshiro128pp:
    def __init__(self, state: list[int]):
        assert any(state), "all-zero state is the forbidden fixed point"
        self.s = list(state)

    def next(self) -> int:
        r = output_pp(self.s)
        self.s = step_state(self.s)
        return r


# ---------------- GF(2) matrix jump ----------------
def _state_to_int(s: list[int]) -> int:
    return s[0] | (s[1] << 32) | (s[2] << 64) | (s[3] << 96)


def _int_to_state(v: int) -> list[int]:
    return [(v >> (32 * i)) & M32 for i in range(4)]


def transition_matrix() -> list[int]:
    """Rows of the 128x128 GF(2) matrix M with next_state = M @ state.
    Row r (an int bitmask over input bits) gives output bit r."""
    cols = []
    for b in range(128):
        cols.append(_state_to_int(step_state(_int_to_state(1 << b))))
    rows = []
    for r in range(128):
        m = 0
        for b in range(128):
            if (cols[b] >> r) & 1:
                m |= 1 << b
        rows.append(m)
    return rows


def mat_vec(rows: list[int], v: int) -> int:
    out = 0
    for r in range(128):
        out |= (bin(rows[r] & v).count("1") & 1) << r
    return out


def mat_mul(a: list[int], b: list[int]) -> list[int]:
    """a @ b over GF(2), rows-as-bitmask representation."""
    bt = []  # columns of b as ints = rows of b^T
    for c in range(128):
        m = 0
        for r in range(128):
            if (b[r] >> c) & 1:
                m |= 1 << r
        bt.append(m)
    return [sum(((bin(a[r] & bt[c]).count("1") & 1) << c) for c in range(128))
            for r in range(128)]


def mat_pow2k(rows: list[int], k: int) -> list[int]:
    """M^(2^k) by k squarings."""
    m = rows
    for _ in range(k):
        m = mat_mul(m, m)
    return m


_M_JUMP64 = None


def jump64_matrix() -> list[int]:
    global _M_JUMP64
    if _M_JUMP64 is None:
        _M_JUMP64 = mat_pow2k(transition_matrix(), 64)
    return _M_JUMP64


def stream_states(seed: int, nstreams: int) -> list[list[int]]:
    """Initial states for nstreams streams spaced 2^64 apart."""
    mj = jump64_matrix()
    v = _state_to_int(seed_state(seed))
    out = []
    for _ in range(nstreams):
        st = _int_to_state(v)
        assert any(st)
        out.append(st)
        v = mat_vec(mj, v)
    return out


# ---------------- vectorized multi-stream (for the S1 battery) ----------------
class FarmVec:
    """All streams stepped together; words[i] = stream i's next output."""

    def __init__(self, states: list[list[int]]):
        a = np.array(states, dtype=np.uint64)      # (n, 4)
        self.s = [a[:, i].astype(np.uint32) for i in range(4)]

    def next_block(self, nwords: int) -> np.ndarray:
        """Returns (nstreams, nwords) uint32."""
        n = self.s[0].shape[0]
        out = np.empty((n, nwords), dtype=np.uint32)
        s0, s1, s2, s3 = self.s
        for w in range(nwords):
            tmp = (s0 + s3)
            out[:, w] = ((tmp << np.uint32(7)) | (tmp >> np.uint32(25))) + s0
            t = s1 << np.uint32(9)
            s2 = s2 ^ s0
            s3 = s3 ^ s1
            s1 = s1 ^ s2
            s0 = s0 ^ s3
            s2 = s2 ^ t
            s3 = (s3 << np.uint32(11)) | (s3 >> np.uint32(21))
        self.s = [s0, s1, s2, s3]
        return out


def self_check() -> None:
    import random
    rng = random.Random(1)
    # matrix reproduces step() on random states
    m = transition_matrix()
    for _ in range(20):
        st = [rng.getrandbits(32) for _ in range(4)]
        assert mat_vec(m, _state_to_int(st)) == _state_to_int(step_state(st))
    # jump64 == 2^16 steps when checked against M^(2^16) stepping (cheap proxy):
    # verify M^(2^4) equals 16 explicit steps, then trust squaring chain
    m16 = mat_pow2k(m, 4)
    st = [rng.getrandbits(32) for _ in range(4)]
    st16 = list(st)
    for _ in range(16):
        st16 = step_state(st16)
    assert mat_vec(m16, _state_to_int(st)) == _state_to_int(st16)
    # streams distinct and non-zero
    states = stream_states(0xC0FFEE, 8)
    firsts = [Xoshiro128pp(s).next() for s in states]
    assert len(set(firsts)) == 8
    # vectorized == scalar
    fv = FarmVec(states)
    blk = fv.next_block(64)
    for i, s in enumerate(states):
        g = Xoshiro128pp(list(s))
        assert [g.next() for _ in range(64)] == list(map(int, blk[i])), f"stream {i}"
    print("xoshiro golden self-check OK (matrix jump verified against step)")


if __name__ == "__main__":
    self_check()
