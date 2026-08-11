"""Chromatic Gibbs golden stack for the 4x4 king-graph fabric (D14/D17, M5).

Frozen fabric-v1 semantics (mirrored bit-for-bit by rtl/pbit/fabric16.sv):
- 16 sites on a 4x4 king graph; slot order per site: (-1,-1),(-1,0),(-1,1),
  (0,-1),(0,1),(1,-1),(1,0),(1,1); color(site) = (row&1)*2 + (col&1) — a
  proper 4-coloring, so same-color sites are never adjacent.
- binary units s in {0,1}; acc_i = h_i + sum_j J_ij * s_j in s1.6.3 raw units;
  P(s_i=1 | rest) = p17(acc_i, beta)/65536, bit-true per golden/pbit.py.
- one sweep = colors 0,1,2,3 in order; all sites of the active color sample
  simultaneously; every PRNG stream advances once per color phase and site i
  consumes its word only on phase color(i); clamped sites consume normally
  but never update.

Three oracles (D17):
- pi_ideal:  exact float64 Boltzmann  pi ∝ exp(beta*(sum_{i<j} J s s + sum h s))
- pi_hw:     EXACT stationary distribution of the bit-true sweep kernel, by
  power iteration. Per color the kernel factorizes: a site's conditional
  depends only on non-color bits, so applying one color phase to a
  distribution is: group-sum over the color bits, then multiply the product
  of per-site Bernoulli factors. Exact up to float64 + convergence tol.
- BitTrueSampler: trajectory-exact chain (same xoshiro words as RTL).

TVD(pi_hw, pi_ideal) is the D15 precision measurement S2 reports.
"""
from __future__ import annotations

import numpy as np

from golden.pbit import p17
from golden.xoshiro import FarmVec, stream_states

N = 16
NSTATES = 1 << N
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def site(r: int, c: int) -> int:
    return r * 4 + c


def neighbors(i: int):
    r, c = divmod(i, 4)
    out = []
    for k, (dr, dc) in enumerate(OFFS):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 4 and 0 <= cc < 4:
            out.append((k, site(rr, cc)))
    return out


def color(i: int) -> int:
    r, c = divmod(i, 4)
    return (r & 1) * 2 + (c & 1)


COLOR_SITES = [[i for i in range(N) if color(i) == c] for c in range(4)]


def make_instance(seed: int, jmax: int = 16, hmax: int = 8):
    """Dense symmetric J (raw s1.6.3 ints), h raw. J only on king edges."""
    rng = np.random.default_rng(seed)
    J = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for _, j in neighbors(i):
            if j > i:
                v = int(rng.integers(-jmax, jmax + 1))
                J[i, j] = J[j, i] = v
    h = rng.integers(-hmax, hmax + 1, size=N).astype(np.int32)
    return J, h


# ---------------- exact references ----------------
_BITS = ((np.arange(NSTATES, dtype=np.uint32)[:, None]
          >> np.arange(N, dtype=np.uint32)) & 1).astype(np.int8)


def acc_table(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """acc_all[x, i] = h_i + sum_j J_ij * bit_j(x), int32 raw units."""
    return (_BITS.astype(np.int32) @ J.T) + h[None, :]


def p17_table(acc_all: np.ndarray, beta_raw: int) -> np.ndarray:
    lo, hi = int(acc_all.min()), int(acc_all.max())
    tab = np.array([p17(a, beta_raw) for a in range(lo, hi + 1)], dtype=np.float64)
    return tab[acc_all - lo] / 65536.0          # q_all[x, i] = P(site i -> 1)


def pi_ideal(J: np.ndarray, h: np.ndarray, beta_raw: int) -> np.ndarray:
    beta = beta_raw / 64.0
    s = _BITS.astype(np.float64)
    e = 0.5 * np.einsum("xi,ij,xj->x", s, J.astype(np.float64) / 8.0, s) \
        + s @ (h.astype(np.float64) / 8.0)
    w = np.exp(beta * (e - e.max()))
    return w / w.sum()


def sweep_kernel_apply(pi: np.ndarray, q_all: np.ndarray) -> np.ndarray:
    """One full sweep (colors 0..3) applied exactly to distribution pi."""
    x = np.arange(NSTATES, dtype=np.int64)
    for c in range(4):
        cmask = 0
        for i in COLOR_SITES[c]:
            cmask |= 1 << i
        m = x & ~cmask                            # non-color context id
        mass = np.zeros(NSTATES)
        np.add.at(mass, m, pi)                    # group-sum over color bits
        prod = np.ones(NSTATES)
        for i in COLOR_SITES[c]:
            qi = q_all[m, i]                      # depends on context only
            bit = (x >> i) & 1
            prod *= np.where(bit == 1, qi, 1.0 - qi)
        pi = mass[m] * prod
    return pi


def pi_hw(J, h, beta_raw, tol=1e-13, max_sweeps=20000):
    q_all = p17_table(acc_table(J, h), beta_raw)
    pi = np.full(NSTATES, 1.0 / NSTATES)
    for it in range(max_sweeps):
        nxt = sweep_kernel_apply(pi, q_all)
        d = np.abs(nxt - pi).sum()
        pi = nxt
        if d < tol:
            return pi, it + 1
    return pi, max_sweeps


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


# ---------------- bit-true trajectory sampler ----------------
class BitTrueSampler:
    """State-for-state identical to fabric16.sv given the same seeds."""

    def __init__(self, J, h, beta_raw, seed, state0=0, clamp=0):
        self.J, self.h, self.beta = J, h, beta_raw
        self.state = state0
        self.clamp = clamp
        self.fv = FarmVec(stream_states(seed, N))

    def sweep(self):
        st = self.state
        for c in range(4):
            words = self.fv.next_block(1)[:, 0]   # every stream advances
            for i in COLOR_SITES[c]:
                if (self.clamp >> i) & 1:
                    continue
                acc = int(self.h[i])
                for _, j in neighbors(i):
                    if (st >> j) & 1:
                        acc += int(self.J[i, j])
                p = p17(acc, self.beta)
                bit = int((int(words[i]) >> 16) < p)
                st = (st & ~(1 << i)) | (bit << i)
        self.state = st
        return st


def self_check():
    J, h = make_instance(1234)
    beta = 32
    # kernel preserves total probability; power iteration converges
    pi, iters = pi_hw(J, h, beta)
    assert abs(pi.sum() - 1.0) < 1e-9 and iters < 20000
    # detailed-balance-free sanity: pi_hw close to pi_ideal at this precision
    d = tvd(pi, pi_ideal(J, h, beta))
    assert d < 0.02, d
    # sampler marginals match pi_hw marginals (n=20k, moment z-test)
    smp = BitTrueSampler(J, h, beta, seed=0xC0FFEE)
    for _ in range(200):
        smp.sweep()
    n = 20000
    counts = np.zeros(N)
    for _ in range(n):
        s = smp.sweep()
        for i in range(N):
            counts[i] += (s >> i) & 1
    marg_hw = (_BITS.astype(np.float64) * pi[:, None]).sum(axis=0)
    z = (counts / n - marg_hw) / np.sqrt(marg_hw * (1 - marg_hw) / n)
    zmax = float(np.abs(z).max())
    # crude bound: 16 correlated marginals, allow 5 sigma
    assert zmax < 5.0, (zmax, "sampler vs pi_hw marginals")
    print(f"gibbs self-check OK — pi_hw in {iters} sweeps, "
          f"TVD(hw, ideal)={d:.2e}, sampler zmax={zmax:.2f}")


if __name__ == "__main__":
    self_check()
