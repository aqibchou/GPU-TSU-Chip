"""Scaled sparse-graph Gibbs golden stack (M6) — general degree-<=8 graphs.

Frozen fabric_grid.sv contract (bit-true, mirrored here):
- graph: N <= 8192 sites, per-site row = bias + up to 8 slots {nbr_idx, J};
  binary (0/1) or bipolar (+/-1) units; J/bias s1.6.3; conditionals via
  golden/pbit.p17 exactly as the cell computes them.
- coloring: computed HERE (greedy by ascending site index — deterministic),
  loaded as config; proper coloring => same-color sites never adjacent.
- schedule: sites sorted by (color, index) = the order list. Lanes 0..P-1
  process consecutive P-chunks of the active color's segment concurrently;
  every PRNG stream (one per lane) advances once per CHUNK; lane l consumes
  its word only if it holds a real site in that chunk. One sweep = all colors.
- clamped sites are simply excluded from the order list (they never update,
  cost no cycles, consume no words) — NOTE: unlike fabric16, which burns
  words on clamped sites. Different frozen contract, same golden discipline.

Includes: G-set parser, energy/cut, a fast vectorized float annealer (the
"science at golden speed" half of S4), and the bit-true lane sampler for
RTL trajectory equivalence.
"""
from __future__ import annotations

import numpy as np

from golden.pbit import p17
from golden.xoshiro import FarmVec, stream_states


class Graph:
    def __init__(self, n: int):
        self.n = n
        self.adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]  # (nbr, J_raw)

    def add_edge(self, i: int, j: int, j_raw: int):
        assert i != j
        self.adj[i].append((j, j_raw))
        self.adj[j].append((i, j_raw))

    def check_degree(self, dmax=8):
        return max(len(a) for a in self.adj) <= dmax

    def coloring(self) -> list[int]:
        col = [-1] * self.n
        for i in range(self.n):
            used = {col[j] for j, _ in self.adj[i] if col[j] >= 0}
            c = 0
            while c in used:
                c += 1
            col[i] = c
        return col


def torus_king(w: int, h: int, jgen, deg4=True) -> Graph:
    """Torus grid graph; deg4 = von Neumann neighbors, else king (8)."""
    g = Graph(w * h)
    offs = [(0, 1), (1, 0)] if deg4 else [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(h):
        for c in range(w):
            for dr, dc in offs:
                rr, cc = (r + dr) % h, (c + dc) % w
                g.add_edge(r * w + c, rr * w + cc, jgen(r, c, dr, dc))
    return g


def load_gset(path: str) -> Graph:
    with open(path) as f:
        n, _m = map(int, f.readline().split())
        g = Graph(n)
        for ln in f:
            parts = ln.split()
            if len(parts) >= 3:
                i, j, w = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
                g.add_edge(i, j, w * 8)          # weight 1.0 -> raw 8 (s1.6.3)
    return g


def cut_value(g: Graph, m: np.ndarray) -> int:
    """m in {0,1} per site; cut counts raw-weight/8 edges crossing."""
    tot = 0
    for i in range(g.n):
        for j, jr in g.adj[i]:
            if j > i and m[i] != m[j]:
                tot += jr // 8
    return tot


def build_schedule(g: Graph, clamp: np.ndarray | None = None):
    col = g.coloring()
    ncol = max(col) + 1
    order = [i for c in range(ncol)
             for i in range(g.n) if col[i] == c
             and (clamp is None or not clamp[i])]
    bounds = []
    at = 0
    for c in range(ncol):
        cnt = sum(1 for i in order[at:] if col[i] == c)
        bounds.append((at, at + cnt))
        at += cnt
    return col, order, bounds


class BitTrueGridSampler:
    """Trajectory-exact model of fabric_grid.sv."""

    def __init__(self, g: Graph, bias_raw: np.ndarray, beta_raw: int, seed: int,
                 lanes: int = 8, bipolar: bool = False,
                 state0: np.ndarray | None = None,
                 clamp: np.ndarray | None = None):
        self.g, self.h = g, bias_raw
        self.beta = beta_raw
        self.P = lanes
        self.bipolar = bipolar
        self.s = np.zeros(g.n, dtype=np.int8) if state0 is None else state0.copy()
        self.clamp = clamp
        _, self.order, self.bounds = build_schedule(g, clamp)
        self.fv = FarmVec(stream_states(seed, lanes))

    def _acc(self, i: int) -> int:
        acc = int(self.h[i])
        for j, jr in self.g.adj[i]:
            if self.bipolar:
                acc += jr if self.s[j] else -jr
            else:
                acc += jr if self.s[j] else 0
        return acc

    def sweep(self):
        for lo, hi in self.bounds:
            pos = lo
            while pos < hi:
                words = self.fv.next_block(1)[:, 0]
                chunk = self.order[pos:min(pos + self.P, hi)]
                newbits = []
                for lane, site in enumerate(chunk):
                    p = p17(self._acc(site), self.beta)
                    newbits.append((site, int((int(words[lane]) >> 16) < p)))
                for site, b in newbits:          # simultaneous within chunk
                    self.s[site] = b
                pos += self.P
        return self.s


def anneal_science(g: Graph, schedule, restarts: int, seed: int,
                   bipolar=True, couple_sign: float = 1.0) -> list[int]:
    """Vectorized float annealer — golden-speed S4 science half.
    g carries PROBLEM weights; couple_sign=-1 gives the MAX-CUT mapping
    (sample with J = -w, score cuts with +w). Returns cut per restart."""
    rng = np.random.default_rng(seed)
    col = np.array(g.coloring())
    ncol = col.max() + 1
    # dense-ish neighbor arrays for vectorization
    deg = max(len(a) for a in g.adj)
    nbr = np.zeros((g.n, deg), dtype=np.int32)
    jw = np.zeros((g.n, deg), dtype=np.float64)
    for i, a in enumerate(g.adj):
        for k, (j, jr) in enumerate(a):
            nbr[i, k], jw[i, k] = j, couple_sign * jr / 8.0
    cuts = []
    for r in range(restarts):
        m = rng.integers(0, 2, g.n).astype(np.float64)      # 0/1
        for beta, sweeps in schedule:
            for _ in range(sweeps):
                for c in range(ncol):
                    idx = np.where(col == c)[0]
                    sgn = 2.0 * m[nbr[idx]] - 1.0 if bipolar else m[nbr[idx]]
                    acc = (jw[idx] * sgn).sum(axis=1)
                    p = 1.0 / (1.0 + np.exp(-np.clip(beta * acc, -30, 30)))
                    m[idx] = (rng.random(idx.size) < p).astype(np.float64)
        cuts.append(cut_value(g, m.astype(np.int8)))
    return cuts


def self_check():
    # small torus, bit-true sampler vs a brute-force marginal sanity
    g = torus_king(4, 4, lambda r, c, dr, dc: 8, deg4=True)   # J=+1.0 ferro
    assert g.check_degree(4)
    col = g.coloring()
    assert max(col) + 1 == 2                                   # bipartite
    h = np.zeros(g.n, dtype=np.int32)
    smp = BitTrueGridSampler(g, h, beta_raw=32, seed=1, lanes=8)
    states = set()
    for _ in range(300):
        smp.sweep()
        states.add(smp.s.tobytes())
    assert len(states) > 5                                     # actually mixing
    # clamp exclusion: clamped site never changes
    clamp = np.zeros(g.n, dtype=bool)
    clamp[3] = True
    s0 = np.zeros(g.n, dtype=np.int8)
    s0[3] = 1
    smp2 = BitTrueGridSampler(g, h, 32, seed=2, state0=s0, clamp=clamp)
    for _ in range(50):
        smp2.sweep()
        assert smp2.s[3] == 1
    # MAX-CUT mapping: weights +1, couple_sign=-1 -> bipartite torus cuts all
    g2 = torus_king(6, 6, lambda r, c, dr, dc: 8, deg4=True)   # w = +1
    cuts = anneal_science(g2, [(0.5, 200), (2.0, 400), (4.0, 400)], 3, seed=3,
                          couple_sign=-1.0)
    assert max(cuts) == 72, cuts        # bipartite torus: all 72 edges cuttable
    print(f"gibbs_grid self-check OK — 2-coloring, clamp exclusion, "
          f"annealer best cut {max(cuts)}/72")


if __name__ == "__main__":
    self_check()
