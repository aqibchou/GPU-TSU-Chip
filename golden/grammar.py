#!/usr/bin/env python3
"""M11: 8-symbol toy regular grammar — exact-LL testbed for trusting
hardware LEARNING (T2).

Grammar (FROZEN, seed 0xB0B): 4-state automaton over Sigma={0..7}; each
state allows 3 symbols with random (normalized) probabilities and random
next-states. Windows of L=4 consecutive symbols from the stationary chain
are the data; their exact distribution is computable by the forward
algorithm, and every model likelihood below is EXACT (12 visible bits ->
p(v) sums 2^8 hidden states; Z enumerates 2^20).

Model: 12 visible + 8 hidden on a bipartite deg<=8 graph (fabric-legal,
2-chromatic). Two trainers, identical recipe (adam 0.01, k_wake 5,
m_dream 5, B=32, C=32 chains — the D-012 amended recipe):
  reference : float Gibbs PCD (the software-reference science lane)
  fabric    : golden/pcd.PcdTrainer — bit-true s1.6.3 quantized sampling,
              the exact form the RTL rig reproduces state-identically
T2 compares held-out exact LL of the two trained models (bar lives in
gates/t2_grammar.py, frozen there).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gibbs_grid import Graph  # noqa: E402
from pcd import PcdTrainer  # noqa: E402

GRAMMAR_SEED = 0xB0B
N_STATES, N_SYM, L = 4, 8, 4
N_VIS, N_HID = 12, 8
N = N_VIS + N_HID


def make_grammar(seed=GRAMMAR_SEED):
    """Frozen automaton: trans[q] = list of (symbol, prob, next_state)."""
    rng = np.random.default_rng(seed)
    trans = []
    for q in range(N_STATES):
        syms = rng.choice(N_SYM, size=3, replace=False)
        probs = rng.dirichlet(np.ones(3) * 2.0)
        nxt = rng.integers(0, N_STATES, size=3)
        trans.append(list(zip(syms.tolist(), probs.tolist(), nxt.tolist())))
    return trans


def state_matrix(trans):
    """P[q, q'] and emission E[q, sym, q'] joint kernels."""
    T = np.zeros((N_STATES, N_SYM, N_STATES))
    for q, rules in enumerate(trans):
        for s, p, qn in rules:
            T[q, s, qn] += p
    return T


def stationary(T):
    P = T.sum(axis=1)
    w, v = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    pi = np.abs(pi) / np.abs(pi).sum()
    return pi


def window_dist(trans):
    """Exact p*(w) over all N_SYM^L windows via the forward algorithm."""
    T = state_matrix(trans)
    pi = stationary(T)
    out = np.zeros(N_SYM ** L)
    for widx in range(N_SYM ** L):
        w = [(widx >> (3 * (L - 1 - i))) & 7 for i in range(L)]
        alpha = pi.copy()
        for s in w:
            alpha = alpha @ T[:, s, :]
        out[widx] = alpha.sum()
    assert abs(out.sum() - 1.0) < 1e-9
    return out


def encode_window(widx):
    """window index -> 12-bit visible vector (3 bits/symbol, MSB first)."""
    return np.array([(widx >> b) & 1 for b in range(3 * L - 1, -1, -1)],
                    dtype=np.float64)


def sample_windows(dist, n, rng):
    idx = rng.choice(len(dist), size=n, p=dist)
    return np.stack([encode_window(i) for i in idx]), idx


def build_model_graph(seed=GRAMMAR_SEED + 1):
    """Bipartite 12v x 8h, hidden degree 8 -> visible degree <= 6."""
    rng = np.random.default_rng(seed)
    g = Graph(N)
    deg = np.zeros(N, int)
    for h in range(N_VIS, N):
        vs = rng.choice(N_VIS, size=8, replace=False)
        for v in vs:
            if deg[v] < 8 and deg[h] < 8:
                g.add_edge(int(v), int(h), 0)
                deg[v] += 1
                deg[h] += 1
    return g


def edge_list(g):
    return [(i, j) for i in range(g.n) for j, _ in g.adj[i] if j > i]


def exact_ll(J, h, beta, test_idx):
    """Exact mean log2-likelihood per window. J dense sym (n,n), h (n,)."""
    states_h = ((np.arange(1 << N_HID)[:, None] >>
                 np.arange(N_HID)[None, :]) & 1).astype(np.float64)
    Jvv = J[:N_VIS, :N_VIS]
    Jvh = J[:N_VIS, N_VIS:]
    Jhh = J[N_VIS:, N_VIS:]
    hv, hh = h[:N_VIS], h[N_VIS:]
    Eh = -0.5 * np.einsum("ri,ij,rj->r", states_h, Jhh, states_h) \
        - states_h @ hh                               # (256,)
    logZ_parts = []
    all_v = ((np.arange(1 << N_VIS)[:, None] >>
              np.arange(N_VIS)[None, :]) & 1).astype(np.float64)
    Ev = -0.5 * np.einsum("ri,ij,rj->r", all_v, Jvv, all_v) - all_v @ hv
    cross = all_v @ Jvh                               # (4096, 8)
    # log p(v) = logsumexp_h(-beta E(v,h)) - logZ
    M = (-beta) * (Ev[:, None] + Eh[None, :] - cross @ states_h.T)
    mmax = M.max()
    lse_v = mmax + np.log(np.exp(M - mmax).sum(axis=1))   # (4096,)
    zmax = lse_v.max()
    logZ = zmax + np.log(np.exp(lse_v - zmax).sum())
    # visible vector index -> row in all_v: bit b of row r is (r>>b)&1,
    # encode_window uses MSB-first — map test windows to rows
    rows = []
    for widx in test_idx:
        v = encode_window(widx)
        rows.append(int(sum(int(v[k]) << k for k in range(N_VIS))))
    ll = (lse_v[rows] - logZ) / np.log(2.0)
    return float(ll.mean())


class FloatPcd:
    """Float-Gibbs PCD with the same D-012 recipe — the software lane."""

    def __init__(self, g, seed, lr=0.01):
        self.g, self.rng = g, np.random.default_rng(seed)
        self.edges = edge_list(g)
        self.J = {e: 0.0 for e in self.edges}
        self.h = np.zeros(g.n)
        self.chains = self.rng.integers(0, 2, (32, g.n)).astype(np.float64)
        self.t = 0
        self.m = {k: 0.0 for k in ("h", "J")}
        self.mh = np.zeros(g.n)
        self.vh = np.zeros(g.n)
        self.mJ = {e: 0.0 for e in self.edges}
        self.vJ = {e: 0.0 for e in self.edges}
        self.lr = lr

    def _sweep(self, s):
        for i in np.random.default_rng(
                self.rng.integers(1 << 31)).permutation(self.g.n):
            field = self.h[i] + sum(
                (self.J[(min(i, j), max(i, j))]) * s[:, j]
                for j, _ in self.g.adj[i])
            p = 1.0 / (1.0 + np.exp(-field))
            s[:, i] = (self.rng.random(s.shape[0]) < p).astype(np.float64)

    def step(self, batch, k_wake=5, m_dream=5):
        B = batch.shape[0]
        wake = np.concatenate(
            [batch, self.rng.integers(0, 2, (B, N_HID)).astype(np.float64)],
            axis=1)
        for _ in range(k_wake):
            for i in np.random.default_rng(
                    self.rng.integers(1 << 31)).permutation(N_HID):
                hn = N_VIS + i
                field = self.h[hn] + sum(
                    self.J[(min(hn, j), max(hn, j))] * wake[:, j]
                    for j, _ in self.g.adj[hn])
                p = 1.0 / (1.0 + np.exp(-field))
                wake[:, hn] = (self.rng.random(B) < p).astype(np.float64)
        for _ in range(m_dream):
            self._sweep(self.chains)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        gh = wake.mean(0) - self.chains.mean(0)
        self.mh = b1 * self.mh + (1 - b1) * gh
        self.vh = b2 * self.vh + (1 - b2) * gh ** 2
        self.h += self.lr * (self.mh / (1 - b1 ** self.t)) / (
            np.sqrt(self.vh / (1 - b2 ** self.t)) + eps)
        for e in self.edges:
            i, j = e
            gj = float((wake[:, i] * wake[:, j]).mean()
                       - (self.chains[:, i] * self.chains[:, j]).mean())
            self.mJ[e] = b1 * self.mJ[e] + (1 - b1) * gj
            self.vJ[e] = b2 * self.vJ[e] + (1 - b2) * gj ** 2
            self.J[e] += self.lr * (self.mJ[e] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vJ[e] / (1 - b2 ** self.t)) + eps)
