"""Independent floating-point anchor for the GPU-TSU stochastic fabric.

The API mirrors pconfig/psample/pdrain and provides an independent semantic
reference for the bit-true sampling-ISA golden model.

Semantics (frozen, matches D13-D19 + the D30 covariance extension):
  sites s in {0,1}^n, replicas R (ensembles are the idle state)
  E(s)   = -0.5 s^T J s - b^T s          (J symmetric, zero diagonal)
  Gibbs  : p(s_i=1 | rest) = sigmoid((J_i . s + b_i) / T)
  updates: chromatic (greedy coloring of the coupling graph); dense graphs
           degrade to per-site sequential scan (still a valid Gibbs kernel)
  clamps : boolean mask + values, honored during sweeps (conditioning)
  drains : 'state' | 'mean' | 'moments' (E[s s^T]) | 'cov'
  ledger : coupling-write and border-byte counters (the D20/D29 tax meters)

Exactness anchors (trust chains end in something exact):
  exact_dist / exact_moments for n <= 16, by enumeration
  quadratic_to_ising verified against enumeration in bayes_head gate
"""
from __future__ import annotations
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def greedy_coloring(adj_mask: np.ndarray) -> list[np.ndarray]:
    """Greedy proper coloring; returns list of index arrays (color classes)."""
    n = adj_mask.shape[0]
    color = -np.ones(n, dtype=int)
    for i in range(n):
        used = {color[j] for j in np.flatnonzero(adj_mask[i]) if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return [np.flatnonzero(color == c) for c in range(color.max() + 1)]


class Fabric:
    def __init__(self, n, n_replicas=64, seed=0xC0FFEE, dense_threshold=16):
        self.n, self.R = int(n), int(n_replicas)
        self.rng = np.random.default_rng(seed)
        self.J = np.zeros((n, n))
        self.b = np.zeros(n)
        self.T = 1.0
        self.S = (self.rng.random((self.R, n)) < 0.5).astype(np.float64)
        self._pattern_hash = None
        self._colors = [np.arange(n)]  # rebuilt on pconfig
        self.dense_threshold = dense_threshold
        # ledger (the tax meters)
        self.coupling_writes = 0
        self.border_bytes = 0
        self.total_sweeps = 0
        self.work = np.zeros(self.R)   # D33: per-replica nonequilibrium work,
        # accumulated as E_new(s)-E_old(s) at every (J,b) parameter write at
        # fixed T. (T-protocols need the beta-scaled formalism; excluded.)
        self.reset_stats()

    # ---------------- ISA surface ----------------
    def pconfig(self, J=None, b=None, T=None):
        track = (J is not None) or (b is not None)
        if track:
            E_old = self.energy()
        if J is not None:
            J = np.asarray(J, dtype=np.float64)
            assert np.allclose(J, J.T), "J must be symmetric"
            self.J = J - np.diag(np.diag(J))
            self.coupling_writes += int(np.count_nonzero(self.J)) // 2
            h = hash(np.count_nonzero(self.J, axis=1).tobytes())
            if h != self._pattern_hash:
                self._pattern_hash = h
                deg = (self.J != 0).sum(1).max() if self.n > 1 else 0
                if deg > self.dense_threshold:
                    self._colors = [np.array([i]) for i in range(self.n)]
                else:
                    self._colors = greedy_coloring(self.J != 0)
        if b is not None:
            self.b = np.asarray(b, dtype=np.float64).copy()
            self.coupling_writes += self.n
        if T is not None:
            self.T = float(T)
        if track:
            self.work += self.energy() - E_old

    def psample(self, sweeps, T=None, clamp_mask=None, clamp_val=None,
                record=False, thin=1, schedule=None):
        """Run Gibbs sweeps on the persistent replica ensemble.
        schedule: optional list of (T, sweeps) pairs (annealing engine)."""
        if schedule is None:
            schedule = [(self.T if T is None else T, sweeps)]
        cm = None if clamp_mask is None else np.asarray(clamp_mask, bool)
        self._cv = None
        if cm is not None:
            cv = np.asarray(clamp_val, np.float64)
            if cv.ndim == 1:                      # one value vector for all
                cv = np.broadcast_to(cv, (self.R, self.n))
            self._cv = cv                          # per-replica conditioning
            self.S[:, cm] = cv[:, cm]
        for Tk, nk in schedule:
            for t in range(int(nk)):
                self._sweep(Tk, cm)
                self.total_sweeps += 1
                if record and (t % thin == 0):
                    self._accumulate()
        return self

    def pdrain(self, mode="state"):
        if mode == "state":
            self.border_bytes += self.S.size
            return self.S.copy()
        if self._cnt == 0:
            self._accumulate()
        m = self._m1 / self._cnt
        if mode == "mean":
            self.border_bytes += self.n * 4
            return m
        M2 = self._m2 / self._cnt
        self.border_bytes += self.n * self.n * 4
        if mode == "moments":
            return M2
        if mode == "cov":
            return M2 - np.outer(m, m)
        raise ValueError(mode)

    # ---------------- internals ----------------
    def _sweep(self, T, cm):
        for C in self._colors:
            z = self.S @ self.J[:, C] + self.b[C]
            p = sigmoid(z / T)
            draw = (self.rng.random((self.R, len(C))) < p).astype(np.float64)
            if cm is not None:
                keep = cm[C]
                if keep.any():
                    draw[:, keep] = self._cv[:, C][:, keep]
            self.S[:, C] = draw

    def _accumulate(self):
        self._m1 += self.S.sum(0)
        self._m2 += self.S.T @ self.S
        self._cnt += self.R

    def reset_stats(self):
        self._m1 = np.zeros(self.n)
        self._m2 = np.zeros((self.n, self.n))
        self._cnt = 0

    def reset_work(self):
        self.work = np.zeros(self.R)

    def resample(self, idx):
        """Multinomial resampling of the replica ensemble (SIMT-side op)."""
        self.S = self.S[np.asarray(idx)].copy()
        self.work = self.work[np.asarray(idx)].copy()

    def reset_state(self, p=0.5):
        self.S = (self.rng.random((self.R, self.n)) < p).astype(np.float64)

    def energy(self, S=None):
        S = self.S if S is None else np.atleast_2d(S)
        return -0.5 * np.einsum("ri,ij,rj->r", S, self.J, S) - S @ self.b


# ---------------- exact oracles (n <= 16) ----------------
def all_states(n):
    idx = np.arange(2 ** n, dtype=np.uint32)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.float64)

def exact_dist(J, b, T=1.0):
    n = len(b)
    assert n <= 16, "enumeration oracle capped at n=16"
    S = all_states(n)
    E = -0.5 * np.einsum("si,ij,sj->s", S, J, S) - S @ b
    w = -(E - E.min()) / T
    p = np.exp(w); p /= p.sum()
    return S, p

def exact_moments(J, b, T=1.0):
    S, p = exact_dist(J, b, T)
    m = p @ S
    M2 = S.T @ (p[:, None] * S)
    return m, M2, M2 - np.outer(m, m)

def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()


def quadratic_to_ising(A, h):
    """Map energy E(sigma) = sigma^T A sigma + h^T sigma over sigma in {-1,+1}^d
    to fabric form E(s) = -0.5 s^T J s - b^T s over s in {0,1}^d (sigma=2s-1),
    equal up to an additive constant. Verified by enumeration in gates."""
    A = np.asarray(A, np.float64); h = np.asarray(h, np.float64)
    A = 0.5 * (A + A.T)
    Aoff = A - np.diag(np.diag(A))
    J = -8.0 * Aoff
    b = 4.0 * Aoff.sum(1) - 2.0 * h
    return J, b


def integrated_act(x, c=5.0):
    """Sokal-style integrated autocorrelation time with automatic windowing.
    Known-answer anchor: AR(1) with coeff phi has tau = (1+phi)/(1-phi)."""
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    n = len(x)
    if x.std() == 0:
        return 1.0
    acf = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    tau = 1.0
    for k in range(1, n // 3):
        tau += 2.0 * acf[k]
        if k >= c * tau:
            break
    return max(tau, 1.0)
