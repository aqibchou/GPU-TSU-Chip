"""Persistent contrastive divergence with fabric semantics (D22/D19) — M8.

The exact mixed-precision pattern the hardware loop will use:
- FP32/64 MASTER weights live host-side; every step they are quantized to
  s1.6.3 raw ints (the fabric's coupling format) and SAMPLING runs on the
  quantized values — gradients update the master, never the quantized copy.
- Wake phase: visibles clamped to a data minibatch, hiddens sampled for
  k sweeps; dream phase: persistent free-running chains, m sweeps. Only
  MOMENT SUMS (<s_i s_j> per edge, <s_i> per node) cross the border (D19).
- Update: dJ = eta*(wake_edge - dream_edge), dh = eta*(wake_node - dream_node),
  with gradient clipping (risk #7's cheap mitigations; ESS-triggered resets
  and temperature jitter arrive with the S5 rig).

Sampling here is the FAST float path (vectorized chromatic Gibbs over chain
batches, sigma(beta*acc) in float64 on quantized couplings) — the science
speed of the Two-Speed Doctrine. Bit-true equivalence against the RTL uses
the same update arithmetic with BitTrueGridSampler-driven moments (M8 rig).
"""
from __future__ import annotations

import numpy as np

from golden.gibbs_grid import Graph, build_schedule


def quantize_s163(x: np.ndarray) -> np.ndarray:
    """Float -> s1.6.3 raw ints (units 1/8), saturating at +/-511."""
    return np.clip(np.rint(x * 8.0), -511, 511).astype(np.int32)


class PcdTrainer:
    def __init__(self, g: Graph, n_visible: int = 0, beta_raw: int = 64,
                 n_chains: int = 64, eta: float = 0.05, clip: float = 0.5,
                 seed: int = 0xD22, visible_idx: np.ndarray | None = None):
        self.g, self.n = g, g.n
        self.vis = (np.arange(n_visible) if visible_idx is None
                    else np.asarray(visible_idx))
        self.beta = beta_raw / 64.0
        self.rng = np.random.default_rng(seed)
        self.edges = [(i, j) for i in range(g.n) for j, _ in g.adj[i] if j > i]
        self.J_master = np.zeros((g.n, g.n))          # dense sym, float64
        self.h_master = np.zeros(g.n)
        self.eta, self.clip = eta, clip
        self.chains = self.rng.integers(0, 2, (n_chains, g.n)).astype(np.float64)
        col = np.array(g.coloring())
        self.color_sites = [np.where(col == c)[0] for c in range(col.max() + 1)]

    # ---------------- quantized sampling (fast float path) ----------------
    def _sweep(self, s: np.ndarray, Jq8: np.ndarray, hq8: np.ndarray,
               clamp_mask: np.ndarray | None, n_sweeps: int) -> np.ndarray:
        for _ in range(n_sweeps):
            for sites in self.color_sites:
                if clamp_mask is not None:
                    sites = sites[~clamp_mask[sites]]
                    if sites.size == 0:
                        continue
                acc = s @ Jq8[:, sites] + hq8[sites]
                p = 1.0 / (1.0 + np.exp(-np.clip(self.beta * acc, -35, 35)))
                s[:, sites] = (self.rng.random((s.shape[0], sites.size)) < p)
        return s

    def _moments(self, s: np.ndarray):
        m1 = s.mean(axis=0)
        m2 = np.array([(s[:, i] * s[:, j]).mean() for i, j in self.edges])
        return m1, m2

    def quantized(self):
        return quantize_s163(self.J_master) / 8.0, quantize_s163(self.h_master) / 8.0

    def enable_adam(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        """Host-side optimizer choice (free under D22 — masters live host-side;
        hardware semantics untouched). Amendment D-012: plain clipped SGD
        undertrained the full-scale S5 model ~4-10x vs the adam-trained
        reference; adam matches the reference recipe's per-parameter scaling."""
        self.adam = {"lr": lr, "b1": b1, "b2": b2, "eps": eps, "t": 0,
                     "mJ": np.zeros(len(self.edges)), "vJ": np.zeros(len(self.edges)),
                     "mh": np.zeros(self.n), "vh": np.zeros(self.n)}

    def apply_update(self, w1, w2, d1, d2):
        """Master update from wake/dream moments — the ONE implementation of
        the update arithmetic; the RTL equivalence rig calls this with
        hardware-produced moments so golden and RTL share it exactly."""
        gh = np.clip(w1 - d1, -self.clip, self.clip)
        gJ = np.clip(w2 - d2, -self.clip, self.clip)
        if getattr(self, "adam", None):
            a = self.adam
            a["t"] += 1
            a["mh"] = a["b1"] * a["mh"] + (1 - a["b1"]) * gh
            a["vh"] = a["b2"] * a["vh"] + (1 - a["b2"]) * gh * gh
            a["mJ"] = a["b1"] * a["mJ"] + (1 - a["b1"]) * gJ
            a["vJ"] = a["b2"] * a["vJ"] + (1 - a["b2"]) * gJ * gJ
            bc1 = 1 - a["b1"] ** a["t"]
            bc2 = 1 - a["b2"] ** a["t"]
            uh = a["lr"] * (a["mh"] / bc1) / (np.sqrt(a["vh"] / bc2) + a["eps"])
            uJ = a["lr"] * (a["mJ"] / bc1) / (np.sqrt(a["vJ"] / bc2) + a["eps"])
        else:
            uh, uJ = self.eta * gh, self.eta * gJ
        self.h_master += uh
        for (i, j), uij in zip(self.edges, uJ):
            self.J_master[i, j] += uij
            self.J_master[j, i] = self.J_master[i, j]
        return {"grad_norm": float(np.abs(gJ).max())}

    def step(self, data_batch: np.ndarray, k_wake: int = 5, m_dream: int = 5):
        """One PCD update. data_batch: (B, len(vis)) in {0,1}."""
        Jq8, hq8 = self.quantized()                   # sample on quantized
        B = data_batch.shape[0]
        wake = self.rng.integers(0, 2, (B, self.n)).astype(np.float64)
        wake[:, self.vis] = data_batch
        clamp = np.zeros(self.n, dtype=bool)
        clamp[self.vis] = True
        if not clamp.all():
            wake = self._sweep(wake, Jq8, hq8, clamp, k_wake)
        w1, w2 = self._moments(wake)
        # dream: persistent chains, free
        self.chains = self._sweep(self.chains, Jq8, hq8, None, m_dream)
        d1, d2 = self._moments(self.chains)
        return self.apply_update(w1, w2, d1, d2)


# ---------------- self-check: recover a known distribution ----------------
def _exact_pi(J8: np.ndarray, h8: np.ndarray, beta: float, n: int):
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
    e = 0.5 * np.einsum("xi,ij,xj->x", bits, J8, bits) + bits @ h8
    w = np.exp(beta * (e - e.max()))
    return w / w.sum(), bits


def self_check():
    n = 8
    g = Graph(n)
    rng = np.random.default_rng(7)
    true_J = np.zeros((n, n))
    for i in range(n):                                # ring, degree 2
        j = (i + 1) % n
        v = float(rng.uniform(-1.5, 1.5))
        g.add_edge(i, j, int(round(v * 8)))
        true_J[i, j] = true_J[j, i] = v
    true_h = rng.uniform(-0.5, 0.5, n)
    beta = 1.0
    pi_true, bits = _exact_pi(true_J, true_h, beta, n)
    data = bits[np.random.default_rng(8).choice(1 << n, 4000, p=pi_true)][:, :n]

    tr = PcdTrainer(g, n_visible=n, beta_raw=64, n_chains=128, eta=0.08)

    def kl_now():
        Jq8 = quantize_s163(tr.J_master) / 8.0
        hq8 = quantize_s163(tr.h_master) / 8.0
        pi_m, _ = _exact_pi(Jq8, hq8, beta, n)
        return float(np.sum(pi_true * np.log((pi_true + 1e-30) / (pi_m + 1e-30))))

    kl0 = kl_now()
    for step in range(300):
        batch = data[tr.rng.choice(len(data), 64)]
        tr.step(batch, k_wake=1, m_dream=3)
    kl1 = kl_now()
    assert kl1 < 0.08 and kl1 < 0.25 * kl0, (kl0, kl1)
    print(f"pcd self-check OK — KL(true||model): {kl0:.3f} -> {kl1:.4f} "
          f"(visible-only ring, quantized sampling, FP master)")


if __name__ == "__main__":
    self_check()
