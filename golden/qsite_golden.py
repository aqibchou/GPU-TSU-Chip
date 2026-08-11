"""QSITE golden — the frozen q-ary oracle (sampling ISA §12, fabric-v2).

Arity 2 images belong to the frozen golden/sampling_isa.py and are
refused here. This module is the bit-level contract for the S2-S4 RTL:
given (image sequence, op sequence, SEEDS), trajectories, counters,
and drain words are exact.

Self-check: python golden/qsite_golden.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from xoshiro import FarmVec as _FarmVec, stream_states   # noqa: E402


class FarmVec(_FarmVec):
    """D-014 accounting: counts farm advances for the I-12/q8 rule."""

    def __init__(self, states):
        super().__init__(states)
        self.blocks_drawn = 0

    def next_block(self, nwords):
        self.blocks_drawn += nwords
        return super().next_block(nwords)

MAGIC = 0x4D4B5331
P_LANES = 8
DEG_MAX = 8
ACC_MIN, ACC_MAX = -8192, 8191                       # s14 (I-8/I-11)

# frozen Gumbel table (§12): s3.4, byte-indexed
GLUT = np.array([max(-128, min(127, round(16.0 * (-math.log(
    -math.log((k + 0.5) / 256.0)))))) for k in range(256)], dtype=np.int64)


def _sx(v: int, bits: int) -> int:
    """decode raw two's-complement field of <bits> width."""
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def enc_slot(nbr: int, j_raw: int) -> int:
    return (1 << 23) | ((nbr & 0x1FFF) << 10) | (j_raw & 0x3FF)


def _bias_words(braw, q):
    """braw: list of q-1 raw 10b lanes (states 1..q-1) -> words."""
    ws = []
    for w0 in range(0, q - 1, 3):
        w = 0
        for li, a in enumerate(range(w0, min(w0 + 3, q - 1))):
            w |= (braw[a] & 0x3FF) << (10 * li)
        ws.append(w)
    return ws


def qsite_image(n, q, *, rows=None, order=None, bounds=None, n_colors=0,
                sched=None, seed=None, seeds=None, state=None, rid=None,
                work_track=False):
    """Sections in T_FLAGS bit order; returns (words, t_flags)."""
    words = [MAGIC, (n & 0xFFFF) | ((1 if work_track else 0) << 17),
             (n_colors & 0xFF) | ((len(sched) if sched else 0) << 8)
             | ((max(r for r in rid) + 1 if rid else 1) << 16)
             | (q << 24)]
    flags = 0
    if rows is not None:
        flags |= 1 << 0
        for i in range(n):
            slots, braw = rows[i]
            for k in range(DEG_MAX):
                words.append(slots[k] if k < len(slots) else 0)
            words += _bias_words(braw, q)
    if order is not None:
        flags |= 1 << 1
        words.append(len(order) & 0xFFFF)
        for c in range(16):
            lo, hi = bounds[c] if c < len(bounds) else (0, 0)
            words.append((lo & 0xFFFF) | ((hi & 0xFFFF) << 16))
        words += [s & 0xFFFF for s in order]
    if sched is not None:
        flags |= 1 << 2
        words += [(b & 0xFF) | ((sw & 0xFFFFFF) << 8) for b, sw in sched]
    if seed is not None or seeds is not None:
        flags |= 1 << 3
        st = seeds if seeds is not None else stream_states(seed, P_LANES)
        for s4 in st:
            words += list(s4)
    if state is not None:
        flags |= 1 << 4
        per = 16 if q == 4 else 8
        bw = 2 if q == 4 else 4
        for w0 in range(0, n, per):
            w = 0
            for li, i in enumerate(range(w0, min(w0 + per, n))):
                w |= (int(state[i]) & ((1 << bw) - 1)) << (bw * li)
            words.append(w)
    if rid is not None:
        flags |= 1 << 5
        for w0 in range(0, n, 4):
            w = 0
            for bi, i in enumerate(range(w0, min(w0 + 4, n))):
                w |= (int(rid[i]) & 0xFF) << (8 * bi)
            words.append(w)
    return words, flags


class QsiteGolden:
    """pconfig/psample/pdrain oracle for q in {4, 8} (§12)."""

    def __init__(self):
        self.n = 0

    def _init(self, n, q):
        self.n, self.q = n, q
        self.valid = np.zeros((n, DEG_MAX), dtype=bool)
        self.nbr = np.zeros((n, DEG_MAX), dtype=np.int64)
        self.jraw = np.zeros((n, DEG_MAX), dtype=np.int64)
        self.bias = np.zeros((n, q), dtype=np.int64)     # lane 0 == 0
        self.x = np.zeros(n, dtype=np.int64)
        self.order, self.bounds = [], []
        self.sched = []
        self.rid = np.zeros(n, dtype=np.int64)
        self.n_replicas = 1
        self.fv = None
        self.cnt = 0
        self.m1 = np.zeros((n, q - 1), dtype=np.int64)
        self.m2 = np.zeros((n, DEG_MAX), dtype=np.int64)
        self.work = np.zeros(64, dtype=np.int64)
        self.work_track = False
        self.sweeps_done = 0
        self.upd_cnt = 0
        self.flip_cnt = 0
        self.cfg_words = 0
        self.drain_words = 0

    # -------------- PCONFIG --------------
    def pconfig(self, words, t_flags):
        assert words[0] == MAGIC
        n = words[1] & 0xFFFF
        assert not (words[1] >> 16) & 1, "I-10: BIPOLAR=0 for q>2"
        wt = (words[1] >> 17) & 1
        q = (words[2] >> 24) & 0xFF
        q = 2 if q == 0 else q
        assert q in (4, 8), "arity 2 runs the frozen golden (I-10)"
        if self.n == 0:
            self._init(n, q)
        assert (self.n, self.q) == (n, q), "n/q immutable (v1)"
        # D33 bracket gates on the PRE-image work_track (spec §2/§5:
        # track_pre is latched at command entry) and E2_new runs right
        # after ROWS, BEFORE the STATE/RID sections apply
        wt_pre = self.work_track
        self.work_track = bool(wt)
        self.n_replicas = max(1, (words[2] >> 16) & 0xFF)
        pos = 3
        bias_words = (q - 1 + 2) // 3
        rows_flagged = bool(t_flags & 1)
        e2_old = self._energy2() if rows_flagged and wt_pre else None
        e2_new = None
        for bit in range(6):
            if not (t_flags >> bit) & 1:
                continue
            if bit == 0:                                 # ROWS
                for i in range(n):
                    for k in range(DEG_MAX):
                        w = words[pos + k]
                        self.valid[i, k] = bool(w >> 23)
                        self.nbr[i, k] = (w >> 10) & 0x1FFF
                        self.jraw[i, k] = _sx(w & 0x3FF, 10)
                    for bw in range(bias_words):
                        w = words[pos + DEG_MAX + bw]
                        for li in range(3):
                            a = 1 + 3 * bw + li
                            if a < q:
                                self.bias[i, a] = _sx((w >> (10 * li))
                                                      & 0x3FF, 10)
                    pos += DEG_MAX + bias_words
                self.cfg_words += n * (DEG_MAX + bias_words)
                if e2_old is not None:
                    e2_new = self._energy2()
            elif bit == 1:                               # ORDER
                n_ord = words[pos] & 0xFFFF
                self.bounds = []
                for c in range(16):
                    w = words[pos + 1 + c]
                    self.bounds.append((w & 0xFFFF, (w >> 16) & 0xFFFF))
                self.order = [words[pos + 17 + i] & 0xFFFF
                              for i in range(n_ord)]
                self.cfg_words += 17 + n_ord
                pos += 17 + n_ord
            elif bit == 2:                               # SCHED
                ns = (words[2] >> 8) & 0xFF
                self.sched = [((words[pos + i]) & 0xFF,
                               (words[pos + i] >> 8) & 0xFFFFFF)
                              for i in range(ns)]
                self.cfg_words += ns
                pos += ns
            elif bit == 3:                               # SEEDS
                st = [[words[pos + 4 * s + j] for j in range(4)]
                      for s in range(P_LANES)]
                self.fv = FarmVec(st)
                self.cfg_words += 32
                pos += 32
            elif bit == 4:                               # STATE
                per = 16 if q == 4 else 8
                bw = 2 if q == 4 else 4
                nw = (n + per - 1) // per
                for i in range(n):
                    v = (words[pos + i // per] >> (bw * (i % per))) \
                        & ((1 << bw) - 1)
                    assert v < q, "I-10: state < q"
                    self.x[i] = v
                self.cfg_words += nw
                pos += nw
            elif bit == 5:                               # RID
                nw = (n + 3) // 4
                for i in range(n):
                    r = (words[pos + i // 4] >> (8 * (i % 4))) & 0xFF
                    assert r < self.n_replicas
                    self.rid[i] = r
                self.cfg_words += nw
                pos += nw
        if e2_old is not None:                           # D33 bracket
            for r in range(64):
                self.work[r] += int(e2_new[r]) - int(e2_old[r])
        if (t_flags >> 8) & 1:                           # WORK_RESET
            self.work[:] = 0

    # -------------- energy / accumulate --------------
    def _energy2(self):
        e2 = np.zeros(64, dtype=np.int64)
        for i in range(self.n):
            pair = 0
            for k in range(DEG_MAX):
                if self.valid[i, k] and self.x[self.nbr[i, k]] == self.x[i]:
                    pair += int(self.jraw[i, k])
            e2[self.rid[i]] -= pair + 2 * int(self.bias[i, self.x[i]])
        return e2

    def _accs(self, i):
        acc = self.bias[i].copy()
        for k in range(DEG_MAX):
            if self.valid[i, k]:
                acc[self.x[self.nbr[i, k]]] += self.jraw[i, k]
        assert acc.min() >= ACC_MIN and acc.max() <= ACC_MAX, "I-11"
        return acc

    # -------------- PSAMPLE --------------
    def psample(self, t_flags, t_m=0, t_k=0):
        if (t_flags >> 1) & 1:                           # STATS_RESET
            self.cnt = 0
            self.m1[:] = 0
            self.m2[:] = 0
        entries = [(t_k & 0xFF, t_m)] if (t_flags >> 2) & 1 else self.sched
        for beta, sweeps in entries:
            for _ in range(sweeps):
                self._sweep(beta)
                self.sweeps_done += 1
                if t_flags & 1:                          # RECORD
                    self._accumulate()

    def _sweep(self, beta):
        q = self.q
        for lo, hi in self.bounds:
            pos = lo
            while pos < hi:
                w0 = self.fv.next_block(1)[:, 0]
                w1 = self.fv.next_block(1)[:, 0] if q == 8 else None
                chunk = self.order[pos:min(pos + P_LANES, hi)]
                news = []
                for lane, site in enumerate(chunk):
                    acc = self._accs(site)
                    sc = np.empty(q, dtype=np.int64)
                    for a in range(q):
                        w = int(w0[lane]) if a < 4 else int(w1[lane])
                        byte = (w >> (8 * (a & 3))) & 0xFF
                        sc[a] = beta * int(acc[a]) + (int(GLUT[byte]) << 5)
                    news.append((site, int(np.argmax(sc))))
                for site, v in news:                     # simultaneous
                    self.upd_cnt += 1
                    if v != self.x[site]:
                        self.flip_cnt += 1
                    self.x[site] = v
                pos += P_LANES

    def _accumulate(self):
        self.cnt += 1
        for a in range(1, self.q):
            self.m1[:, a - 1] += (self.x == a)
        for k in range(DEG_MAX):
            hit = self.valid[:, k] & (self.x == self.x[self.nbr[:, k]])
            self.m2[:, k] += hit

    # -------------- PDRAIN --------------
    def pdrain(self, mode):
        q, n = self.q, self.n
        if mode == 0:                                    # STATE-q
            per = 16 if q == 4 else 8
            bw = 2 if q == 4 else 4
            out = np.zeros((n + per - 1) // per, dtype=np.uint32)
            for i in range(n):
                out[i // per] |= np.uint32(int(self.x[i]) << (bw * (i % per)))
        elif mode == 1:                                  # MOMENTS-q
            out = np.zeros(1 + n * (q - 1) + n * DEG_MAX, dtype=np.uint32)
            out[0] = self.cnt
            out[1:1 + n * (q - 1)] = self.m1.reshape(-1)
            out[1 + n * (q - 1):] = self.m2.reshape(-1)
        elif mode == 3:                                  # WORK
            out = np.zeros(1 + 2 * self.n_replicas, dtype=np.uint32)
            out[0] = self.n_replicas
            for r in range(self.n_replicas):
                v = int(self.work[r]) & 0xFFFFFFFFFFFFFFFF
                out[1 + 2 * r] = v & 0xFFFFFFFF
                out[2 + 2 * r] = v >> 32
        elif mode == 4:                                  # TELEMETRY
            out = np.zeros(12, dtype=np.uint32)
            out[0] = self.sweeps_done
            out[1] = self.upd_cnt & 0xFFFFFFFF
            out[2] = self.upd_cnt >> 32
            out[3] = self.flip_cnt & 0xFFFFFFFF
            out[4] = self.flip_cnt >> 32
            out[5] = self.cfg_words
            out[6] = self.drain_words
            out[7] = self.cnt
        else:
            raise AssertionError("COV-q deferred (spec §12)")
        self.drain_words += len(out)
        return out


# ================= self-check =================
def _chain_marginals(h, J, q):
    V = h.shape[0]
    psi = np.exp(h - h.max(axis=1, keepdims=True))
    fwd = np.zeros((V, q))
    fwd[0] = psi[0] / psi[0].sum()
    for v in range(1, V):
        M = np.ones((q, q)) + (np.exp(J[v - 1]) - 1.0) * np.eye(q)
        a = fwd[v - 1] @ M
        fwd[v] = a * psi[v]
        fwd[v] /= fwd[v].sum()
    bwd = np.ones((V, q))
    for v in range(V - 2, -1, -1):
        M = np.ones((q, q)) + (np.exp(J[v]) - 1.0) * np.eye(q)
        b = M @ (psi[v + 1] * bwd[v + 1])
        bwd[v] = b / b.max()
    m = fwd * bwd
    return m / m.sum(axis=1, keepdims=True)


def _mk_chain(n, q, seed, jr=6):
    rng = np.random.default_rng(seed)
    braw = rng.integers(-6, 7, (n, q - 1))
    rows = []
    for i in range(n):
        slots = []
        if i > 0:
            slots.append(enc_slot(i - 1, jr))
        if i < n - 1:
            slots.append(enc_slot(i + 1, jr))
        rows.append((slots, list(braw[i])))
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    bounds = [(0, (n + 1) // 2), ((n + 1) // 2, n)]
    return rows, braw, order, bounds


def self_check():
    # 1) q4 equilibrium vs transfer matrix (quantized-real params)
    n, q, jr = 12, 4, 6
    rows, braw, order, bounds = _mk_chain(n, q, 0xC0DE, jr)
    g = QsiteGolden()
    w, f = qsite_image(n, q, rows=rows, order=order, bounds=bounds,
                       n_colors=2, seed=0xA5A5)
    g.pconfig(w, f)
    g.psample(0b111, t_m=200, t_k=64)                    # burn, reset+rec+imm
    g.psample(0b101, t_m=2000, t_k=64)                   # record 2000 sweeps
    dr = g.pdrain(1)
    cnt = int(dr[0])
    m1 = np.asarray(dr[1:1 + n * (q - 1)], dtype=np.float64)
    m1 = m1.reshape(n, q - 1)
    p = np.zeros((n, q))
    p[:, 1:] = m1 / cnt
    p[:, 0] = 1.0 - p[:, 1:].sum(axis=1)
    h = np.zeros((n, q))
    h[:, 1:] = braw / 8.0
    marg = _chain_marginals(h, np.full(n - 1, jr / 8.0), q)
    tv = 0.5 * np.abs(p - marg).sum(axis=1).mean()
    assert tv < 0.02, f"equilibrium TV {tv:.4f}"
    print(f"  eq q4 chain: TV={tv:.4f} (bar 0.02)")

    # 2) determinism + bring-up vector
    g2 = QsiteGolden()
    g2.pconfig(w, f)
    g2.psample(0b111, t_m=200, t_k=64)
    g2.psample(0b101, t_m=2000, t_k=64)
    t1, t2 = g.pdrain(4), g2.pdrain(4)
    assert np.array_equal(g.x, g2.x)
    assert np.array_equal(t1[:6], t2[:6]) and t1[7] == t2[7], \
        "telemetry mismatch (drain_words excluded by design)"
    print(f"  determinism: state hash {hash(g.x.tobytes()) & 0xFFFFFFFF:08x}")

    # 3) m2 brute-force agreement on a fresh short run
    g3 = QsiteGolden()
    g3.pconfig(w, f)
    ref = np.zeros((n, DEG_MAX), dtype=np.int64)
    c = 0
    for _ in range(50):
        g3.psample(0b101, t_m=1, t_k=64)
        c += 1
        for i in range(n):
            for k in range(DEG_MAX):
                if g3.valid[i, k] and g3.x[i] == g3.x[g3.nbr[i, k]]:
                    ref[i, k] += 1
    d = g3.pdrain(1)
    m2 = np.asarray(d[1 + n * (q - 1):], dtype=np.int64).reshape(n, DEG_MAX)
    assert int(d[0]) == c and np.array_equal(m2, ref), "m2 mismatch"
    print("  m2 agreement counter: exact vs brute force")

    # 4) clamp invariance (site 5 absent from order)
    g4 = QsiteGolden()
    o2 = [s for s in order if s != 5]
    b2 = [(0, 5), (5, 11)]
    st = np.zeros(n, dtype=np.int64)
    st[5] = 3
    w4, f4 = qsite_image(n, q, rows=rows, order=o2, bounds=b2, n_colors=2,
                         seed=0xA5A5, state=st)
    g4.pconfig(w4, f4)
    g4.psample(0b100, t_m=300, t_k=64)
    assert g4.x[5] == 3, "I-6 clamp violated"
    print("  clamp (order exclusion): held")

    # 5) q8 sanity + double-advance rule
    n8 = 16
    rows8, _, order8, bounds8 = _mk_chain(n8, 8, 0xBEEF, jr=4)
    g8 = QsiteGolden()
    w8, f8 = qsite_image(n8, 8, rows=rows8, order=order8, bounds=bounds8,
                         n_colors=2, seed=0x1D)
    g8.pconfig(w8, f8)
    g8.psample(0b100, t_m=10, t_k=64)
    chunks = sum((hi - lo + P_LANES - 1) // P_LANES
                 for lo, hi in bounds8) * 10
    assert g8.fv.blocks_drawn == 2 * chunks, "D-014-q8 double advance"
    assert g8.x.max() < 8
    print(f"  q8: {chunks} chunks, {g8.fv.blocks_drawn} farm draws (2x)")
    print("qsite golden self-check: ALL PASS")


if __name__ == "__main__":
    self_check()
