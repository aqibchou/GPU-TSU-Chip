"""Sampling ISA golden model (M21, S7-sigma) — the bit-frozen oracle for
docs/HARDWARE_ARCHITECTURE.md#sampling-isa. SPEC + GOLDEN ONLY at M21 open; RTL follows the
freeze review.

Implements the S-cluster ops exactly as frozen:
  PCONFIG (T_OP=8): word-image parse (header + ROWS/ORDER/SCHED/SEEDS/STATE/
      RID sections in T_FLAGS bit order); D33 work update on ROWS writes when
      the PRE-command header had WORK_TRACK, using the command-entry state
      and RID map; WORK_RESET after.
  PSAMPLE (T_OP=9): chromatic lane sweeps — the frozen gibbs_grid contract
      verbatim (P=8 lanes, one farm word per NON-empty chunk per D-014,
      simultaneous writes within a chunk, p17 conditionals, rnd[31:16] < p17
      decision); IMM (T_M sweeps at beta T_K) or loaded-schedule form;
      RECORD accumulates D19 moments at every sweep end (thin=1);
      STATS_RESET zeroes cnt/m1/m2 first.
  PDRAIN (T_OP=10): frozen little-endian u32 readback layouts — STATE,
      MOMENTS (cnt, m1[n], m2[n][8] slot-major), COV (D30: craw = cnt*m2 -
      m1_i*m1_j int64 on edge slots + diagonal, division never on device),
      WORK (per-replica int64, units 1/16 model energy), TELEMETRY.
      Drains mutate nothing.

Numeric contracts inherited frozen: J/bias s1.6.3, beta u2.6, p17 from
golden/pbit.py (M4), xoshiro128++ jump-spaced streams from golden/xoshiro.py
(D16). Semantics anchor for D30/D33: golden/fabric_anchor.py (independent
paragraph 8a) via the forced-state protocol — exercised in self_check().

Self-check (python -m golden.sampling_isa from the repository root):
  1. trajectory exactness vs gibbs_grid.BitTrueGridSampler on the directed
     S7.1 shapes D1-D5 (incl. the D-014 empty-color regression, clamping,
     multi-entry schedules, stream persistence, IMM==SCHED);
  2. moment accumulators vs naive recount + reset/extend discipline;
  3. D33 work vs the independent floating-point emulator, exact (dyadic float64);
  4. D30 moments/cov vs the independent floating-point emulator, integers exact / cov 1e-9;
  5. distributional: n=1 analytic p17 rate (|z|<5) and an n=6 ring vs the
     EXACT stationary distribution of the quantized chromatic sweep kernel
     (power iteration, the S2 pi_hw pattern), chi2 with tail pooling;
  6. drain formats, telemetry counters, determinism/seed separation.
"""
from __future__ import annotations

import numpy as np

from golden.pbit import ACC_MAX, ACC_MIN, p17
from golden.xoshiro import FarmVec, stream_states

# ---------------- frozen encodings (spec section 1/2) ----------------
OP_PCONFIG, OP_PSAMPLE, OP_PDRAIN = 8, 9, 10
MAGIC = 0x4D4B5331                      # "MKS1"
P_LANES = 8
N_MAX, DEG_MAX, COLORS_MAX, SCHED_MAX, R_MAX = 8192, 8, 16, 32, 64
CNT_MAX = 1 << 24

CF_ROWS, CF_ORDER, CF_SCHED, CF_SEEDS, CF_STATE, CF_RID = 1, 2, 4, 8, 16, 32
CF_WORK_RESET = 1 << 8
SF_RECORD, SF_STATS_RESET, SF_IMM = 1, 2, 4
DR_STATE, DR_MOMENTS, DR_COV, DR_WORK, DR_TELEMETRY = 0, 1, 2, 3, 4

M32 = 0xFFFFFFFF


def enc_slot(nbr: int, j_raw: int) -> int:
    """ROWS slot word: {valid, nbr u13, J s1.6.3 two's-complement u10}."""
    assert 0 <= nbr < N_MAX and -511 <= j_raw <= 511
    return (1 << 23) | ((nbr & 0x1FFF) << 10) | (j_raw & 0x3FF)


def seeds_words(mother_seed: int) -> list[int]:
    """The NORMATIVE SEEDS derivation (spec section 7): 8 jump-spaced streams."""
    return [w for st in stream_states(mother_seed, P_LANES) for w in st]


def build_image(n: int, *, rows=None, order=None, bounds=None, n_colors=0,
                sched=None, seeds=None, state=None, rid=None, n_replicas=1,
                bipolar=False, work_track=False, work_reset=False):
    """Encode a PCONFIG image. rows: per-site ([(nbr, j_raw)...], bias_raw).
    order/bounds: gibbs_grid.build_schedule outputs. sched: [(beta, sweeps)].
    seeds: flat 32 words. state: 0/1 per site. rid: replica id per site.
    Returns (words, t_flags) — T_M is len(words)."""
    assert 1 <= n <= N_MAX and 0 <= n_replicas - 1 < R_MAX
    words = [MAGIC,
             (n & 0xFFFF) | (int(bipolar) << 16) | (int(work_track) << 17),
             (n_colors & 0xFF) | ((len(sched or []) & 0xFF) << 8)
             | ((n_replicas & 0xFF) << 16)]
    flags = CF_WORK_RESET if work_reset else 0
    if rows is not None:
        flags |= CF_ROWS
        assert len(rows) == n
        for slots, bias in rows:
            assert len(slots) <= DEG_MAX
            w = [enc_slot(nb, j) for nb, j in slots]
            words += w + [0] * (DEG_MAX - len(w))
            assert -511 <= bias <= 511
            words.append(bias & 0x3FF)
    if order is not None:
        flags |= CF_ORDER
        assert n_colors == len(bounds) and n_colors <= COLORS_MAX
        words.append(len(order) & 0xFFFF)
        cb = [(lo & 0xFFFF) | ((hi & 0xFFFF) << 16) for lo, hi in bounds]
        words += cb + [0] * (COLORS_MAX - len(cb))
        words += [i & 0xFFFF for i in order]
    if sched is not None:
        flags |= CF_SCHED
        assert 1 <= len(sched) <= SCHED_MAX
        for beta, sweeps in sched:
            assert 0 <= beta <= 255 and 1 <= sweeps < (1 << 24)
            words.append((beta & 0xFF) | (sweeps << 8))
    if seeds is not None:
        flags |= CF_SEEDS
        assert len(seeds) == 4 * P_LANES
        words += [w & M32 for w in seeds]
    if state is not None:
        flags |= CF_STATE
        assert len(state) == n
        for w0 in range(0, n, 32):
            w = 0
            for b in range(min(32, n - w0)):
                w |= int(state[w0 + b]) << b
            words.append(w)
    if rid is not None:
        flags |= CF_RID
        assert len(rid) == n and max(rid) < n_replicas
        for w0 in range(0, n, 4):
            w = 0
            for b in range(min(4, n - w0)):
                w |= int(rid[w0 + b]) << (8 * b)
            words.append(w)
    return words, flags


class SClusterGolden:
    """Device-exact machine: consumes/produces the frozen word formats."""

    def __init__(self):
        self.n = None

    def _init(self, n: int):
        self.n = n
        self.nbr = np.zeros((n, DEG_MAX), dtype=np.int32)
        self.jraw = np.zeros((n, DEG_MAX), dtype=np.int32)
        self.valid = np.zeros((n, DEG_MAX), dtype=bool)
        self.bias = np.zeros(n, dtype=np.int32)
        self.order: list[int] = []
        self.bounds: list[tuple[int, int]] = []
        self.sched: list[tuple[int, int]] = []
        self.s = np.zeros(n, dtype=np.int64)
        self.rid = np.zeros(n, dtype=np.int32)
        self.n_replicas = 1
        self.bipolar = False
        self.work_track = False
        self.fv = None
        self.work = np.zeros(1, dtype=np.int64)
        self.cnt = 0
        self.m1 = np.zeros(n, dtype=np.int64)
        self.m2 = np.zeros((n, DEG_MAX), dtype=np.int64)
        self.sweeps_done = 0
        self.upd_cnt = 0
        self.flip_cnt = 0
        self.cfg_words = 0
        self.drain_words = 0

    # ---------------- PCONFIG ----------------
    def pconfig(self, words, t_flags: int):
        assert words[0] == MAGIC, "bad config magic"
        n = words[1] & 0xFFFF
        bipolar = bool((words[1] >> 16) & 1)
        work_track = bool((words[1] >> 17) & 1)
        n_colors = words[2] & 0xFF
        n_sched = (words[2] >> 8) & 0xFF
        n_replicas = (words[2] >> 16) & 0xFF
        if self.n is None:
            assert 1 <= n <= N_MAX
            self._init(n)
        assert n == self.n, "n_sites is immutable after first PCONFIG (v1)"
        assert 1 <= n_replicas <= R_MAX
        pre_track = self.work_track
        do_work = pre_track and bool(t_flags & CF_ROWS)
        if do_work:
            assert not self.bipolar and not bipolar, \
                "WORK_TRACK is binary-mode only (v1)"
            e_old = self._energy2()
        # header applies (n_replicas resize keeps low registers)
        self.bipolar, self.work_track = bipolar, work_track
        if n_replicas != self.n_replicas:
            w = np.zeros(n_replicas, dtype=np.int64)
            w[:min(n_replicas, self.n_replicas)] = \
                self.work[:min(n_replicas, self.n_replicas)]
            self.work, self.n_replicas = w, n_replicas
        pos = 3
        if t_flags & CF_ROWS:
            for i in range(self.n):
                for k in range(DEG_MAX):
                    w = words[pos + k]
                    self.valid[i, k] = bool((w >> 23) & 1)
                    self.nbr[i, k] = (w >> 10) & 0x1FFF
                    j = w & 0x3FF
                    self.jraw[i, k] = j - 1024 if j >= 512 else j
                b = words[pos + 8] & 0x3FF
                self.bias[i] = b - 1024 if b >= 512 else b
                pos += 9
            self.nbr[~self.valid] = 0
            self.jraw[~self.valid] = 0
            if do_work:
                self.work += self._energy2() - e_old
        if t_flags & CF_ORDER:
            assert n_colors <= COLORS_MAX
            n_ord = words[pos] & 0xFFFF
            pos += 1
            self.bounds = [((words[pos + c]) & 0xFFFF, (words[pos + c] >> 16)
                            & 0xFFFF) for c in range(n_colors)]
            pos += COLORS_MAX
            self.order = [words[pos + i] & 0xFFFF for i in range(n_ord)]
            pos += n_ord
        if t_flags & CF_SCHED:
            assert 1 <= n_sched <= SCHED_MAX
            self.sched = [(words[pos + e] & 0xFF, words[pos + e] >> 8)
                          for e in range(n_sched)]
            pos += n_sched
        if t_flags & CF_SEEDS:
            states = [[words[pos + 4 * l + k] for k in range(4)]
                      for l in range(P_LANES)]
            pos += 4 * P_LANES
            self.fv = FarmVec(states)
        if t_flags & CF_STATE:
            for w0 in range(0, self.n, 32):
                w = words[pos]
                pos += 1
                for b in range(min(32, self.n - w0)):
                    self.s[w0 + b] = (w >> b) & 1
        if t_flags & CF_RID:
            for w0 in range(0, self.n, 4):
                w = words[pos]
                pos += 1
                for b in range(min(4, self.n - w0)):
                    self.rid[w0 + b] = (w >> (8 * b)) & 0xFF
            assert self.rid.max() < self.n_replicas
        assert pos == len(words), "T_M / image length mismatch"
        if t_flags & CF_WORK_RESET:
            self.work[:] = 0
        self.cfg_words += len(words) - 3
        self._check_invariants(t_flags)

    def _check_invariants(self, t_flags: int):
        # I-3: no coupling crosses a RID partition
        if self.work_track or (t_flags & CF_RID):
            assert (self.rid[self.nbr[self.valid]]
                    == np.repeat(self.rid, self.valid.sum(1))).all(), \
                "I-3: coupling crosses replica partition"
        # I-4: proper coloring over the order list
        col = {}
        for c, (lo, hi) in enumerate(self.bounds):
            for i in self.order[lo:hi]:
                col[i] = c
        for i, c in col.items():
            for k in range(DEG_MAX):
                if self.valid[i, k]:
                    assert col.get(int(self.nbr[i, k]), -1) != c, \
                        "I-4: same-color neighbors"

    # ---------------- energy scan (spec section 5) ----------------
    def _energy2(self) -> np.ndarray:
        """Per-replica 2x energy in raw 1/8 units (units 1/16), int64."""
        pair = ((self.jraw * self.s[self.nbr]) * self.valid).sum(axis=1)
        term = self.s * (pair + 2 * self.bias)          # binary mode only
        e2 = np.zeros(self.n_replicas, dtype=np.int64)
        np.add.at(e2, self.rid, term.astype(np.int64))
        return -e2

    # ---------------- PSAMPLE ----------------
    def psample(self, t_flags: int, t_m: int = 0, t_k: int = 0):
        assert self.n is not None and self.fv is not None
        if t_flags & SF_STATS_RESET:
            self.cnt = 0
            self.m1[:] = 0
            self.m2[:] = 0
        if t_flags & SF_IMM:
            assert 1 <= t_m < (1 << 24)
            sched = [(t_k & 0xFF, t_m)]
        else:
            assert t_m == 0 and t_k == 0 and self.sched
            sched = self.sched
        record = bool(t_flags & SF_RECORD)
        for beta, sweeps in sched:
            for _ in range(sweeps):
                self._sweep(beta)
                self.sweeps_done += 1
                if record:
                    self._accumulate()

    def _acc(self, i: int, beta_unused=None) -> int:
        acc = int(self.bias[i])
        for k in range(DEG_MAX):
            if self.valid[i, k]:
                sj = int(self.s[self.nbr[i, k]])
                j = int(self.jraw[i, k])
                acc += (j if sj else -j) if self.bipolar else (j if sj else 0)
        assert ACC_MIN <= acc <= ACC_MAX, "I-8: MAC overflow (degree contract)"
        return acc

    def _sweep(self, beta: int):
        for lo, hi in self.bounds:
            pos = lo
            while pos < hi:                 # empty segment: no words (D-014)
                words = self.fv.next_block(1)[:, 0]
                chunk = self.order[pos:min(pos + P_LANES, hi)]
                newbits = []
                for lane, site in enumerate(chunk):
                    p = p17(self._acc(site), beta)
                    newbits.append((site, int((int(words[lane]) >> 16) < p)))
                for site, b in newbits:     # simultaneous within chunk
                    self.upd_cnt += 1
                    if b != self.s[site]:
                        self.flip_cnt += 1
                    self.s[site] = b
                pos += P_LANES

    def _accumulate(self):
        assert self.cnt < CNT_MAX, "I-8: cnt cap"
        self.cnt += 1
        self.m1 += self.s
        self.m2 += (self.s[:, None] & self.s[self.nbr]) * self.valid

    # ---------------- PDRAIN ----------------
    def pdrain(self, mode: int) -> np.ndarray:
        assert self.n is not None
        prior_drain_words = self.drain_words
        if mode == DR_STATE:
            nw = (self.n + 31) // 32
            out = np.zeros(nw, dtype=np.uint32)
            for i in range(self.n):
                if self.s[i]:
                    out[i // 32] |= np.uint32(1 << (i % 32))
        elif mode == DR_MOMENTS:
            assert self.m1.max(initial=0) < (1 << 32)
            out = np.concatenate([[self.cnt], self.m1,
                                  self.m2.reshape(-1)]).astype(np.uint32)
        elif mode == DR_COV:
            cnt = np.int64(self.cnt)
            diag = cnt * self.m1 - self.m1 * self.m1
            slots = (cnt * self.m2
                     - self.m1[:, None] * self.m1[self.nbr]) * self.valid
            rec = np.concatenate([diag[:, None], slots], axis=1)  # (n, 9)
            assert np.abs(rec).max(initial=0) < (1 << 48), "I-8: craw bound"
            out = np.concatenate(
                [np.array([self.cnt], dtype=np.uint32),
                 rec.astype("<i8").reshape(-1).view(np.uint32)])
        elif mode == DR_WORK:
            out = np.concatenate(
                [np.array([self.n_replicas], dtype=np.uint32),
                 self.work.astype("<i8").view(np.uint32)])
        elif mode == DR_TELEMETRY:
            out = np.array(
                [self.sweeps_done & M32,
                 self.upd_cnt & M32, (self.upd_cnt >> 32) & 0xFFFF,
                 self.flip_cnt & M32, (self.flip_cnt >> 32) & 0xFFFF,
                 self.cfg_words & M32, prior_drain_words & M32,
                 self.cnt & M32,
                 0, 0,          # busy_cycles: timing observable, golden = 0
                 0, 0], dtype=np.uint32)
        else:
            raise ValueError(f"reserved PDRAIN mode {mode}")
        self.drain_words += out.size
        return out


# ---------------- config helpers (gate/self-check plumbing) ----------------
def image_from_graph(g, bias_raw, sched, *, seed=None, seeds=None, state=None,
                     clamp=None, rid=None, n_replicas=1, bipolar=False,
                     work_track=False):
    """Full image from a gibbs_grid.Graph using the FROZEN coloring/order
    convention (build_schedule: greedy ascending, clamp = order exclusion)."""
    from golden.gibbs_grid import build_schedule
    _, order, bounds = build_schedule(g, clamp)
    rows = [(g.adj[i], int(bias_raw[i])) for i in range(g.n)]
    if seeds is None:
        seeds = seeds_words(seed)
    return build_image(g.n, rows=rows, order=order, bounds=bounds,
                       n_colors=len(bounds), sched=sched, seeds=seeds,
                       state=state, rid=rid, n_replicas=n_replicas,
                       bipolar=bipolar, work_track=work_track)


def rows_only_image(n, g, bias_raw, *, bipolar=False, work_track=False,
                    n_replicas=1, work_reset=False):
    rows = [(g.adj[i], int(bias_raw[i])) for i in range(n)]
    return build_image(n, rows=rows, bipolar=bipolar, work_track=work_track,
                       n_replicas=n_replicas, work_reset=work_reset)


# ---------------- self-check (S7 golden-side halves) ----------------
def _torus(nx, ny, jval=None, seed=None, deg4=True):
    from golden.gibbs_grid import Graph
    rng = np.random.default_rng(seed if seed is not None else 0)
    g = Graph(nx * ny)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for j in (y * nx + (x + 1) % nx, ((y + 1) % ny) * nx + x):
                jr = jval if jval is not None else _rnz(rng, 16)
                g.add_edge(i, j, jr)
            if not deg4:                       # king: add diagonals
                for j in (((y + 1) % ny) * nx + (x + 1) % nx,
                          ((y + 1) % ny) * nx + (x - 1) % nx):
                    g.add_edge(i, j, _rnz(rng, 16))
    return g


def _rnz(rng, m):
    v = 0
    while v == 0:
        v = int(rng.integers(-m, m + 1))
    return v


def _traj_pair(g, bias, beta, sweeps, seed, *, bipolar=False, clamp=None,
               state0=None):
    """golden ISA vs BitTrueGridSampler, per-sweep exact states."""
    from golden.gibbs_grid import BitTrueGridSampler
    sc = SClusterGolden()
    img, fl = image_from_graph(g, bias, [(beta, 1)], seed=seed,
                               bipolar=bipolar, clamp=clamp, state=state0)
    sc.pconfig(img, fl)
    ref = BitTrueGridSampler(g, bias, beta, seed, bipolar=bipolar,
                             clamp=clamp, state0=state0)
    for _ in range(sweeps):
        sc.psample(SF_IMM, t_m=1, t_k=beta)
        ref.sweep()
        assert np.array_equal(sc.s, ref.s), "S7.1 trajectory divergence"
    return sc


def self_check():
    from golden.gibbs_grid import Graph
    rng = np.random.default_rng(0x5337)

    # ---- 1. trajectory exactness, D1-D5 shapes ----
    g1 = _torus(4, 4, jval=8)
    _traj_pair(g1, np.zeros(16, np.int64), 32, 60, 0xD1)
    g2 = _torus(6, 6, seed=2, deg4=False)
    b2 = np.array([_rnz(rng, 8) for _ in range(36)], np.int64)
    _traj_pair(g2, b2, 48, 40, 0xD2, bipolar=True)
    cl = rng.random(36) < 0.2
    st = (rng.random(36) < 0.5).astype(np.int8)
    _traj_pair(g2, b2, 48, 40, 0xD3, clamp=cl, state0=st)
    # D4: 12+8 bipartite, all visibles clamped -> empty color segment
    g4 = Graph(20)
    for v in range(12):
        for h in range(12, 20):
            if (v + h) % 3 == 0:
                g4.add_edge(v, h, _rnz(rng, 10))
    cl4 = np.zeros(20, bool); cl4[:12] = True
    st4 = (rng.random(20) < 0.5).astype(np.int8)
    _traj_pair(g4, np.zeros(20, np.int64), 40, 30, 0xD4, clamp=cl4,
               state0=st4)
    # D5: schedule form == concatenated IMM calls; persistence split
    scA = SClusterGolden()
    imgA, flA = image_from_graph(g2, b2, [(16, 10), (64, 20)], seed=0xD5)
    scA.pconfig(imgA, flA)
    scA.psample(0)                              # SCHED form
    scB = SClusterGolden()
    scB.pconfig(imgA, flA)
    scB.psample(SF_IMM, t_m=10, t_k=16)
    scB.psample(SF_IMM, t_m=20, t_k=64)         # no SEEDS reload between
    assert np.array_equal(scA.s, scB.s), "D5 SCHED==IMM"
    print("S7 self-check 1 (trajectory D1-D5): OK")

    # ---- 2. moments vs naive recount; reset/extend discipline ----
    sc = SClusterGolden()
    img, fl = image_from_graph(g2, b2, [(48, 1)], seed=7)
    sc.pconfig(img, fl)
    states = []
    sc.psample(SF_IMM | SF_RECORD | SF_STATS_RESET, t_m=5, t_k=48)
    for _ in range(5):
        pass
    # recount by replay: fresh machine, same seeds, accumulate manually
    sc2 = SClusterGolden()
    sc2.pconfig(img, fl)
    cnt = 0; m1 = np.zeros(36, np.int64); m2 = np.zeros((36, 8), np.int64)
    for _ in range(5):
        sc2.psample(SF_IMM, t_m=1, t_k=48)
        cnt += 1; m1 += sc2.s
        m2 += (sc2.s[:, None] & sc2.s[sc2.nbr]) * sc2.valid
    d = sc.pdrain(DR_MOMENTS)
    want = np.concatenate([[cnt], m1, m2.reshape(-1)]).astype(np.uint32)
    assert np.array_equal(d, want), "S7.2 moments recount"
    sc.psample(SF_IMM | SF_RECORD, t_m=3, t_k=48)   # extend, no reset
    assert sc.cnt == 8, "extend discipline"
    sc.psample(SF_IMM | SF_RECORD | SF_STATS_RESET, t_m=1, t_k=48)
    assert sc.cnt == 1, "reset discipline"
    print("S7 self-check 2 (moments/reset/extend): OK")

    # ---- 3+4. D33 work + D30 cov vs thermoml (forced-state protocol) ----
    from golden.fabric_anchor import Fabric
    n = 12
    gt = _torus(4, 3, seed=9)
    bias = np.array([_rnz(rng, 8) for _ in range(n)], np.int64)
    # all-clamped: RECORD accumulates forced states, zero updates
    clA = np.ones(n, bool)
    forced = [(rng.random(n) < 0.5).astype(np.int8) for _ in range(6)]
    sc = SClusterGolden()
    img, fl = image_from_graph(gt, bias, [(64, 1)], seed=1, clamp=clA,
                               state=forced[0], work_track=True)
    sc.pconfig(img, fl)
    fab = Fabric(n, n_replicas=1)
    J = np.zeros((n, n))
    for i in range(n):
        for k in range(DEG_MAX):
            if sc.valid[i, k]:
                J[i, sc.nbr[i, k]] = sc.jraw[i, k] / 8.0
    fab.pconfig(J=J, b=bias / 8.0, T=1.0)
    fab.work = 0.0
    fab.S = forced[0][None, :].astype(np.float64)
    fab.reset_stats()
    for t, st_ in enumerate(forced):
        simg, sfl = build_image(n, state=st_, work_track=True)
        sc.pconfig(simg, sfl)
        sc.psample(SF_IMM | SF_RECORD | (SF_STATS_RESET if t == 0 else 0),
                   t_m=1, t_k=64)
        fab.S = st_[None, :].astype(np.float64)
        fab._accumulate()
    # moments/cov
    dm = sc.pdrain(DR_MOMENTS)
    assert dm[0] == len(forced)
    m1f = np.asarray([s.sum() for s in np.array(forced).T])
    assert np.array_equal(dm[1:1 + n], m1f.astype(np.uint32)), "m1 anchor"
    dc = sc.pdrain(DR_COV)
    craw = dc[1:].view("<i8").reshape(n, 9)
    cnt = int(dc[0])
    covf = fab.pdrain("cov")                    # emulator float cov
    for i in range(n):
        assert abs(craw[i, 0] / cnt**2
                   - covf[i, i]) < 1e-9, "diag cov anchor"
        for k in range(DEG_MAX):
            if sc.valid[i, k]:
                assert abs(craw[i, 1 + k] / cnt**2
                           - covf[i, sc.nbr[i, k]]) < 1e-9, "cov anchor"
    # work: rewrite ROWS (scaled couplings), forced state held
    j2 = np.minimum(np.maximum(sc.jraw * 2, -511), 511) * sc.valid
    rows2 = [([(int(sc.nbr[i, k]), int(j2[i, k])) for k in range(DEG_MAX)
               if sc.valid[i, k]], int(bias[i])) for i in range(n)]
    img2, fl2 = build_image(n, rows=rows2, work_track=True)
    sc.pconfig(img2, fl2)
    J2 = np.zeros((n, n))
    for i in range(n):
        for k in range(DEG_MAX):
            if sc.valid[i, k]:
                J2[i, sc.nbr[i, k]] = j2[i, k] / 8.0
    fab.pconfig(J=J2, b=bias / 8.0)
    dw = sc.pdrain(DR_WORK)
    wdev = dw[1:].view("<i8")[0] / 16.0
    assert abs(wdev - fab.work[0]) < 1e-12, f"work anchor {wdev} {fab.work}"
    print("S7 self-check 3+4 (independent work/cov anchors): OK")

    # ---- 5. distributional: n=1 analytic; n=6 ring vs exact kernel ----
    g0 = Graph(1)
    b0 = np.array([12], np.int64)
    sc = SClusterGolden()
    img, fl = image_from_graph(g0, b0, [(64, 1)], seed=42)
    sc.pconfig(img, fl)
    N = 20000
    hits = 0
    for _ in range(N):
        sc.psample(SF_IMM, t_m=1, t_k=64)
        hits += int(sc.s[0])
    pexp = p17(12, 64) / 65536.0
    z = (hits - N * pexp) / np.sqrt(N * pexp * (1 - pexp))
    assert abs(z) < 5, f"n=1 rate z={z:.2f}"
    # n=6 ring: exact stationary dist of the quantized sweep kernel
    g6 = Graph(6)
    for i in range(6):
        g6.add_edge(i, (i + 1) % 6, 6)
    b6 = np.array([2, -3, 1, 0, -2, 4], np.int64)
    sc = SClusterGolden()
    img, fl = image_from_graph(g6, b6, [(48, 1)], seed=11)
    sc.pconfig(img, fl)
    # exact per-sweep transition matrix (colors sequential, within-color
    # sites factorize: same-color sites are non-adjacent)
    col, order, bounds = None, sc.order.copy(), sc.bounds
    T = np.eye(64)
    for lo, hi in bounds:
        Tc = np.zeros((64, 64))
        sites = order[lo:hi]
        for st in range(64):
            sv = [(st >> i) & 1 for i in range(6)]
            probs = []
            for i in sites:
                acc = int(b6[i])
                for k in range(DEG_MAX):
                    if sc.valid[i, k]:
                        acc += int(sc.jraw[i, k]) if sv[sc.nbr[i, k]] else 0
                probs.append(p17(acc, 48) / 65536.0)
            for bits in range(1 << len(sites)):
                p = 1.0
                nv = sv[:]
                for idx, i in enumerate(sites):
                    b_ = (bits >> idx) & 1
                    p *= probs[idx] if b_ else 1 - probs[idx]
                    nv[i] = b_
                ns = sum(v << i for i, v in enumerate(nv))
                Tc[st, ns] += p
        T = T @ Tc
    w, v = np.linalg.eig(T.T)
    pi = np.real(v[:, np.argmax(np.real(w))]); pi /= pi.sum()
    M = 40000
    counts = np.zeros(64)
    for _ in range(M):
        sc.psample(SF_IMM, t_m=1, t_k=48)
        counts[sum(int(sc.s[i]) << i for i in range(6))] += 1
    # chi2 with tail pooling at expected>=5
    exp = pi * M
    big = exp >= 5
    chi2 = float((((counts[big] - exp[big]) ** 2) / exp[big]).sum()
                 + ((counts[~big].sum() - exp[~big].sum()) ** 2)
                 / max(exp[~big].sum(), 1e-9))
    dof = int(big.sum())
    assert chi2 < dof + 5 * np.sqrt(2 * dof), f"ring chi2 {chi2:.1f}/{dof}"
    print(f"S7 self-check 5 (analytic rate z={z:.2f}; "
          f"ring chi2 {chi2:.1f}, dof~{dof}): OK")

    # ---- 6. drains, telemetry, determinism ----
    d1 = sc.pdrain(DR_TELEMETRY)
    assert d1.size == 12 and d1[8] == 0 and d1[9] == 0
    s_before = sc.s.copy(); c_before = sc.cnt
    _ = sc.pdrain(DR_STATE); _ = sc.pdrain(DR_MOMENTS)
    assert np.array_equal(sc.s, s_before) and sc.cnt == c_before, \
        "PDRAIN mutates nothing"
    scX = SClusterGolden(); scY = SClusterGolden()
    for m_ in (scX, scY):
        m_.pconfig(img, fl)
        m_.psample(SF_IMM, t_m=17, t_k=48)
    assert np.array_equal(scX.s, scY.s) and scX.upd_cnt == scY.upd_cnt, \
        "determinism"
    print("S7 self-check 6 (drain purity, telemetry, determinism): OK")
    print("sampling_isa self-check: ALL OK")


if __name__ == "__main__":
    self_check()
