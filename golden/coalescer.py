#!/usr/bin/env python3
"""M16 golden coalescer (docs/HARDWARE_ARCHITECTURE.md#memory-hierarchy §4) — the behavioral contract
for merging one warp memory op's <= L lane accesses into cache requests.

Rules (frozen):
- word granularity (v1): distinct WORD addresses each get one cache
  request; service order = ascending lane-index-of-first-toucher.
- loads: every enabled lane receives the word of its address (lanes
  sharing a word share one request — the bandwidth win).
- stores: per word, byte-enables OR-merge across lanes; conflicting bytes
  resolve LANE-ASCENDING (highest lane wins) — exactly golden/simt_iss's
  sequential lane order semantics.
Returns the request list so the RTL coalescer diffs 1:1 (order included).
"""


def coalesce(lanes):
    """lanes: list of (lane, we, addr, wdata, be) with word-aligned addr
    semantics handled by caller (addr may be unaligned-in-word; be marks
    the bytes). Returns ordered requests:
      loads : [{"addr", "lanes": [lane, ...]}]
      stores: [{"addr", "wdata", "be"}]
    """
    if not lanes:
        return []
    is_store = lanes[0][1]
    order = []                       # word addr by first toucher
    groups = {}
    for (ln, we, addr, wdata, be) in sorted(lanes, key=lambda x: x[0]):
        assert we == is_store, "mixed op kinds in one warp memory op"
        w = addr & ~3
        if w not in groups:
            groups[w] = []
            order.append(w)
        groups[w].append((ln, addr, wdata, be))
    out = []
    for w in order:
        if not is_store:
            out.append({"addr": w, "lanes": [g[0] for g in groups[w]]})
        else:
            wdata, be = 0, 0
            for (_ln, addr, wd, lbe) in groups[w]:    # lane-ascending
                sh = (addr & 3) * 8
                for b in range(4):
                    if (lbe >> b) & 1:
                        byte = (wd >> (8 * b)) & 0xFF
                        pos = ((addr & 3) + b) & 3    # byte pos in word
                        wdata = (wdata & ~(0xFF << (8 * pos))) | \
                                (byte << (8 * pos))
                        be |= 1 << pos
            _ = sh
            out.append({"addr": w, "wdata": wdata, "be": be})
    return out


if __name__ == "__main__":
    import random
    rng = random.Random(3)
    for trial in range(20000):
        is_store = rng.random() < 0.5
        n = rng.randint(1, 8)
        lanes = []
        for ln in sorted(rng.sample(range(8), n)):
            size = rng.choice([1, 2, 4])
            addr = rng.randrange(0, 256, size)
            be = ((1 << size) - 1)
            lanes.append((ln, is_store, addr, rng.getrandbits(32),
                          be))
        reqs = coalesce(lanes)
        if is_store:
            # reference: apply lane-by-lane to a flat page, compare
            ref = bytearray(256)
            for (ln, _we, addr, wd, be) in lanes:
                for b in range(4):
                    if (be >> b) & 1:
                        ref[addr + b] = (wd >> (8 * b)) & 0xFF
            got = bytearray(256)
            seen = set()
            for r in reqs:
                assert r["addr"] not in seen, "duplicate word request"
                seen.add(r["addr"])
                for b in range(4):
                    if (r["be"] >> b) & 1:
                        got[r["addr"] + b] = (r["wdata"] >> (8 * b)) & 0xFF
            assert got == ref, (trial, lanes, reqs)
        else:
            covered = set()
            for r in reqs:
                covered.update(r["lanes"])
            assert covered == {ln for (ln, *_r) in lanes}
    print("golden coalescer self-check OK: 20k random warp ops, "
          "store merges == lane-sequential reference, loads cover all lanes")
