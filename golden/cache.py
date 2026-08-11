#!/usr/bin/env python3
"""M16 golden cache model (docs/HARDWARE_ARCHITECTURE.md#memory-hierarchy §3) — the bit-true contract
the RTL diffs against. 2-way set-associative, 32 B lines, LRU, write-back
+ write-allocate, blocking. State exposed for exact end-state diffing:
tags/valid/dirty/lru arrays + line data. Timing-free (the RTL bench pairs
it with axi_pessimistic; values must be latency-invariant).

Miss flow (architectural order): LRU victim -> dirty writeback (full
line) -> fill (full line) -> perform access -> LRU update.
"""
LINE = 32
WAYS = 2


class GoldenCache:
    def __init__(self, size_bytes, backing):
        self.sets = size_bytes // (LINE * WAYS)
        self.mem = backing                      # bytearray (shared truth)
        self.valid = [[False] * WAYS for _ in range(self.sets)]
        self.dirty = [[False] * WAYS for _ in range(self.sets)]
        self.tag = [[0] * WAYS for _ in range(self.sets)]
        self.data = [[bytearray(LINE) for _ in range(WAYS)]
                     for _ in range(self.sets)]
        self.lru = [0] * self.sets              # way to evict NEXT
        self.hits = self.misses = self.writebacks = 0

    def _decode(self, addr):
        off = addr % LINE
        s = (addr // LINE) % self.sets
        t = addr // (LINE * self.sets)
        return t, s, off

    def _lookup(self, t, s):
        for w in range(WAYS):
            if self.valid[s][w] and self.tag[s][w] == t:
                return w
        return None

    def _fill(self, t, s):
        w = self.lru[s]
        if self.valid[s][w] and self.dirty[s][w]:
            base = (self.tag[s][w] * self.sets + s) * LINE
            self.mem[base:base + LINE] = self.data[s][w]
            self.writebacks += 1
        base = (t * self.sets + s) * LINE
        self.data[s][w] = bytearray(self.mem[base:base + LINE])
        self.valid[s][w] = True
        self.dirty[s][w] = False
        self.tag[s][w] = t
        return w

    def _touch(self, s, w):
        self.lru[s] = 1 - w                     # 2-way: evict the other

    def access(self, we, addr, wdata=0, be=0xF):
        """One 32-bit access. Returns rdata (reads) or None (writes)."""
        t, s, off = self._decode(addr & ~3)
        w = self._lookup(t, s)
        if w is None:
            self.misses += 1
            w = self._fill(t, s)
        else:
            self.hits += 1
        self._touch(s, w)
        if we:
            for b in range(4):
                if (be >> b) & 1:
                    self.data[s][w][off + b] = (wdata >> (8 * b)) & 0xFF
            self.dirty[s][w] = True
            return None
        return int.from_bytes(self.data[s][w][off:off + 4], "little")

    def flush(self):
        """Write back all dirty lines (end-state comparison aid)."""
        for s in range(self.sets):
            for w in range(WAYS):
                if self.valid[s][w] and self.dirty[s][w]:
                    base = (self.tag[s][w] * self.sets + s) * LINE
                    self.mem[base:base + LINE] = self.data[s][w]
                    self.dirty[s][w] = False


if __name__ == "__main__":
    import random
    rng = random.Random(2)
    SIZE = 1 << 16
    truth = bytearray(rng.randbytes(SIZE))
    ref = bytearray(truth)
    c = GoldenCache(4096, bytearray(truth))
    for i in range(200_000):
        # mix random with directed same-set thrash (3 tags, one set)
        if rng.random() < 0.3:
            addr = (rng.choice([0, 1, 2]) * (4096 // WAYS)) + \
                   (7 * LINE) + rng.randrange(0, LINE, 4)
        else:
            addr = rng.randrange(0, SIZE, 4)
        if rng.random() < 0.5:
            wd, be = rng.getrandbits(32), (rng.getrandbits(4) or 0xF)
            c.access(True, addr, wd, be)
            for b in range(4):
                if (be >> b) & 1:
                    ref[addr + b] = (wd >> (8 * b)) & 0xFF
        else:
            got = c.access(False, addr)
            want = int.from_bytes(ref[addr:addr + 4], "little")
            assert got == want, (hex(addr), hex(got), hex(want), i)
    c.flush()
    assert c.mem == ref, "post-flush memory image mismatch"
    print(f"golden cache self-check OK: 200k accesses, hits {c.hits} "
          f"misses {c.misses} writebacks {c.writebacks}, image exact")
