"""Golden model: synchronous show-ahead FIFO (matches rtl/common/fifo.sv).

Semantics, per clock edge:
- write accepted iff wr_en and not full (full/empty judged on PRE-edge state)
- read  accepted iff rd_en and not empty
- simultaneous read+write allowed (including when full: the write is refused,
  because `full` is judged pre-edge — same as the RTL gating)
- outputs are post-edge state: count/full/empty, and rd_data shows the head
  combinationally (show-ahead); rd_data is DON'T-CARE when empty (model
  returns None; the bench skips the compare).
"""


class FifoModel:
    def __init__(self, depth: int):
        self.depth = depth
        self.q: list[int] = []

    def step(self, wr_en: bool, wr_data: int, rd_en: bool) -> dict:
        do_wr = wr_en and len(self.q) < self.depth
        do_rd = rd_en and len(self.q) > 0
        if do_rd:
            self.q.pop(0)
        if do_wr:
            self.q.append(wr_data)
        return {
            "count": len(self.q),
            "full": int(len(self.q) == self.depth),
            "empty": int(len(self.q) == 0),
            "rd_data": self.q[0] if self.q else None,
        }


if __name__ == "__main__":
    # self-check: fill, drain, simultaneous ops at boundaries
    m = FifoModel(2)
    assert m.step(True, 1, False)["count"] == 1
    assert m.step(True, 2, False)["full"] == 1
    r = m.step(True, 3, False)          # write refused when full
    assert r["count"] == 2 and r["rd_data"] == 1
    r = m.step(True, 3, True)           # rd yes, wr refused (full judged pre-edge)
    assert r["count"] == 1 and r["rd_data"] == 2
    r = m.step(False, 0, True)
    assert r["empty"] == 1 and r["rd_data"] is None
    r = m.step(False, 0, True)          # read on empty: refused
    assert r["empty"] == 1
    print("fifo golden self-check OK")
