"""barrel_sched bench — exact per-cycle diff against a python rotation
model, with random (contract-respecting) busy set/clr stimulus. The SVA
inside the DUT checks INV-1/2/3/4 on every cycle of this walk; the bench
additionally diffs issue decisions and the whole occupancy pipe.
"""
import cocotb

from mkutil import Rng, check, drive_edge, get_nvec, get_seed, reset_n, \
    sample_edge, start_clock

TB = "barrel_sched"
W, DEPTH = 8, 5


class Model:
    def __init__(self):
        self.phase = 0
        self.busy = [False] * W
        self.pipe = [(False, 0)] * DEPTH

    def tick(self, set_b, set_w, clr_b, clr_w):
        issue_v, issue_w = not self.busy[self.phase], self.phase
        self.pipe = [(issue_v, issue_w)] + self.pipe[:-1]
        if set_b:
            self.busy[set_w] = True
        if clr_b:
            self.busy[clr_w] = False
        self.phase = (self.phase + 1) % W
        return issue_v, issue_w


async def run(dut, seed, n, target):
    rng = Rng(seed)
    start_clock(dut)
    await reset_n(dut, ("set_busy", "set_warp", "clr_busy", "clr_warp"))
    m = Model()
    # align model phase to DUT (reset_n consumed some cycles): with all
    # warps idle, issue_warp == phase at any sample point
    await drive_edge(dut)
    await sample_edge(dut)
    m.phase = int(dut.issue_warp.value)

    for i in range(n):
        # contract-respecting stimulus: set only idle warps that are in the
        # pipe-eligible window, clr only busy ones, never both same warp
        idle = [w for w in range(W) if not m.busy[w]]
        busy = [w for w in range(W) if m.busy[w]]
        set_b = bool(idle) and rng.chance(0.15)
        set_w = rng.choice(idle) if set_b else 0
        clr_ok = [w for w in busy if not (set_b and w == set_w)]
        clr_b = bool(clr_ok) and rng.chance(0.25)
        clr_w = rng.choice(clr_ok) if clr_b else 0

        await drive_edge(dut)
        dut.set_busy.value = int(set_b)
        dut.set_warp.value = set_w
        dut.clr_busy.value = int(clr_b)
        dut.clr_warp.value = clr_w

        await sample_edge(dut)
        m.tick(set_b, set_w, clr_b, clr_w)
        # post-edge DUT pipe == post-tick model pipe (pipe[0] is the
        # decision of the just-elapsed cycle, which tick() just pushed)
        if i >= DEPTH + 1:
            sv = int(dut.stage_valid.value)
            sw = int(dut.stage_warp.value)
            for s in range(DEPTH):
                v, w = m.pipe[s]
                check(((sv >> s) & 1) == int(v),
                      f"stage_valid[{s}] got {(sv >> s) & 1} want {int(v)}",
                      i, seed, TB, target)
                if v:
                    got_w = (sw >> (s * 3)) & 7
                    check(got_w == w,
                          f"stage_warp[{s}] got {got_w} want {w}",
                          i, seed, TB, target)
        # post-edge combinational issue view = post-tick model state
        exp_v, exp_w = (not m.busy[m.phase]), m.phase
        check(int(dut.issue_valid.value) == int(exp_v),
              f"issue_valid got {int(dut.issue_valid.value)} want {exp_v}",
              i, seed, TB, target)
        if exp_v:
            check(int(dut.issue_warp.value) == exp_w,
                  f"issue_warp got {int(dut.issue_warp.value)} want {exp_w}",
                  i, seed, TB, target)


@cocotb.test()
async def smoke(dut):
    await run(dut, get_seed(), get_nvec(500), "smoke")


@cocotb.test()
async def fuzz(dut):
    await run(dut, get_seed(), get_nvec(20000), "fuzz")
