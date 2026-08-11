"""Golden model: pipelined unsigned multiplier (matches rtl/common/mul_pipe.sv).

Latency convention (matches the bench's post-edge sampling): operands captured
at edge N cross STAGES flip-flops and are observable at the sample point of
edge N+STAGES-1 — i.e., step() call K's input emerges from step() call
K+STAGES-1. p is DON'T-CARE while out_valid is 0 (model returns None for it).
"""


class MulPipeModel:
    def __init__(self, width: int = 16, stages: int = 3):
        self.width = width
        self.stages = stages
        # a step both appends and pops, so a backlog of STAGES-1 entries
        # yields the (STAGES-1)-call observation delay of STAGES flops
        self.pipe: list[tuple[int, int]] = [(0, 0)] * (stages - 1)

    def step(self, in_valid: bool, a: int, b: int) -> dict:
        mask = (1 << (2 * self.width)) - 1
        self.pipe.append((int(in_valid), (a * b) & mask))
        valid, prod = self.pipe.pop(0)
        return {"out_valid": valid, "p": prod if valid else None}


if __name__ == "__main__":
    m = MulPipeModel(16, 3)
    outs = [m.step(True, 3, 5), m.step(False, 0, 0), m.step(True, 1000, 1000),
            m.step(False, 0, 0), m.step(False, 0, 0), m.step(False, 0, 0)]
    assert outs[0]["out_valid"] == 0 and outs[1]["out_valid"] == 0
    assert outs[2] == {"out_valid": 1, "p": 15}          # call 0 -> call 0+3-1
    assert outs[3]["out_valid"] == 0
    assert outs[4] == {"out_valid": 1, "p": 1000000}     # call 2 -> call 4
    assert outs[5]["out_valid"] == 0
    print("mul golden self-check OK")
