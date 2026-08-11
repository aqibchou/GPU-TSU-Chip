#!/usr/bin/env python3
"""M16: the pessimistic memory-truth model (docs/HARDWARE_ARCHITECTURE.md#memory-hierarchy §2).

EVERY parameter is a guess pending Bring-up calibration (risk #10b) —
deliberately pessimistic so sigma performance claims under-promise.

Cycle-stepped: call tick() once per core cycle; issue(port, we, addr,
wdata, be) starts a transaction (returns False if the port is busy);
completed transactions surface via take_response(port) -> (rdata) once.
Backing store is a flat bytearray; DRAMsim3 mounts behind this same
class interface as the high-fidelity lane (dramsim3_backend=True once the
glue lands; the timing source changes, the semantics do not).
"""
BASE_LATENCY = 40          # cycles; GUESS: DDR4 CL+tRCD+fabric @100MHz
TOTAL_BW_BYTES_PER_CYC = 100.0   # GUESS: 10 GB/s at 100 MHz
BUCKET_CAP_BYTES = 64.0    # GUESS: two lines of burst absorption


class AxiPessimistic:
    def __init__(self, size, n_ports=1, base_latency=BASE_LATENCY,
                 bw_bytes_per_cyc=None):
        self.mem = bytearray(size)
        self.n = n_ports
        self.lat = base_latency
        self.bw = (bw_bytes_per_cyc if bw_bytes_per_cyc is not None
                   else TOTAL_BW_BYTES_PER_CYC / n_ports)
        self.tokens = [BUCKET_CAP_BYTES] * n_ports
        self.inflight = [None] * n_ports   # (done_cycle, we, addr, wdata, be)
        self.resp = [None] * n_ports
        self.cycle = 0
        self.stall_cycles = [0] * n_ports  # observability (G-gate fuel)

    def issue(self, port, we, addr, wdata=0, be=0xF):
        if self.inflight[port] is not None or self.resp[port] is not None:
            return False
        cost = 4.0                          # one 32-bit beat
        if self.tokens[port] < cost:
            self.stall_cycles[port] += 1
            return False                    # bandwidth throttle: retry
        self.tokens[port] -= cost
        self.inflight[port] = (self.cycle + self.lat, we, addr & ~3,
                               wdata & 0xFFFFFFFF, be & 0xF)
        return True

    def tick(self):
        self.cycle += 1
        for p in range(self.n):
            self.tokens[p] = min(BUCKET_CAP_BYTES, self.tokens[p] + self.bw)
            tr = self.inflight[p]
            if tr is not None and self.cycle >= tr[0]:
                _, we, addr, wdata, be = tr
                if we:
                    for b in range(4):
                        if (be >> b) & 1:
                            self.mem[addr + b] = (wdata >> (8 * b)) & 0xFF
                    self.resp[p] = 0
                else:
                    self.resp[p] = int.from_bytes(
                        self.mem[addr:addr + 4], "little")
                self.inflight[p] = None

    def take_response(self, port):
        r = self.resp[port]
        self.resp[port] = None
        return r


if __name__ == "__main__":
    # self-check: values identical to a flat model; latency ~= base+queue
    import random
    rng = random.Random(1)
    m = AxiPessimistic(4096, n_ports=1)
    ref = bytearray(4096)
    t_issue, lats = None, []
    pending = None
    for cyc in range(20000):
        m.tick()
        if pending is not None:
            r = m.take_response(0)
            if r is not None:
                we, addr, wd, be, t0 = pending
                lats.append(m.cycle - t0)
                if we:
                    for b in range(4):
                        if (be >> b) & 1:
                            ref[addr + b] = (wd >> (8 * b)) & 0xFF
                else:
                    assert r == int.from_bytes(ref[addr:addr + 4], "little")
                pending = None
        if pending is None and rng.random() < 0.7:
            we = rng.random() < 0.5
            addr = rng.randrange(0, 4096, 4)
            wd, be = rng.getrandbits(32), rng.getrandbits(4) or 0xF
            if m.issue(0, we, addr, wd, be):
                pending = (we, addr, wd, be, m.cycle)
    assert min(lats) >= BASE_LATENCY, min(lats)
    print(f"axi_pessimistic self-check OK: {len(lats)} transactions, "
          f"latency min/mean {min(lats)}/{sum(lats)/len(lats):.1f} cycles")
