"""port_arbiter bench — the D-032c credit arbiter checked against its
port CONTRACTS (arbitration order is recipe, not contract, so there is
no cycle-exact golden):

  - every beat a port presents reaches the downstream face exactly
    once, payload-faithful, in that port's issue order (serialized
    addresses make duplicates/losses unambiguous — the D-017 class);
  - responses route back to the issuing port strictly in its issue
    order, carrying the response-cycle rdata;
  - <= CRED beats outstanding downstream, <= 1 for the legacy port A,
    <= CRED_B / CRED_C for the edge ports (credit loops honored).

Drivers: A is the v1 level-valid one-outstanding requester (payload
swaps or retires at the ack edge, the core's shape). B and C (stage
C2/C1 migrations) are v2 edge issuers, each with its own credit
counter and a settable burst appetite. The slave is an in-order queue
with random per-beat latency, one completion pulse per cycle — the
soc_harness shape. B and C are never concurrent in the SoC (one
outstanding command globally), but the fuzz drives them concurrently
anyway: the module as built is stronger, so hold it to that. All
driving happens at falling edges; Mealy outputs (acks on our own
driven rsp) are sampled in the settled PRE-edge ReadOnly window.
"""
from collections import deque

import cocotb
from cocotb.triggers import ReadOnly

from mkutil import Rng, drive_edge, get_nvec, get_seed, reset_n, \
    start_clock

TB = "port_arbiter"
CRED = 4
EDGE_CRED = {"b": 4, "c": 4}          # CRED_B / CRED_C

BASE = {"a": 0x0000_0000, "b": 0x1000_0000, "c": 0x2000_0000}
BY_NIB = {0x0: "a", 0x1: "b", 0x2: "c"}


def f_mem(addr):
    """Deterministic background memory pattern."""
    return (addr * 2654435761) & 0xFFFFFFFF


class Port:
    """Per-port bookkeeping: beats presented -> seen downstream ->
    responded, strictly in order."""

    def __init__(self, name):
        self.name = name
        self.serial = 0
        self.exp_dn = deque()      # presented, not yet seen downstream
        self.await_rsp = deque()   # downstream, awaiting response
        self.done = 0

    def new_beat(self, rng):
        we = rng.chance(0.5)
        beat = {"we": int(we),
                "addr": BASE[self.name] | ((self.serial << 2) & 0xF_FFFC),
                "wdata": rng.bits(32) if we else 0,
                "be": rng.bits(4) if we else 0xF,
                "serial": self.serial}
        self.serial += 1
        self.exp_dn.append(beat)
        return beat

    def outstanding(self):
        return len(self.exp_dn) + len(self.await_rsp)


async def run(dut, seed, n, target, *, a_rate=0.6, b_rate=0.0, c_rate=0.6,
              lat_lo=1, lat_hi=6):
    """One traffic leg: n active cycles, then a drain tail. A port's
    rate is its chance of presenting/pulsing on an eligible falling
    edge (0.0 disables the port)."""
    rng = Rng(seed)
    ports = {p: Port(p) for p in ("a", "b", "c")}
    rate = {"a": a_rate, "b": b_rate, "c": c_rate}
    a_live = None
    credits = dict(EDGE_CRED)
    mem = {}
    slave = deque()                # (due_tick, port_name, beat)
    t = 0

    def drive_beat(prefix, beat):
        getattr(dut, prefix + "_we").value = beat["we"]
        getattr(dut, prefix + "_addr").value = beat["addr"]
        getattr(dut, prefix + "_wdata").value = beat["wdata"]
        getattr(dut, prefix + "_be").value = beat["be"]

    def apply_write(b):
        old = mem.get(b["addr"], f_mem(b["addr"]))
        new = 0
        for byte in range(4):
            src = b["wdata"] if (b["be"] >> byte) & 1 else old
            new |= src & (0xFF << (8 * byte))
        mem[b["addr"]] = new

    for i in range(n + 400):
        draining = i >= n
        await drive_edge(dut)
        # ---- legacy level (A): present on idle, hold until the ack
        # edge; a held level stays up through the drain tail ----
        if a_live is None:
            if not draining and rng.chance(rate["a"]):
                a_live = ports["a"].new_beat(rng)
                drive_beat("a", a_live)
                dut.a_valid.value = 1
            else:
                dut.a_valid.value = 0
        # ---- edge ports (B, C): pulse when their credit pools allow --
        for pfx in ("b", "c"):
            if not draining and credits[pfx] > 0 and rng.chance(rate[pfx]):
                drive_beat(pfx, ports[pfx].new_beat(rng))
                getattr(dut, pfx + "_req").value = 1
                credits[pfx] -= 1
            else:
                getattr(dut, pfx + "_req").value = 0
        # ---- slave: complete the oldest due beat (one per cycle) ----
        rsp = None
        if slave and slave[0][0] <= t:
            _, sport, sbeat = slave.popleft()
            if sbeat["we"]:
                apply_write(sbeat)
                sbeat["rdata"] = 0
            else:
                sbeat["rdata"] = mem.get(sbeat["addr"], f_mem(sbeat["addr"]))
            dut.m_rsp_rdata.value = sbeat["rdata"]
            dut.m_rsp_valid.value = 1
            rsp = (sport, sbeat)
        else:
            dut.m_rsp_valid.value = 0

        # sample in the settled PRE-edge window: the acks/rdata are
        # combinational (Mealy on our own m_rsp_valid drive) — the
        # post-edge view already shows next-cycle state
        await ReadOnly()
        t += 1
        # ---- capture the downstream pulse ----
        if int(dut.m_req.value):
            assert len(slave) < CRED, \
                f"[{i}] >CRED beats outstanding downstream (seed {seed})"
            got = {"we": int(dut.m_we.value), "addr": int(dut.m_addr.value),
                   "wdata": int(dut.m_wdata.value), "be": int(dut.m_be.value)}
            src = ports[BY_NIB[got["addr"] >> 28]]
            assert src.exp_dn, \
                f"[{i}] {src.name}: downstream beat with none presented " \
                f"(duplicate grant? seed {seed}) got {got}"
            want = src.exp_dn.popleft()
            for k in ("we", "addr", "wdata", "be"):
                assert got[k] == want[k], \
                    f"[{i}] {src.name} beat #{want['serial']} {k}: got " \
                    f"{got[k]:#x} want {want[k]:#x} (seed {seed})"
            if src.name == "a":
                assert not src.await_rsp, \
                    f"[{i}] a: legacy port >1 outstanding (seed {seed})"
            else:
                assert len(src.await_rsp) < EDGE_CRED[src.name], \
                    f"[{i}] {src.name} beyond its CRED outstanding " \
                    f"(seed {seed})"
            src.await_rsp.append(want)
            slave.append((t + rng.randint(lat_lo, lat_hi), src.name, got))
        # ---- check response routing + per-port order ----
        acks = {"a": int(dut.a_ack.value), "b": int(dut.b_rsp_valid.value),
                "c": int(dut.c_rsp_valid.value)}
        if rsp is None:
            assert not any(acks.values()), \
                f"[{i}] ack with no completion pulse (seed {seed}): {acks}"
        else:
            sport, sbeat = rsp
            for name, v in acks.items():
                assert v == (1 if name == sport else 0), \
                    f"[{i}] rsp routing: beat of {sport} acked {acks} " \
                    f"(seed {seed})"
            port = ports[sport]
            assert port.await_rsp, \
                f"[{i}] {sport}: response with nothing awaited (seed {seed})"
            done = port.await_rsp.popleft()
            rdata = int({"a": dut.a_rdata, "b": dut.b_rsp_rdata,
                         "c": dut.c_rsp_rdata}[sport].value)
            if not done["we"]:
                assert rdata == sbeat["rdata"], \
                    f"[{i}] {sport} beat #{done['serial']} rdata: got " \
                    f"{rdata:#x} want {sbeat['rdata']:#x} (seed {seed})"
            port.done += 1
            if sport == "a":
                a_live = None          # retired at the ack edge
            else:
                credits[sport] += 1
                assert credits[sport] <= EDGE_CRED[sport], \
                    f"[{i}] {sport} credit overflow (seed {seed})"
        if draining and not slave and a_live is None \
                and not any(p.exp_dn or p.await_rsp for p in ports.values()):
            break
    assert not any(p.exp_dn or p.await_rsp for p in ports.values()), \
        f"undrained beats at end (seed {seed}): " + str(
            [(p.name, len(p.exp_dn), len(p.await_rsp))
             for p in ports.values()])
    done = {p.name: p.done for p in ports.values()}
    print(f"[{TB}:{target}] beats completed {done}")
    return sum(done.values())


async def boot(dut):
    start_clock(dut)
    await reset_n(dut, zero_signals=("a_valid", "a_we", "a_addr", "a_wdata",
                                     "a_be", "b_req", "b_we", "b_addr",
                                     "b_wdata", "b_be", "c_req", "c_we",
                                     "c_addr", "c_wdata", "c_be",
                                     "m_rsp_valid", "m_rsp_rdata"))


@cocotb.test()
async def smoke(dut):
    """Directed phases: each port alone (A legacy greedy; B and C at
    full burst against a slow slave), then pairwise contention, then
    all three at once."""
    await boot(dut)
    n = get_nvec(300)
    total = 0
    # core path alone, greedy (back-to-back held levels)
    total += await run(dut, 0xC0FFEE + 1, n, "smoke",
                       a_rate=0.95, b_rate=0, c_rate=0)
    # tensor path alone at max appetite vs slow slave (stage C2 shape)
    total += await run(dut, 0xC0FFEE + 2, n, "smoke",
                       a_rate=0, b_rate=1.0, c_rate=0, lat_lo=4, lat_hi=8)
    # sampler path alone at max appetite vs slow slave
    total += await run(dut, 0xC0FFEE + 3, n, "smoke",
                       a_rate=0, b_rate=0, c_rate=1.0, lat_lo=4, lat_hi=8)
    # A vs B (the serving deployment shape)
    total += await run(dut, 0xC0FFEE + 4, n, "smoke",
                       a_rate=0.9, b_rate=0.8, c_rate=0)
    # A vs C (the planning deployment shape)
    total += await run(dut, 0xC0FFEE + 5, n, "smoke",
                       a_rate=0.9, b_rate=0, c_rate=0.8)
    # everything at once (unreachable in the SoC; held to anyway)
    total += await run(dut, 0xC0FFEE + 6, n, "smoke",
                       a_rate=0.7, b_rate=0.5, c_rate=0.7)
    assert total > n, f"suspiciously little traffic: {total}"


@cocotb.test()
async def fuzz(dut):
    """Random mixes, seed-keyed; rates and slave latency swept per leg."""
    await boot(dut)
    seed = get_seed()
    n = get_nvec(8000)
    rng = Rng(seed)
    legs = max(1, n // 2000)
    for leg in range(legs):
        await run(dut, seed + leg, n // legs, "fuzz",
                  a_rate=rng.choice((0.0, 0.3, 0.9)),
                  b_rate=rng.choice((0.0, 0.5, 1.0)),
                  c_rate=rng.choice((0.3, 0.7, 1.0)),
                  lat_lo=1, lat_hi=rng.choice((2, 4, 8)))
