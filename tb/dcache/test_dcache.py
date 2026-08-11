"""dcache bench — exact per-access diff vs golden/cache.py with
axi_pessimistic backing the memory face at REAL latencies. Latency
invariance proven by running the same traffic at base latency 1 and 40
(value streams must be identical). Directed same-set thrash mixed into
random traffic; final read-sweep validates the merged image.
"""
import cocotb

from golden.cache import GoldenCache
from sim.axi_pessimistic import AxiPessimistic
from mkutil import Rng, check, drive_edge, get_nvec, get_seed, sample_edge, \
    start_clock

TB = "dcache"
SIZE = 1 << 15


async def run_traffic(dut, seed, n, target, latency):
    rng = Rng(seed)
    start_clock(dut)
    axi = AxiPessimistic(SIZE, n_ports=1, base_latency=latency)
    init = bytes(rng.getrandbits(8) for _ in range(SIZE))
    axi.mem[:] = init
    ref_backing = bytearray(init)
    gold = GoldenCache(4096, ref_backing)
    values = []

    await drive_edge(dut)
    dut.rst_n.value = 0
    dut.c_valid.value = 0
    dut.m_ack.value = 0
    await sample_edge(dut)
    await drive_edge(dut)
    dut.rst_n.value = 1
    await sample_edge(dut)

    issued = 0
    pending = None
    mem_wait = False
    for cyc in range(n * 400 + 2000):
        await drive_edge(dut)
        # memory face service
        axi.tick()
        r = axi.take_response(0)
        if r is not None:
            dut.m_ack.value = 1
            dut.m_rdata.value = r
            mem_wait = False
        else:
            dut.m_ack.value = 0
        if r is None and not mem_wait and int(dut.m_valid.value):
            ok = axi.issue(0, bool(int(dut.m_we.value)),
                           int(dut.m_addr.value) % SIZE,
                           int(dut.m_wdata.value), int(dut.m_be.value))
            if ok:
                mem_wait = True
        # core face drive
        if pending is None and issued < n:
            if rng.random() < 0.3:      # same-set thrash
                addr = (rng.choice([0, 1, 2]) * 2048) + (7 * 32) + \
                       rng.randrange(0, 32, 4)
            else:
                addr = rng.randrange(0, SIZE, 4)
            we = rng.random() < 0.5
            wd = rng.getrandbits(32)
            be = rng.getrandbits(4) or 0xF
            dut.c_valid.value = 1
            dut.c_we.value = int(we)
            dut.c_addr.value = addr
            dut.c_wdata.value = wd
            dut.c_be.value = be
            pending = (we, addr, wd, be)
            issued += 1
        else:
            dut.c_valid.value = 0
        await sample_edge(dut)
        if pending is not None and int(dut.c_ack.value):
            we, addr, wd, be = pending
            exp = gold.access(we, addr, wd, be)
            if not we:
                got = int(dut.c_rdata.value)
                check(got == exp,
                      f"rdata @{addr:08x} got {got:08x} want {exp:08x}",
                      issued, seed, TB, target)
                values.append(got)
            pending = None
        if issued >= n and pending is None:
            break
    check(issued >= n, f"only issued {issued}/{n}", 0, seed, TB, target)
    # final read-sweep over the thrash set + random sample
    sweep = [(t * 2048) + (7 * 32) + o for t in (0, 1, 2)
             for o in range(0, 32, 4)]
    sweep += [rng.randrange(0, SIZE, 4) for _ in range(64)]
    for addr in sweep:
        dut_r = None
        for _ in range(2000):
            await drive_edge(dut)
            axi.tick()
            r = axi.take_response(0)
            if r is not None:
                dut.m_ack.value = 1
                dut.m_rdata.value = r
                mem_wait = False
            else:
                dut.m_ack.value = 0
            if r is None and not mem_wait and int(dut.m_valid.value):
                if axi.issue(0, bool(int(dut.m_we.value)),
                             int(dut.m_addr.value) % SIZE,
                             int(dut.m_wdata.value), int(dut.m_be.value)):
                    mem_wait = True
            if dut_r is None:
                dut.c_valid.value = 1
                dut.c_we.value = 0
                dut.c_addr.value = addr
                dut_r = "issued"
            else:
                dut.c_valid.value = 0
            await sample_edge(dut)
            if int(dut.c_ack.value):
                got = int(dut.c_rdata.value)
                exp = gold.access(False, addr)
                check(got == exp,
                      f"sweep @{addr:08x} got {got:08x} want {exp:08x}",
                      0, seed, TB, target)
                values.append(got)
                break
        else:
            check(False, f"sweep @{addr:08x} timed out", 0, seed, TB, target)
    print(f"[{TB}] lat={latency}: {issued} accesses + {len(sweep)} sweep, "
          f"exact (hits {int(dut.cnt_hit.value)} misses "
          f"{int(dut.cnt_miss.value)} wb {int(dut.cnt_wb.value)})")
    return values


@cocotb.test()
async def smoke(dut):
    seed = get_seed()
    v40 = await run_traffic(dut, seed, get_nvec(300), "smoke", 40)


@cocotb.test()
async def fuzz(dut):
    seed = get_seed()
    n = get_nvec(3000)
    v40 = await run_traffic(dut, seed, n, "fuzz", 40)
    v1 = await run_traffic(dut, seed, n, "fuzz", 1)
    check(v40 == v1, "value stream differs across memory latencies",
          0, seed, TB, "fuzz")
    print(f"[{TB}] latency-invariance: {len(v40)} values identical at "
          f"lat 1 vs 40")
