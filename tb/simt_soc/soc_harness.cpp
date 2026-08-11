// M17 persistent SoC harness (runtime_spec §5): Verilated simt_core +
// flat RAM behind the frozen memory contracts (imem: req -> rdata
// next cycle; dmem: v2 CREDIT face per mem_spec §1b — req PULSE per
// beat, in-order completion pulses, each beat serviced at capture
// cycle + LAT with pipelined overlap, the axi_pessimistic shape).
// D-032c note: the v1 harness never seeded its countdown from `lat`,
// so the LAT knob was a NO-OP and G1's latency-invariance ran
// vacuously at effective latency 1; this rewrite implements LAT for
// real. Speaks a line protocol on stdin/stdout so
// host/run_kernel.py can drive many kernels through ONE process:
//   LOAD <hexaddr> <hexbytes>          -> OK
//   PEEK <hexaddr> <len>               -> <hexbytes>
//   RESET                              -> OK        (RAM persists)
//   RUN <cycles>                       -> RAN <cycles> <commits>
//   COUNTERS  -> CYCLES n COMMITS n WBCOMMITS n MEMCOMMITS n DBEATS n
//   QUIT
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "Vsimt_soc.h"
#include "Vsimt_soc___024root.h"
#include "verilated.h"

static const size_t MEM_SIZE = 0x10000000u; // 256 MiB (gpt2_spec §1 amendment)

struct Soc {
    Vsimt_soc top;
    std::vector<uint8_t> mem;
    // registered memory-face state
    bool i_req = false;
    uint32_t i_addr = 0;
    struct Dreq {
        bool we;
        uint32_t addr, wdata;
        uint8_t be;
        uint64_t done;              // absolute service cycle
    };
    std::deque<Dreq> dq;            // in-flight beats, issue order
    uint32_t lat = 1;               // service latency (G1 knob)
    // counters
    uint64_t cycles = 0, wb_commits = 0, mem_commits = 0, dbeats = 0;
    uint64_t tbusy = 0;             // sidecar busy cycles (G5 fuel)
    // leg-A pilot sampler (SCHEDCLR arms it; off by default so the
    // sigma gates/certs pay nothing). Classes are PC ranges set by
    // SCHEDCLASS; class 0 = "other/worker".
    bool sched_on = false;
    uint64_t s_cyc = 0, s_hist[9] = {0}, s_w0k[9] = {0};
    uint64_t s_wrdy[8] = {0}, s_wiss[8] = {0}, s_wcls[8][5] = {{0}};
    uint32_t s_clo[4] = {0}, s_chi[4] = {0};
    // PSAMPLE-decompose sampler (PSTATCLR arms it; off by default so
    // certs pay nothing). Per-cycle histogram of the s_cluster FSM
    // state (u_sampler.st, 5b): P_ACC walk vs P_WAIT sweep vs
    // idle/config/drain shares in one readout.
    bool pstat_on = false;
    uint64_t p_cyc = 0, p_hist[32] = {0};

    void sched_clear() {
        s_cyc = 0;
        for (int i = 0; i < 9; i++) { s_hist[i] = 0; s_w0k[i] = 0; }
        for (int w = 0; w < 8; w++) {
            s_wrdy[w] = 0; s_wiss[w] = 0;
            for (int c = 0; c < 5; c++) s_wcls[w][c] = 0;
        }
        sched_on = true;
    }

    int classify(uint32_t pc) {
        for (int i = 0; i < 4; i++)
            if (s_chi[i] > s_clo[i] && pc >= s_clo[i] && pc < s_chi[i])
                return i + 1;
        return 0;
    }

    void sched_sample() {
        auto* r = top.rootp;
        uint8_t busy = r->simt_soc__DOT__u_core__DOT__busy;
        uint8_t phase = r->simt_soc__DOT__u_core__DOT__u_sched__DOT__phase & 7;
        uint8_t ready = (uint8_t)~busy;
        int cnt = __builtin_popcount(ready);
        s_cyc++;
        s_hist[cnt]++;
        if (ready & 1) s_w0k[cnt]++;
        for (int w = 0; w < 8; w++) {
            if ((ready >> w) & 1) {
                s_wrdy[w]++;
                uint32_t pc = r->simt_soc__DOT__u_core__DOT__pc[w];
                s_wcls[w][classify(pc)]++;
            }
        }
        if ((ready >> phase) & 1) s_wiss[phase]++;
    }

    Soc() : mem(MEM_SIZE, 0) {
        top.rst_n = 0;
        top.imem_rdata = 0;
        top.dmem_rdata = 0;
        top.dmem_ack = 0;
        top.clk = 0;
    }

    uint32_t rd32(uint32_t a) {
        a %= MEM_SIZE;
        return (uint32_t)mem[a] | ((uint32_t)mem[a + 1] << 8) |
               ((uint32_t)mem[a + 2] << 16) | ((uint32_t)mem[a + 3] << 24);
    }

    void reset() {
        // phase-deterministic: end at clk=0 (falling), release rst there,
        // capture the release-cycle fetch — regardless of prior phase.
        top.rst_n = 0;
        top.dmem_ack = 0;
        top.clk = 0;
        top.eval();
        for (int i = 0; i < 2; i++) {
            top.clk = 1; top.eval();
            top.clk = 0; top.eval();
        }
        i_req = false;
        dq.clear();
        top.rst_n = 1;
        top.eval();
        i_req = top.imem_req;
        i_addr = top.imem_addr % MEM_SIZE;
    }

    void tick() {
        // falling edge: complete the oldest due beat (in-order; one
        // completion pulse per cycle; beats overlap their latency)
        top.clk = 0;
        if (i_req) top.imem_rdata = rd32(i_addr);
        if (!dq.empty() && cycles >= dq.front().done) {
            const Dreq& r = dq.front();
            if (r.we) {
                for (int b = 0; b < 4; b++)
                    if ((r.be >> b) & 1)
                        mem[(r.addr + b) % MEM_SIZE] =
                            (r.wdata >> (8 * b)) & 0xFF;
                top.dmem_rdata = 0;
            } else {
                top.dmem_rdata = rd32(r.addr);
            }
            top.dmem_ack = 1;
            dbeats++;
            dq.pop_front();
        } else {
            top.dmem_ack = 0;
        }
        top.eval();
        // capture THIS cycle's request pulse (post-falling; each
        // pulse is exactly one beat — no dedup state needed)
        i_req = top.imem_req;
        i_addr = top.imem_addr % MEM_SIZE;
        if (top.dmem_req && dq.size() < 8) {
            dq.push_back({top.dmem_we != 0,
                          top.dmem_addr % MEM_SIZE,
                          top.dmem_wdata,
                          (uint8_t)top.dmem_be,
                          cycles + (lat < 1 ? 1 : lat)});
        }
        // rising edge
        top.clk = 1;
        top.eval();
        cycles++;
        if (sched_on) sched_sample();
        if (pstat_on) {
#if MK_PROFILE != 1
            p_hist[top.rootp->simt_soc__DOT__g_samp__DOT__u_sampler__DOT__st & 31]++;
#endif
            p_cyc++;
        }
        if (top.cmt_valid) wb_commits++;
        if (top.mcmt_valid) mem_commits++;
        // busy proxy: the sidecar is mid-command between doorbell and
        // cnt_ops increment; count via its memory activity + compute —
        // exposed precisely enough through tensor busy CSR is internal,
        // so count cycles where its arbiter port or compute is active:
        // approximated by sampling t_busy via tensor_ops delta handled
        // host-side; here count cycles with sidecar port activity:
        tbusy += top.tensor_busy;
    }
};

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Soc soc;
    soc.reset();
    setvbuf(stdout, nullptr, _IOLBF, 0);

    char line[1 << 16];
    while (fgets(line, sizeof line, stdin)) {
        char cmd[16] = {0};
        if (sscanf(line, "%15s", cmd) != 1) continue;
        if (!strcmp(cmd, "LOAD")) {
            uint32_t addr = 0;
            char* p = strchr(line, ' ');
            addr = strtoul(p, &p, 16);
            while (*p == ' ') p++;
            uint32_t a = addr % MEM_SIZE;
            while (hexval(p[0]) >= 0 && hexval(p[1]) >= 0) {
                soc.mem[a++ % MEM_SIZE] =
                    (uint8_t)((hexval(p[0]) << 4) | hexval(p[1]));
                p += 2;
            }
            printf("OK\n");
        } else if (!strcmp(cmd, "PEEK")) {
            uint32_t addr, len;
            sscanf(line, "%*s %x %u", &addr, &len);
            for (uint32_t i = 0; i < len; i++)
                printf("%02x", soc.mem[(addr + i) % MEM_SIZE]);
            printf("\n");
        } else if (!strcmp(cmd, "RESET")) {
            soc.reset();
            printf("OK\n");
        } else if (!strcmp(cmd, "RUN")) {
            uint64_t n = 0;
            sscanf(line, "%*s %llu", (unsigned long long*)&n);
            uint64_t c0 = soc.wb_commits + soc.mem_commits;
            for (uint64_t i = 0; i < n; i++) soc.tick();
            printf("RAN %llu %llu\n", (unsigned long long)n,
                   (unsigned long long)(soc.wb_commits + soc.mem_commits
                                        - c0));
        } else if (!strcmp(cmd, "COUNTERS")) {
            printf("CYCLES %llu COMMITS %llu WBCOMMITS %llu MEMCOMMITS "
                   "%llu DBEATS %llu TENSOROPS %u TBUSY %llu LAT %u\n",
                   (unsigned long long)soc.cycles,
                   (unsigned long long)(soc.wb_commits + soc.mem_commits),
                   (unsigned long long)soc.wb_commits,
                   (unsigned long long)soc.mem_commits,
                   (unsigned long long)soc.dbeats,
                   (unsigned)soc.top.tensor_ops,
                   (unsigned long long)soc.tbusy, soc.lat);
        } else if (!strcmp(cmd, "SCHEDCLR")) {
            soc.sched_clear();
            printf("OK\n");
        } else if (!strcmp(cmd, "SCHEDCLASS")) {
            unsigned i = 0, lo = 0, hi = 0;
            sscanf(line, "%*s %u %x %x", &i, &lo, &hi);
            if (i < 4) { soc.s_clo[i] = lo; soc.s_chi[i] = hi; }
            printf("OK\n");
        } else if (!strcmp(cmd, "SCHED")) {
            printf("SCHED CYC %llu HIST", (unsigned long long)soc.s_cyc);
            for (int i = 0; i < 9; i++)
                printf(" %llu", (unsigned long long)soc.s_hist[i]);
            printf(" W0K");
            for (int i = 0; i < 9; i++)
                printf(" %llu", (unsigned long long)soc.s_w0k[i]);
            for (int w = 0; w < 8; w++) {
                printf(" W%d %llu %llu", w,
                       (unsigned long long)soc.s_wrdy[w],
                       (unsigned long long)soc.s_wiss[w]);
                for (int c = 0; c < 5; c++)
                    printf(" %llu", (unsigned long long)soc.s_wcls[w][c]);
            }
            printf("\n");
        } else if (!strcmp(cmd, "PSTATCLR")) {
            soc.pstat_on = true;
            soc.p_cyc = 0;
            for (int i = 0; i < 32; i++) soc.p_hist[i] = 0;
            printf("OK\n");
        } else if (!strcmp(cmd, "PSTAT")) {
            printf("PSTAT CYC %llu HIST", (unsigned long long)soc.p_cyc);
            for (int i = 0; i < 32; i++)
                printf(" %llu", (unsigned long long)soc.p_hist[i]);
            printf("\n");
        } else if (!strcmp(cmd, "LAT")) {
            sscanf(line, "%*s %u", &soc.lat);
            if (soc.lat < 1) soc.lat = 1;
            printf("OK\n");
        } else if (!strcmp(cmd, "QUIT")) {
            printf("BYE\n");
            break;
        } else {
            printf("ERR unknown %s\n", cmd);
        }
    }
    return 0;
}
