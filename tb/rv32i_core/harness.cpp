// Program-driven lockstep harness for rv32i_core: pure Verilator C++ (no
// cocotb — ~MHz speed for the 1M-instruction nightly). Loads a flat memory
// image, runs to the tohost store, writes the canonical commit log
// (golden/iss.py format) plus a final "HALT <value>" line.
//
// Args: +bin=PATH +entry=HEX +tohost=HEX +log=PATH +max=CYCLES
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vrv32i_core.h"
#include "verilated.h"

static const uint32_t MEM_BASE = 0x80000000u;
static const size_t MEM_SIZE = 0x800000u;

static std::string plus_arg(int argc, char** argv, const char* key) {
    std::string pre = std::string("+") + key + "=";
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], pre.c_str(), pre.size())) return argv[i] + pre.size();
    return "";
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string bin = plus_arg(argc, argv, "bin");
    std::string log = plus_arg(argc, argv, "log");
    uint32_t entry = strtoul(plus_arg(argc, argv, "entry").c_str(), nullptr, 0);
    uint32_t tohost = strtoul(plus_arg(argc, argv, "tohost").c_str(), nullptr, 0);
    uint64_t max_cycles = strtoull(plus_arg(argc, argv, "max").c_str(), nullptr, 0);
    if (bin.empty() || log.empty() || !entry || !max_cycles) {
        fprintf(stderr, "usage: +bin= +entry= +tohost= +log= +max=\n");
        return 2;
    }

    std::vector<uint8_t> mem(MEM_SIZE, 0);
    FILE* bf = fopen(bin.c_str(), "rb");
    if (!bf) { fprintf(stderr, "cannot open %s\n", bin.c_str()); return 2; }
    size_t got = fread(mem.data(), 1, MEM_SIZE, bf);
    fclose(bf);
    if (!got) { fprintf(stderr, "empty image\n"); return 2; }

    FILE* lf = fopen(log.c_str(), "w");
    if (!lf) { fprintf(stderr, "cannot open %s\n", log.c_str()); return 2; }

    Vrv32i_core top;
    top.reset_pc = entry;
    top.rst_n = 0;
    top.mem_ack = 0;
    top.mem_rdata = 0;
    for (int i = 0; i < 8; i++) { top.clk = !top.clk; top.eval(); }
    top.rst_n = 1;

    bool pend = false, pend_we = false;
    uint32_t pend_addr = 0, pend_wdata = 0;
    uint8_t pend_be = 0;
    int rc = 3;  // 3 = max-cycles hit

    for (uint64_t cyc = 0; cyc < max_cycles; cyc++) {
        // present response to last cycle's request
        if (pend) {
            uint32_t off = pend_addr - MEM_BASE;
            if (off >= MEM_SIZE - 4) { fprintf(stderr, "OOB %08x\n", pend_addr); rc = 4; break; }
            if (pend_we) {
                for (int b = 0; b < 4; b++)
                    if (pend_be & (1 << b)) mem[off + b] = (pend_wdata >> (8 * b)) & 0xFF;
            }
            top.mem_rdata = mem[off] | (mem[off + 1] << 8) | (mem[off + 2] << 16)
                          | ((uint32_t)mem[off + 3] << 24);
            top.mem_ack = 1;
            pend = false;
        } else {
            top.mem_ack = 0;
        }

        top.clk = 1;
        top.eval();

        if (top.cmt_valid) {
            char line[96];
            int n = snprintf(line, sizeof line, "%08x:%08x", top.cmt_pc, top.cmt_instr);
            if (top.cmt_rd)
                n += snprintf(line + n, sizeof line - n, ":x%u=%08x",
                              (unsigned)top.cmt_rd, top.cmt_wdata);
            if (top.cmt_st_valid)
                n += snprintf(line + n, sizeof line - n, ":S@%08x=%08x.%u",
                              top.cmt_st_addr, top.cmt_st_data, (unsigned)top.cmt_st_size);
            fprintf(lf, "%s\n", line);
            if (top.cmt_st_valid && tohost && top.cmt_st_addr == tohost) {
                fprintf(lf, "HALT %08x\n", top.cmt_st_data);
                rc = 0;
                break;
            }
        }

        if (top.mem_req) {
            pend = true;
            pend_we = top.mem_we;
            pend_be = top.mem_be;
            pend_addr = top.mem_addr;
            pend_wdata = top.mem_wdata;
        }

        top.clk = 0;
        top.eval();
    }

    fclose(lf);
    top.final();
    if (rc == 3) fprintf(stderr, "max cycles (%llu) hit\n", (unsigned long long)max_cycles);
    return rc;
}
