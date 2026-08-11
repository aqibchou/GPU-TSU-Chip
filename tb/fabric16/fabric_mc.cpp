// S2 histogram harness: run fabric16 free-running, record the state at every
// thin-th sweep boundary into a 65536-bin histogram. Also counts consecutive
// identical recorded states (f_same) for the pre-registered thinning check.
//
// Args: +jfile= (16x9 int16 LE: slots 0-7 J, slot 8 bias, raw s1.6.3)
//       +seeds= (16x4 u32 LE)  +beta=RAW  +burn=  +thin=  +n=  +hist=OUT
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vfabric16.h"
#include "verilated.h"

static std::string plus_arg(int argc, char** argv, const char* key) {
    std::string pre = std::string("+") + key + "=";
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], pre.c_str(), pre.size())) return argv[i] + pre.size();
    return "";
}

static void tick(Vfabric16& t) {
    t.clk = 1; t.eval();
    t.clk = 0; t.eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string jf = plus_arg(argc, argv, "jfile");
    std::string sf = plus_arg(argc, argv, "seeds");
    std::string hf = plus_arg(argc, argv, "hist");
    int beta = atoi(plus_arg(argc, argv, "beta").c_str());
    long long burn = atoll(plus_arg(argc, argv, "burn").c_str());
    long long thin = atoll(plus_arg(argc, argv, "thin").c_str());
    long long n = atoll(plus_arg(argc, argv, "n").c_str());
    if (jf.empty() || sf.empty() || hf.empty() || !thin || !n) {
        fprintf(stderr, "usage: +jfile= +seeds= +beta= +burn= +thin= +n= +hist=\n");
        return 2;
    }

    int16_t jmem[16][9];
    uint32_t seeds[16][4];
    FILE* f = fopen(jf.c_str(), "rb");
    if (!f || fread(jmem, 2, 16 * 9, f) != 16 * 9) { fprintf(stderr, "bad jfile\n"); return 2; }
    fclose(f);
    f = fopen(sf.c_str(), "rb");
    if (!f || fread(seeds, 4, 16 * 4, f) != 16 * 4) { fprintf(stderr, "bad seeds\n"); return 2; }
    fclose(f);

    Vfabric16 top;
    top.clk = 0; top.rst_n = 0;
    top.cfg_we = 0; top.cfg_site = 0; top.cfg_slot = 0; top.cfg_data = 0;
    top.clamp = 0; top.s_load = 0; top.s_init = 0;
    top.beta = (unsigned)beta & 0xFF; top.bipolar = 0;
    top.seed_we = 0; top.seed_stream = 0; top.seed_sel = 0; top.seed_word = 0;
    top.start = 0; top.sweeps = 0;
    for (int i = 0; i < 4; i++) tick(top);
    top.rst_n = 1;

    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 9; k++) {
            top.cfg_we = 1; top.cfg_site = i; top.cfg_slot = k;
            top.cfg_data = (unsigned)jmem[i][k] & 0x3FF;
            tick(top);
        }
    top.cfg_we = 0;
    for (int i = 0; i < 16; i++)
        for (int s = 0; s < 4; s++) {
            top.seed_we = 1; top.seed_stream = i; top.seed_sel = s;
            top.seed_word = seeds[i][s];
            tick(top);
        }
    top.seed_we = 0;
    top.s_load = 1; top.s_init = 0; tick(top); top.s_load = 0;

    std::vector<uint32_t> hist(65536, 0);
    long long total_sweeps = burn + n * thin;
    top.start = 1; top.sweeps = (unsigned)total_sweeps; tick(top); top.start = 0;

    long long sweep = 0, taken = 0, same = 0;
    int last = -1;
    while (taken < n) {
        tick(top);
        if (top.sweep_pulse) {
            sweep++;
            if (sweep > burn && ((sweep - burn) % thin) == 0) {
                int s = top.s_out;
                hist[s]++;
                if (s == last) same++;
                last = s;
                taken++;
            }
        }
        if (!top.busy && taken < n) { fprintf(stderr, "ended early\n"); return 3; }
    }

    f = fopen(hf.c_str(), "wb");
    fwrite(hist.data(), 4, hist.size(), f);
    fclose(f);
    printf("n=%lld same=%lld\n", taken, same);
    top.final();
    return 0;
}
