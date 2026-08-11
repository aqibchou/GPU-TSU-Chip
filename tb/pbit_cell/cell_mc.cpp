// Monte-Carlo driver for pbit_mc_top: real xoshiro stream -> cell decisions.
// Emits "ones=<k> n=<n>" for the gate's exact binomial test.
// Args: +s0..+s3 (stream state words) +bias=RAW +beta=RAW +n=SAMPLES
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "Vpbit_mc_top.h"
#include "verilated.h"

static std::string plus_arg(int argc, char** argv, const char* key) {
    std::string pre = std::string("+") + key + "=";
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], pre.c_str(), pre.size())) return argv[i] + pre.size();
    return "";
}

static void tick(Vpbit_mc_top& t) {
    t.clk = 1; t.eval();
    t.clk = 0; t.eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    uint32_t st[4];
    for (int i = 0; i < 4; i++)
        st[i] = strtoul(plus_arg(argc, argv, (std::string("s") + char('0' + i)).c_str()).c_str(),
                        nullptr, 0);
    int bias = (int)strtol(plus_arg(argc, argv, "bias").c_str(), nullptr, 0);
    int beta = (int)strtol(plus_arg(argc, argv, "beta").c_str(), nullptr, 0);
    long long n = strtoll(plus_arg(argc, argv, "n").c_str(), nullptr, 0);
    if (!n) { fprintf(stderr, "need +n=\n"); return 2; }

    Vpbit_mc_top top;
    top.clk = 0; top.rst_n = 0;
    top.seed_we = 0; top.seed_sel = 0; top.seed_word = 0;
    top.bipolar = 0; top.acc_clear = 0; top.bias = (unsigned)bias & 0x3FF;
    top.acc_en = 0; top.j_val = 0; top.s_in = 0;
    top.sample_en = 0; top.beta = (unsigned)beta & 0xFF;
    for (int i = 0; i < 4; i++) tick(top);
    top.rst_n = 1;
    for (int i = 0; i < 4; i++) { top.seed_we = 1; top.seed_sel = i; top.seed_word = st[i]; tick(top); }
    top.seed_we = 0;
    tick(top);

    long long ones = 0;
    for (long long k = 0; k < n; k++) {
        top.acc_clear = 1; tick(top);
        top.acc_clear = 0; top.sample_en = 1; tick(top);   // stage 1 + PRNG step
        top.sample_en = 0; tick(top);                      // stage 2
        if (!top.s_valid) { fprintf(stderr, "no s_valid\n"); return 3; }
        ones += top.s_out;
    }
    printf("ones=%lld n=%lld\n", ones, n);
    top.final();
    return 0;
}
