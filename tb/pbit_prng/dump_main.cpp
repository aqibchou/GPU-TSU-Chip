// prng_dump — stream the RTL farm's words to stdout as raw LE bytes, at
// Verilator native speed, for PractRand / TestU01 consumption (Σ.4c).
//
// Args:
//   +seeds=FILE   binary: NSTREAMS * 4 LE uint32 initial states (from golden)
//   +mode=single|interleave
//   +stream=K     (single mode) which stream's words to emit
//   +bytes=N      stop after N bytes (0 = until SIGPIPE/kill)
//   +selftest     emit first 8 words per stream as hex text, then exit
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vprng_farm.h"
#include "verilated.h"

static const int NSTREAMS = 16;  // must match prng_farm default

static std::string plus_arg(int argc, char** argv, const char* key) {
    std::string pre = std::string("+") + key;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], pre.c_str())) return "1";
        if (!strncmp(argv[i], (pre + "=").c_str(), pre.size() + 1))
            return argv[i] + pre.size() + 1;
    }
    return "";
}

static void tick(Vprng_farm& t) {
    t.clk = 1; t.eval();
    t.clk = 0; t.eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string seeds = plus_arg(argc, argv, "seeds");
    std::string mode = plus_arg(argc, argv, "mode");
    int stream = atoi(plus_arg(argc, argv, "stream").c_str());
    unsigned long long budget = strtoull(plus_arg(argc, argv, "bytes").c_str(), nullptr, 0);
    bool selftest = !plus_arg(argc, argv, "selftest").empty();
    bool single = (mode != "interleave");
    if (seeds.empty()) { fprintf(stderr, "need +seeds=\n"); return 2; }

    uint32_t st[NSTREAMS][4];
    FILE* sf = fopen(seeds.c_str(), "rb");
    if (!sf || fread(st, 4, NSTREAMS * 4, sf) != NSTREAMS * 4) {
        fprintf(stderr, "bad seeds file\n");
        return 2;
    }
    fclose(sf);

    Vprng_farm top;
    top.rst_n = 0; top.step = 0; top.seed_we = 0;
    top.seed_stream = 0; top.seed_sel = 0; top.seed_word = 0;
    for (int i = 0; i < 4; i++) tick(top);
    top.rst_n = 1;
    for (int i = 0; i < NSTREAMS; i++)
        for (int s = 0; s < 4; s++) {
            top.seed_we = 1; top.seed_stream = i; top.seed_sel = s;
            top.seed_word = st[i][s];
            tick(top);
        }
    top.seed_we = 0;
    tick(top);

    if (selftest) {
        std::vector<std::vector<uint32_t>> first(NSTREAMS);
        for (int w = 0; w < 8; w++) {
            for (int i = 0; i < NSTREAMS; i++) first[i].push_back(top.rnd[i]);
            top.step = 1; tick(top); top.step = 0;
        }
        for (int i = 0; i < NSTREAMS; i++) {
            for (int w = 0; w < 8; w++) printf("%d %d %08x\n", i, w, first[i][w]);
        }
        return 0;
    }

    static uint32_t buf[16384];
    size_t fill = 0;
    unsigned long long emitted = 0;
    top.step = 1;
    while (!budget || emitted < budget) {
        if (single) {
            buf[fill++] = top.rnd[stream];
            tick(top);
        } else {
            for (int i = 0; i < NSTREAMS && fill < 16384; i++) buf[fill++] = top.rnd[i];
            tick(top);
        }
        if (fill >= 16384 - NSTREAMS) {
            if (fwrite(buf, 4, fill, stdout) != fill) break;  // SIGPIPE/EOF
            emitted += 4ull * fill;
            fill = 0;
        }
    }
    if (fill) fwrite(buf, 4, fill, stdout);
    return 0;
}
