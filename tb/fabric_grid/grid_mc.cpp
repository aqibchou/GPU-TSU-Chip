// S3/S4 heavy-run harness for fabric_grid: load a graph + schedule, free-run,
// optionally dump the state every K sweeps (host computes energies/cuts),
// report counters (flips/updates/cycles — the sigma flips-per-second fuel).
//
// Binary input formats (all little-endian), produced by the gate scripts:
//   +rows=F   : per site, 9 x u32 — slots 0-7 {valid<<23|nbr<<10|(J&0x3FF)},
//               then bias raw (u32, low 10 bits)
//   +ord=F    : u16 per entry (order list)
//   +cb=F     : 16 x {u16 start, u16 end}
//   +sched=F  : n x u32 {beta<<24 | sweeps}
//   +seeds=F  : P x 4 u32
//   +sinit=F  : N/8 bytes (optional)
// Args: +n=SITES +nord= +ncol= +nsched= +bipolar=0/1 +dump=F +every=K
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vfabric_grid.h"
#include "verilated.h"

static const int P = 8;

static std::string plus_arg(int argc, char** argv, const char* key) {
    std::string pre = std::string("+") + key + "=";
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], pre.c_str(), pre.size())) return argv[i] + pre.size();
    return "";
}

static std::vector<uint8_t> slurp(const std::string& p) {
    std::vector<uint8_t> v;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) v.clear();
    fclose(f);
    return v;
}

static void tick(Vfabric_grid& t) {
    t.clk = 1; t.eval();
    t.clk = 0; t.eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto rows = slurp(plus_arg(argc, argv, "rows"));
    auto ord = slurp(plus_arg(argc, argv, "ord"));
    auto cb = slurp(plus_arg(argc, argv, "cb"));
    auto sched = slurp(plus_arg(argc, argv, "sched"));
    auto seeds = slurp(plus_arg(argc, argv, "seeds"));
    auto sinit = slurp(plus_arg(argc, argv, "sinit"));
    int n = atoi(plus_arg(argc, argv, "n").c_str());
    int nord = atoi(plus_arg(argc, argv, "nord").c_str());
    int ncol = atoi(plus_arg(argc, argv, "ncol").c_str());
    int nsched = atoi(plus_arg(argc, argv, "nsched").c_str());
    int bipolar = atoi(plus_arg(argc, argv, "bipolar").c_str());
    std::string dumpf = plus_arg(argc, argv, "dump");
    long long every = atoll(plus_arg(argc, argv, "every").c_str());
    if (!n || rows.empty() || ord.empty() || cb.empty() || sched.empty() || seeds.empty()) {
        fprintf(stderr, "missing inputs\n");
        return 2;
    }

    Vfabric_grid top;
    top.clk = 0; top.rst_n = 0;
    top.row_we = 0; top.ord_we = 0; top.cb_we = 0; top.sch_we = 0;
    top.stw_we = 0; top.seed_we = 0; top.start = 0;
    top.bipolar = bipolar;
    for (int i = 0; i < 4; i++) tick(top);
    top.rst_n = 1;

    const uint32_t* rw = (const uint32_t*)rows.data();
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 9; k++) {
            top.row_we = 1; top.row_addr = i; top.row_slot = k;
            top.row_data = rw[i * 9 + k] & 0xFFFFFF;
            tick(top);
        }
    top.row_we = 0;
    const uint16_t* ow = (const uint16_t*)ord.data();
    for (int a = 0; a < nord; a++) {
        top.ord_we = 1; top.ord_addr = a; top.ord_data = ow[a];
        tick(top);
    }
    top.ord_we = 0;
    const uint16_t* cw = (const uint16_t*)cb.data();
    for (int c = 0; c < 16; c++) {
        top.cb_we = 1; top.cb_idx = c; top.cb_start = cw[c * 2]; top.cb_end = cw[c * 2 + 1];
        tick(top);
    }
    top.cb_we = 0;
    top.n_colors = ncol;
    const uint32_t* sw = (const uint32_t*)sched.data();
    for (int e = 0; e < nsched; e++) {
        top.sch_we = 1; top.sch_idx = e;
        top.sch_beta = sw[e] >> 24; top.sch_sweeps = sw[e] & 0xFFFFFF;
        tick(top);
    }
    top.sch_we = 0;
    top.n_sched = nsched;
    const uint32_t* qw = (const uint32_t*)seeds.data();
    for (int s = 0; s < P; s++)
        for (int k = 0; k < 4; k++) {
            top.seed_we = 1; top.seed_stream = s; top.seed_sel = k;
            top.seed_word = qw[s * 4 + k];
            tick(top);
        }
    top.seed_we = 0;
    for (int w = 0; w < 256; w++) {
        uint32_t v = 0;
        if (!sinit.empty())
            for (int b = 0; b < 4; b++)
                if ((size_t)(w * 4 + b) < sinit.size()) v |= (uint32_t)sinit[w * 4 + b] << (8 * b);
        top.stw_we = 1; top.stw_addr = w; top.stw_data = v;
        tick(top);
    }
    top.stw_we = 0;

    FILE* df = nullptr;
    if (!dumpf.empty()) df = fopen(dumpf.c_str(), "wb");
    int nbytes = (n + 7) / 8;

    top.start = 1; tick(top); top.start = 0;
    long long sweep = 0;
    while (top.busy) {
        tick(top);
        if (top.sweep_pulse) {
            sweep++;
            if (df && every && (sweep % every) == 0) {
                std::vector<uint8_t> st(nbytes);
                for (int b = 0; b < nbytes; b++) {
                    int word = b / 4, off = (b % 4) * 8;
                    st[b] = (top.state_flat[word] >> off) & 0xFF;
                }
                fwrite(st.data(), 1, nbytes, df);
            }
        }
    }
    if (df) fclose(df);
    printf("sweeps=%lld upd=%llu flip=%llu cycles=%llu\n", sweep,
           (unsigned long long)top.upd_cnt, (unsigned long long)top.flip_cnt,
           (unsigned long long)top.cycle_cnt);
    top.final();
    return 0;
}
