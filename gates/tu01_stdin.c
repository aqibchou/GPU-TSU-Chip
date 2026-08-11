/* TestU01 stdin driver: pipe raw LE uint32 words in, run a battery.
 * Usage: prng_dump ... | tu01_stdin small|crush
 * Pass = stdout contains "All tests were passed". Built by gates/s1_prng.py.
 * (Secondary battery replacing dieharder — see D-001/D-011.) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unif01.h"
#include "bbattery.h"

static unsigned int next_u32(void) {
    static unsigned char buf[1 << 16];
    static size_t pos = 0, have = 0;
    if (pos + 4 > have) {
        have = fread(buf, 1, sizeof buf, stdin);
        pos = 0;
        if (have < 4) {
            fprintf(stderr, "tu01_stdin: input exhausted\n");
            exit(3);
        }
    }
    unsigned int v;
    memcpy(&v, buf + pos, 4);
    pos += 4;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: tu01_stdin small|crush\n"); return 2; }
    unif01_Gen* g = unif01_CreateExternGenBits("mk_stdin32", next_u32);
    if (!strcmp(argv[1], "small"))      bbattery_SmallCrush(g);
    else if (!strcmp(argv[1], "crush")) bbattery_Crush(g);
    else return 2;
    unif01_DeleteExternGenBits(g);
    return 0;
}
