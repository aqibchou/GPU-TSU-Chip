/* FASTPATH leg A pilot workload (a): the tid0-serial worst case —
   thread 0 runs a pure integer loop (mcycle-fenced), every other
   hart returns immediately and done-spins in crt0's park. This is
   the shape of the dreamer's serial update phases.
   args: n_iter, out (t0, t1, acc). */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u) {
        u32 n = mk_arg(0);
        volatile u32 *out = (volatile u32 *)mk_arg(1);
        u32 acc = 1u;
        u32 t0 = mk_mcycle();
        for (u32 i = 0u; i < n; i++)
            acc = acc * 1664525u + 1013904223u;    /* ALU-only LCG */
        u32 t1 = mk_mcycle();
        out[0] = t0;
        out[1] = t1;
        out[2] = acc;                              /* keep the loop */
    }
}
