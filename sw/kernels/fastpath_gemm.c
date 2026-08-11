/* FASTPATH leg C stage C2 bar workload: mcycle-fenced GEMM8 pairs.
   Per iteration: t0; GEMM8(measured shape — load-dominated); t1;
   GEMM8(control shape — compute-dominated); t2. The fences cover the
   whole op (kernel stamps cannot cut inside the engine — disclosed in
   the card); the shape choice makes DMA dominate the measured one.
   args: A, B, C, m1, n1, k1, m2, n2, k2, stamps_out, iters. */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u) {
        volatile u32 *out = (volatile u32 *)mk_arg(9);
        u32 iters = mk_arg(10);
        for (u32 it = 0u; it < iters; it++) {
            u32 t0 = mk_mcycle();
            mk_gemm8(mk_arg(0), mk_arg(1), mk_arg(2),
                     mk_arg(3), mk_arg(4), mk_arg(5), 0u);
            u32 t1 = mk_mcycle();
            mk_gemm8(mk_arg(0), mk_arg(1), mk_arg(2),
                     mk_arg(6), mk_arg(7), mk_arg(8), 0u);
            u32 t2 = mk_mcycle();
            out[3u * it + 0u] = t0;
            out[3u * it + 1u] = t1;
            out[3u * it + 2u] = t2;
        }
    } else {
        mk_t_wait();
    }
}
