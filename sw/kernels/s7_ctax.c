/* S7.4 border-tax workload C* (sampling_isa_spec §10): an untimed full
   PCONFIG (seeds/order/schedule land once), then iterations of
   PCONFIG(ROWS only) + PSAMPLE(IMM, RECORD) + PDRAIN(MOMENTS), fenced
   with mcycle so B = T_total - S charges config DMA, drain DMA,
   doorbells, polling and kernel overhead against S = the PSAMPLE busy
   window. Stamps per iteration: {t0, t1, t2, t3}.
   args: fimg, fwords, fflags, img, words, cflags, sweeps, beta,
         sflags, drain_dest, stamps_out, iters. */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u) {
        u32 img = mk_arg(3), words = mk_arg(4), cflags = mk_arg(5);
        u32 sweeps = mk_arg(6), beta = mk_arg(7), sflags = mk_arg(8);
        u32 dest = mk_arg(9);
        volatile u32 *out = (volatile u32 *)mk_arg(10);
        u32 iters = mk_arg(11);
        mk_pconfig(mk_arg(0), mk_arg(1), mk_arg(2));   /* untimed full */
        for (u32 it = 0u; it < iters; it++) {
            u32 t0 = mk_mcycle();
            mk_pconfig(img, words, cflags);
            u32 t1 = mk_mcycle();
            mk_psample(sweeps, beta, sflags);
            u32 t2 = mk_mcycle();
            mk_pdrain(dest, 1u);            /* MOMENTS */
            u32 t3 = mk_mcycle();
            out[4u * it + 0u] = t0;
            out[4u * it + 1u] = t1;
            out[4u * it + 2u] = t2;
            out[4u * it + 3u] = t3;
        }
    } else {
        mk_t_wait();
    }
}
