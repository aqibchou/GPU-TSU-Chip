/* FASTPATH leg D chain workload: n_ops serving-chain ops issued
   back-to-back by tid0 (GEMV, GELU, GEMV, GELU, ...), each through
   the mk_* issue-and-wait discipline. The leg-D statistic is the
   CHAIN-LENGTH DIFFERENTIAL (the stage-C2 lesson): comparing n_ops=3
   against n_ops=1 cancels boot, the pre-first-op setup, and the
   tail completion-detect exactly, leaving the inter-op gaps plus the
   added ops' busy time — and TBUSY (harness) removes the latter.
   args: A, B, C, LUTDST, n_ops, out (t0, t1), mode
   (0 = wait per op, the pre-D discipline; 1 = post all then one
   wait, the D-032d queued discipline). */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u) {
        u32 a = mk_arg(0), b = mk_arg(1), c = mk_arg(2);
        u32 ld = mk_arg(3), n = mk_arg(4), mode = mk_arg(6);
        volatile u32 *out = (volatile u32 *)mk_arg(5);
        u32 t0 = mk_mcycle();
        for (u32 i = 0u; i < n; i++) {
            if (mode == 0u) {
                if ((i & 1u) == 0u)
                    mk_gemm8(a, b, c, 64u, 1u, 64u, 0u);
                else
                    mk_lut(MK_T_GELU, a, ld, 64u);
            } else {
                if ((i & 1u) == 0u)
                    mk_t_post(MK_T_GEMM8, a, b, c, 64u, 1u, 64u, 0u);
                else
                    mk_t_post(MK_T_GELU, a, 0u, ld, 64u, 0u, 0u, 0u);
            }
        }
        if (mode != 0u) mk_t_wait();
        u32 t1 = mk_mcycle();
        out[0] = t0;
        out[1] = t1;
    } else {
        mk_t_wait();
    }
}
