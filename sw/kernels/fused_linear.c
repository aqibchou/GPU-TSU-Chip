/* G4: fused quantized linear layer — sidecar GEMM, warp-0 lanes bias+
   quant, sidecar GELU. ALL work in warp 0: intra-warp lockstep makes the
   store->LUT ordering architectural (no cross-warp sync in v1 — the
   multi-warp version is two launches; see
   docs/SOFTWARE_AND_VALIDATION.md#cuda-shaped-programming-model).
   args: x, W, bias, acc, q, y, K, N (N <= 64). */
#include "mk.h"
void kernel_main(u32 tid) {
    u32 xa = mk_arg(0), wa = mk_arg(1), ba = mk_arg(2);
    u32 acca = mk_arg(3), qa = mk_arg(4), ya = mk_arg(5);
    u32 K = mk_arg(6), N = mk_arg(7);
    if (tid >= 8u) return;               /* warp 0 only */
    if (tid == 0u)
        mk_gemm8(xa, wa, acca, 1u, N, K, 0u);
    mk_t_wait();                         /* lockstep: all 8 lanes aligned */
    volatile i32 *acc = (volatile i32 *)acca;
    const volatile i32 *bias = (const volatile i32 *)ba;
    volatile signed char *q = (volatile signed char *)qa;
    for (u32 i = tid; i < 64u; i += 8u) {    /* uniform bound */
        if (i < N) {
            i32 v = (acc[i] + bias[i]) >> 8;
            q[i] = (signed char)mk_clamp(v, -128, 127);
        }
    }
    if (tid == 0u)
        mk_lut(MK_T_GELU, qa, ya, N);    /* lockstep: stores retired */
    mk_t_wait();
}
