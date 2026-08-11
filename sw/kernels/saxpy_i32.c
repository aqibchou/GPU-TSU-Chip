/* y[i] = alpha*x[i] + y[i], integer alpha via shift-add (RV32I: no MUL).
   The mul loop is warp-uniform (alpha is a scalar arg). */
#include "mk.h"
void kernel_main(u32 tid) {
    (void)tid;
    const volatile i32 *x = (const volatile i32 *)mk_arg(0);
    volatile i32 *y = (volatile i32 *)mk_arg(1);
    i32 alpha = (i32)mk_arg(2);
    u32 n = mk_arg(3);
    MK_GRID_STRIDE(i, n) {
        y[i] = mk_mul(x[i], alpha) + y[i];   /* mk_mul is transform-proof */
    }
}
