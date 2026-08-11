/* Per-thread partial sums: partial[tid] = sum of x[i] for its stride
   slice; the host finishes the 64-way reduction. Uniform loops only. */
#include "mk.h"
void kernel_main(u32 tid) {
    const volatile i32 *x = (const volatile i32 *)mk_arg(0);
    volatile i32 *partial = (volatile i32 *)mk_arg(1);
    u32 n = mk_arg(2);
    i32 acc = 0;
    MK_GRID_STRIDE(i, n) {
        acc += x[i];
    }
    partial[tid] = acc;
}
