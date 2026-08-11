/* c[i] = a[i] + b[i] — the hello-world: grid-stride, no divergence. */
#include "mk.h"
void kernel_main(u32 tid) {
    (void)tid;
    const volatile i32 *a = (const volatile i32 *)mk_arg(0);
    const volatile i32 *b = (const volatile i32 *)mk_arg(1);
    volatile i32 *c = (volatile i32 *)mk_arg(2);
    u32 n = mk_arg(3);
    MK_GRID_STRIDE(i, n) {
        c[i] = a[i] + b[i];
    }
}
