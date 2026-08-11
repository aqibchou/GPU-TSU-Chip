/* out[i] = x[i] + s[i & 7] — every lane of a warp reads the SAME s word
   per iteration in lockstep phases: the coalescing showcase (DBEATS
   counter should show ~1 request per warp for the s loads). */
#include "mk.h"
void kernel_main(u32 tid) {
    (void)tid;
    const volatile i32 *x = (const volatile i32 *)mk_arg(0);
    const volatile i32 *s = (const volatile i32 *)mk_arg(1);
    volatile i32 *out = (volatile i32 *)mk_arg(2);
    u32 n = mk_arg(3);
    MK_GRID_STRIDE(i, n) {
        out[i] = x[i] + s[(i >> 6) & 7];   /* warp-uniform s index */
    }
}
