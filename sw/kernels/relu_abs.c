/* out[i] = x[i] > 0 ? x[i] : -x[i]  interleaved with a relu — dense
   FORWARD divergence per lane (the divergence-coverage kernel). */
#include "mk.h"
void kernel_main(u32 tid) {
    (void)tid;
    const volatile i32 *x = (const volatile i32 *)mk_arg(0);
    volatile i32 *out = (volatile i32 *)mk_arg(1);
    u32 n = mk_arg(2);
    MK_GRID_STRIDE(i, n) {
        i32 v = x[i];
        if (v < 0) {
            if (i & 1) v = -v;           /* abs on odd indices */
            else       v = 0;            /* relu on even */
        }
        out[i] = v;
    }
}
