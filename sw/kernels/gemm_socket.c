/* G2 kernel: thread 0 drives the tensor sidecar through the D4 socket;
   everyone else waits on the (uniform) busy poll. args: A, B, C, M, N, K,
   acc. */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u)
        mk_gemm8(mk_arg(0), mk_arg(1), mk_arg(2),
                 mk_arg(3), mk_arg(4), mk_arg(5), mk_arg(6));
    else
        mk_t_wait();
}
