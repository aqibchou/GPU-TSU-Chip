/* S7.5 red path: a socket CSR write with a lane-divergent source must
   trap mcause 2 (the v1 uniform-source rule). Warp 0 writes T_A with
   each lane contributing its own tid. */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid < 8u)
        _MK_CSRW(0x8C1, tid);               /* divergent source -> trap */
}
