/* S7.5 red path (D-032d §1b boundary): GO at FULL must trap mcause
   2 — and GO while merely busy must NOT (it enqueues). Thread 0
   launches a PCONFIG (long: the whole image DMA), enqueues QDEPTH=2
   more commands without waiting (legal since D-032d), then rings the
   doorbell once more at FULL -> trap. args: img, img_words, cflags. */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u) {
        mk_t_cmd(MK_S_PCONFIG, mk_arg(0), 0u, 0u, mk_arg(1), 0u, 0u,
                 mk_arg(2));
        _MK_CSRW(0x8C8, 1u);            /* enqueue #1 (legal now)   */
        _MK_CSRW(0x8C8, 1u);            /* enqueue #2 -> FULL       */
        _MK_CSRW(0x8C8, 1u);            /* GO at FULL -> mcause 2   */
        mk_t_wait();
    } else {
        mk_t_wait();
    }
}
