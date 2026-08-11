/* S7 suite kernel: thread 0 replays a host-built command table through
   the D4 socket (T_OP 8-10); everyone else waits on the uniform poll.
   Table: n records of 5 words {op, a, m, k, flags}:
     8 PCONFIG  a=image base, m=image words, flags=sections
     9 PSAMPLE  m=sweeps (0=SCHED form), k=beta, flags=RECORD/RESET/IMM
    10 PDRAIN   a=dest base, flags=mode
   args: table, n_cmds. */
#include "mk.h"
void kernel_main(u32 tid) {
    if (tid == 0u) {
        const volatile u32 *t = (const volatile u32 *)mk_arg(0);
        u32 n = mk_arg(1);
        for (u32 i = 0u; i < n; i++) {
            const volatile u32 *r = t + 5u * i;
            u32 op = r[0];
            if (op == MK_S_PCONFIG)      mk_pconfig(r[1], r[2], r[4]);
            else if (op == MK_S_PSAMPLE) mk_psample(r[2], r[3], r[4]);
            else                         mk_pdrain(r[1], r[4]);
        }
    } else {
        mk_t_wait();
    }
}
