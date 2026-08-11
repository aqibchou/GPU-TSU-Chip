/* PR4 probe (tensor_spec §1c, D-036): read T_PROFILE, optionally GO
   an op expected ABSENT in this profile. The absence trap is mcause 2
   at the GO site, surfaced by crt0 as assert code 0x80000002; the
   gate asserts exactly that. The probe NEVER GOes a present op (a
   garbage-operand command would launch for real).
     arg0 = result addr (res[0] = T_PROFILE; res[1] = 0xD0 only if the
            probed GO unexpectedly proceeded)
     arg1 = T_OP to probe, or 0xFFFF = read-only run */
#include "mk.h"

#define MK_STRUCT __attribute__((noinline, optimize("O0"), \
                                  section(".text.main")))

MK_STRUCT void kernel_main(u32 tid) {
    if (tid == 0u) {
        volatile u32 *res = (volatile u32 *)mk_arg(0);
        u32 probe = mk_arg(1);
        u32 id;
        __asm__ volatile("csrr %0, 0x8CA" : "=r"(id));
        res[0] = id;
        if (probe != 0xFFFFu) {
            __asm__ volatile("csrw 0x8C0, %0" :: "r"(probe));
            __asm__ volatile("csrwi 0x8C8, 1");
            res[1] = 0xD0u;               /* unreachable if absent */
        }
    }
}
