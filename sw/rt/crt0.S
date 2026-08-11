# crt0 (M17 runtime_spec §1/§2): per-thread stack, kernel_main(tid),
# DONE mailbox, warp-uniform park. Every branch here is warp-uniform.
    .section .text.init
    .globl _start
_start:
    la    t0, _mk_trap           # trap visibility: mcause -> host, park
    csrw  mtvec, t0
    csrr  a0, mhartid            # tid
    li    t0, 0x00200000         # stack top
    slli  t1, a0, 14             # tid * 16KiB (M19: -O0 frames overflowed
                                 # 2KiB into the neighbor hart's stack;
                                 # 64*16KiB = 0x100000..0x200000, free
                                 # since the heap moved to 0x200000+)
    sub   sp, t0, t1
    call  kernel_main            # kernel_main(tid)
    csrr  a0, mhartid
    li    t0, 0x00020100         # DONE mailboxes
    slli  t1, a0, 2
    add   t0, t0, t1
    li    t1, 1
    sw    t1, 0(t0)
1:  j     1b                     # park (uniform: all lanes arrive)

# trap handler: report 0x8000_0000|mcause in the ASSERT slot, then park.
# (host raises KernelAssert with the decoded cause instead of a timeout)
_mk_trap:
    csrr  t0, mhartid
    slli  t0, t0, 2
    li    t1, 0x00022200         # ASSERT_BASE
    add   t0, t0, t1
    csrr  t1, mcause
    li    t2, 0x80000000
    or    t1, t1, t2
    sw    t1, 0(t0)
    # mepc + mtval into the log ring for host-side diagnosis
    csrr  t0, mhartid
    slli  t0, t0, 2
    slli  t1, t0, 5              # tid*4*32
    add   t0, t0, t1             # tid*4*33
    li    t1, 0x00020200         # LOG_BASE
    add   t0, t0, t1
    li    t1, 2
    sw    t1, 0(t0)              # count = 2
    csrr  t1, mepc
    sw    t1, 4(t0)
    csrr  t1, mtval
    sw    t1, 8(t0)
2:  j     2b
