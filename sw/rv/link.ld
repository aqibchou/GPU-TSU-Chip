/* Layout for rvfuzz / smoke programs (riscv-tests use their own env/p script).
   tohost sits below text so arbitrarily large straight-line fuzz programs
   never collide with it; data trails text wherever it ends. */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .tohost : { *(.tohost) }
  . = 0x80001000;
  .text : { *(.text*) }
  .data : ALIGN(4096) { *(.data*) *(.sdata*) }
  .bss  : { *(.bss*) *(.sbss*) }
  /DISCARD/ : { *(.riscv.attributes) *(.comment) *(.eh_frame) }
}
