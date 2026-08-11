/* M17 kernel image: code at 0 (RESET_PC), everything below the param
   block at 0x20000 (runtime_spec §2). */
ENTRY(_start)
MEMORY { CODE (rwx) : ORIGIN = 0x00000000, LENGTH = 0x20000 }
SECTIONS {
  .text : { *(.text.init) *(.text.mk0) *(.text) *(.text.main) } > CODE
  .rodata : { *(.rodata*) *(.srodata*) } > CODE
  .data : { *(.data*) *(.sdata*) } > CODE
  .bss : { *(.bss*) *(.sbss*) *(COMMON) } > CODE
  /DISCARD/ : { *(.comment) *(.note*) *(.riscv.attributes) *(.eh_frame*) }
}
