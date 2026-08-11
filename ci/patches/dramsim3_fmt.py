#!/usr/bin/env python3
"""D-009: DRAMsim3 vendors fmt v5, whose pre-C++20 char8_t emulation
(`enum char8_t : unsigned char {}`) instantiates std::char_traits<fmt::char8_t> —
a generic template modern libc++ removed. DRAMsim3 itself never touches the
u8/UTF-8 surface, so the minimal fix is making the emulated type an alias of
`char` (which has real char_traits). Idempotent.
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
f = root / "ext/fmt/include/fmt/format.h"
text = f.read_text()

OLD = "enum char8_t: unsigned char {};"
NEW = ("typedef char char8_t;  "
       "/* D-009: was `enum char8_t: unsigned char {}`; libc++ dropped generic char_traits */")

if NEW in text:
    print("already patched")
elif OLD in text:
    f.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {f}")
else:
    sys.exit("anchor not found — fmt version differs; patch by hand")
