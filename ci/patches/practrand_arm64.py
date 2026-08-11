#!/usr/bin/env python3
"""D-007: PractRand predates Apple Silicon — it includes <x86intrin.h> for any
__GNUC__ compiler and calls __rdtsc() for entropy/timing. On AArch64 we guard
the include and shim __rdtsc() with the virtual counter (cntvct_el0), which is
userspace-readable on Apple Silicon and serves the same purpose (fast,
monotonic, fine-grained). Idempotent: safe to rerun.
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
f = next(root.glob("PractRand*/src/platform_specifics.cpp"))
text = f.read_text()

OLD = "#ifdef __GNUC__\n#include <x86intrin.h>\n#endif\n"
NEW = (
    "#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))\n"
    "#include <x86intrin.h>\n"
    "#elif defined(__GNUC__) && defined(__aarch64__)\n"
    "/* D-007 arm64 shim: virtual counter stands in for the x86 TSC */\n"
    "static inline unsigned long long __rdtsc(void) {\n"
    "    unsigned long long v;\n"
    "    __asm__ __volatile__(\"mrs %0, cntvct_el0\" : \"=r\"(v));\n"
    "    return v;\n"
    "}\n"
    "#endif\n"
)

if NEW in text:
    print("already patched")
elif OLD in text:
    f.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {f}")
else:
    sys.exit("anchor not found — PractRand source layout changed; patch by hand")

# tools/dummy_rng.h: same unguarded include, no __rdtsc use — narrow the guard.
g = next(root.glob("PractRand*/tools/dummy_rng.h"))
gtext = g.read_text()
G_OLD = "#if defined __GNUC__\n#include <x86intrin.h>\n#endif\n"
G_NEW = ("#if defined __GNUC__ && (defined(__i386__) || defined(__x86_64__))\n"
         "#include <x86intrin.h>\n#endif\n")
if G_NEW in gtext:
    print("dummy_rng.h already patched")
elif G_OLD in gtext:
    g.write_text(gtext.replace(G_OLD, G_NEW, 1))
    print(f"patched {g}")
else:
    sys.exit("dummy_rng.h anchor not found — patch by hand")
