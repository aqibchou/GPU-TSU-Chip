#!/bin/zsh
# Report whether the local machine can run the chip verification stack.
ROOT="${GPU_TSU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TOOL_ROOT="${GPU_TSU_TOOL_ROOT:-$ROOT}"
fail=0

pass() { printf "PASS  %-22s %s\n" "$1" "$2"; }
bad()  { printf "FAIL  %-22s %s\n" "$1" "$2"; fail=1; }
warn() { printf "WARN  %-22s %s\n" "$1" "$2"; }

if command -v verilator >/dev/null 2>&1; then
  pass verilator "$(verilator --version | head -1)"
else
  bad verilator "not found"
fi

py="$ROOT/.venv/bin/python"
[[ -x "$py" ]] || py="$(command -v python3)"
if [[ -n "$py" && -x "$py" ]]; then
  pass python "$($py -V 2>&1)"
  for mod in cocotb numpy scipy emcee; do
    $py -c "import $mod" >/dev/null 2>&1 && pass "py:$mod" ok || bad "py:$mod" missing
  done
else
  bad python "not found"
fi

gcc="$TOOL_ROOT/tools/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-gcc"
[[ -x "$gcc" ]] && pass riscv-gcc "$($gcc -dumpversion)" || warn riscv-gcc "needed for compiled-kernel gates"
[[ -x "$TOOL_ROOT/tools/spike/bin/spike" ]] && pass spike present || warn spike "needed for P1 lockstep"
[[ -x "$TOOL_ROOT/tools/oss-cad-suite/bin/yosys" ]] && pass yosys present || warn yosys "needed for synthesis checks"
command -v vivado >/dev/null 2>&1 && pass vivado "$(vivado -version 2>/dev/null | head -1)" || warn vivado "needed for K26 implementation"
command -v dtc >/dev/null 2>&1 && pass dtc present || warn dtc "needed for the SG0 overlay"

exit $fail
