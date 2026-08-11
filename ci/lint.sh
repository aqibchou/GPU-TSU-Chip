#!/bin/zsh
# Σ.6 stage: verilator -Wall lint, zero warnings tolerated (any %Warning fails).
# Each RTL file lints standalone as its own top; -I flags cover cross-includes.
setopt null_glob
MK="${MK:-$(cd "$(dirname "$0")/.." && pwd)}"
files=($MK/rtl/**/*.sv $MK/rtl/**/*.v)
if (( ${#files[@]} == 0 )); then
  echo "no RTL yet — lint vacuously green"
  exit 0
fi
ydirs=()
for d in $MK/rtl/*(N/); do ydirs+=(-y "$d"); done
fail=0
for f in "${files[@]}"; do
  out=$(verilator --lint-only --timing -Wall -I"$MK/rtl" "${ydirs[@]}" "$f" 2>&1)
  rc=$?
  if (( rc != 0 )) || print -r -- "$out" | grep -q "%Warning"; then
    echo "LINT FAIL: $f"
    print -r -- "$out"
    fail=1
  else
    echo "lint clean: ${f#$MK/}"
  fi
done
exit $fail
