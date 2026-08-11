#!/bin/zsh
# Σ.6 stage 2: every declared unit target for every registered testbench.
# Most benches declare smoke + fuzz through tb/common.mk. Integration benches
# may declare only smoke, and input-driven conformance benches may declare no
# generic unit target because their gate supplies the external test image.
# NOTE: the seed variable is deliberately NOT named RANDOM_SEED — cocotb reads
# that env name itself (deprecated) and dies on an empty value.
setopt null_glob
MK="${MK:-$(cd "$(dirname "$0")/.." && pwd)}"
fail=0 found=0
# stage 0: repo-wide warning-clean lint — benches only exercise the tops
# they instantiate, so an un-benched legacy instantiator with a stale
# port list is invisible to every suite below (the nightly caught one;
# the battery must too)
echo "== lint: repo-wide"
zsh $MK/ci/lint.sh > /dev/null || { echo "FAIL lint"; fail=1; }
for d in $MK/tb/*/; do
  [[ -f "$d/Makefile" ]] || continue
  if [[ " ${MK_SKIP_TB:-} " == *" $(basename $d) "* ]]; then
    echo "== $(basename $d): skipped (MK_SKIP_TB)"
    continue
  fi
  targets=$(make -s -C "$d" unit-targets) || {
    echo "FAIL unit-targets $(basename $d)"
    fail=1
    continue
  }
  if [[ -z "$targets" ]]; then
    echo "== $(basename $d): gate-only (no generic unit target)"
    continue
  fi
  found=1
  for target in ${(z)targets}; do
    echo "== $(basename $d): $target"
    make -s -C "$d" "$target" MK_FUZZ_SEED="${MK_FUZZ_SEED:-$(date +%s)}" || {
      echo "FAIL $target $(basename $d)"
      fail=1
    }
  done
done
(( found )) || echo "no testbenches yet — units vacuously green"
exit $fail
