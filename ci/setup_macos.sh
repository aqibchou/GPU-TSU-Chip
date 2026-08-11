#!/bin/zsh
# Apple Silicon development setup for the chip repository.
set -e
ROOT="${GPU_TSU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TOOLS="$ROOT/tools"

export HOMEBREW_NO_AUTO_UPDATE=1
brew install verilator python@3.12 dtc ccache uv cmake || true
brew install --cask gtkwave || true

mkdir -p "$TOOLS/bin" "$TOOLS/src" "$TOOLS/dl"

if [[ ! -x "$TOOLS/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-gcc" ]]; then
  url=$(curl -s https://api.github.com/repos/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/latest \
    | grep -o '"browser_download_url": *"[^"]*darwin-arm64[^"]*\.tar\.gz"' \
    | head -1 | sed 's/.*"\(https[^\"]*\)"$/\1/')
  curl -sL -o "$TOOLS/dl/xpack.tgz" "$url"
  mkdir -p "$TOOLS/xpack-riscv-none-elf-gcc"
  tar xf "$TOOLS/dl/xpack.tgz" -C "$TOOLS/xpack-riscv-none-elf-gcc" --strip-components=1
  rm "$TOOLS/dl/xpack.tgz"
fi

if [[ ! -x "$TOOLS/oss-cad-suite/bin/yosys" ]]; then
  url=$(curl -s https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest \
    | grep -o '"browser_download_url": *"[^"]*darwin-arm64[^"]*"' \
    | head -1 | sed 's/.*"\(https[^\"]*\)"$/\1/')
  curl -sL -o "$TOOLS/dl/oss-cad.tgz" "$url"
  tar xf "$TOOLS/dl/oss-cad.tgz" -C "$TOOLS"
  rm "$TOOLS/dl/oss-cad.tgz"
fi

if [[ ! -x "$TOOLS/spike/bin/spike" ]]; then
  [[ -d "$TOOLS/src/riscv-isa-sim" ]] || git clone --depth 1 \
    https://github.com/riscv-software-src/riscv-isa-sim "$TOOLS/src/riscv-isa-sim"
  mkdir -p "$TOOLS/src/riscv-isa-sim/build"
  (
    cd "$TOOLS/src/riscv-isa-sim/build"
    ../configure --prefix="$TOOLS/spike"
    make -j 8
    make install
  )
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$ROOT/.venv"
fi
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"

echo "Setup complete. Run: source '$ROOT/env.sh' && zsh '$ROOT/ci/doctor.sh'"
