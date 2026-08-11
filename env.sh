#!/bin/zsh
# Source from any directory: source /path/to/GPU-TSU-Chip/env.sh
GPU_TSU_ROOT="${GPU_TSU_ROOT:-$(cd "$(dirname "${(%):-%N}")" && pwd)}"
export GPU_TSU_ROOT
export MK="$GPU_TSU_ROOT"       # compatibility with the original test stack
export GPU_TSU_TOOL_ROOT="${GPU_TSU_TOOL_ROOT:-$GPU_TSU_ROOT}"

[[ -f "$GPU_TSU_ROOT/.venv/bin/activate" ]] && \
  source "$GPU_TSU_ROOT/.venv/bin/activate"

export PATH="$GPU_TSU_TOOL_ROOT/tools/xpack-riscv-none-elf-gcc/bin:$GPU_TSU_TOOL_ROOT/tools/spike/bin:$GPU_TSU_TOOL_ROOT/tools/bin:$PATH"
export PATH="$PATH:$GPU_TSU_TOOL_ROOT/tools/oss-cad-suite/bin"
export PYTHONPATH="$GPU_TSU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GPU_TSU_CACHE_DIR="${GPU_TSU_CACHE_DIR:-$GPU_TSU_ROOT/data/kernel_cache}"

# Apple Command Line Tools occasionally omit the default libc++ include path.
if [[ -d /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 ]]; then
  export CPLUS_INCLUDE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
