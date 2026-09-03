#!/usr/bin/env sh
#
# Builds the release (Dawn) benchmark binary with CMake and runs it,
# passing every argument through to wb_benchmark_kernel.
#
# Usage:
#   ./run_kernel_benchmarks.sh [args...]   # forwarded verbatim to the binary

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/build-release/wb_benchmark_kernel"

cmake --preset release
cmake --build --preset release --target wb_benchmark_kernel

exec "$BIN" "$@"
