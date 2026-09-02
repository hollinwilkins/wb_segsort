#!/usr/bin/env sh
#
# Runs every valid single-kernel benchmark listed in the experiments CSV,
# writing one <root>/<name>.csv per experiment.
#
# Usage:
#   ./run_kernel_benchmarks.sh <output_root_dir> [n_keys] [runs] [experiments_csv]
#
# n_keys defaults to 1048576 (or the KEYS env var); runs defaults to 50
# (or the RUNS env var); experiments_csv defaults to
# <repo>/experiments/kernel_experiments.csv (or the EXPERIMENTS_CSV env var).
#
# The experiment list is the set of kernel configs valid for THIS device
# (subgroup=32, 32KB workgroup storage). Regenerate it with
# scratch/gen_kernel_experiments.py if the device or kernel set changes.
#
# Tunables (override via environment):
#   EXPERIMENTS_CSV  path to the experiment list   (default: <repo>/experiments/kernel_experiments.csv; overridden by the experiments_csv argument)
#   BIN              benchmark binary              (default: <repo>/build-debug/wb_benchmark_kernel)
#   SEED             RNG seed                      (default: 1)
#   RUNS             timed iterations per config   (default: 50; overridden by the runs argument)
#   KEYS             total keys in the workload    (default: 1048576; overridden by the n_keys argument)
#   SAMPLER          segment-size sampler          (default: uniform)
#   VALIDATE         if set (non-empty, not 0), validate each kernel once against a
#                    CPU reference sort and exit per-config WITHOUT benchmarking

set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <output_root_dir> [n_keys] [runs] [experiments_csv]" >&2
    exit 1
fi

ROOT_DIR="$1"
KEYS="${2:-${KEYS:-1048576}}"
RUNS="${3:-${RUNS:-50}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS_CSV="${4:-${EXPERIMENTS_CSV:-$SCRIPT_DIR/experiments/kernel_experiments.csv}}"
BIN="${BIN:-$SCRIPT_DIR/build-debug/wb_benchmark_kernel}"
SEED="${SEED:-1}"
SAMPLER="${SAMPLER:-uniform}"

# Validate mode: pass -validate and let the kernel's PASSED/FAILED output through.
validate_flag=""
case "${VALIDATE:-}" in
    ""|0|false|no|FALSE|NO) ;;
    *) validate_flag="-validate" ;;
esac

if [ ! -x "$BIN" ]; then
    echo "benchmark binary not found or not executable: $BIN" >&2
    exit 1
fi
if [ ! -f "$EXPERIMENTS_CSV" ]; then
    echo "experiments csv not found: $EXPERIMENTS_CSV" >&2
    exit 1
fi

mkdir -p "$ROOT_DIR"

# Uses the loop-scoped vars (root, memory, ...). $validate_flag is empty in
# benchmark mode and "-validate" in validate mode.
run_one() {
    "$BIN" "$root" $validate_flag \
        --memory "$memory" --store "$store" \
        --N "$N" --M "$M" --R "$R" --subgroups "$subgroups" \
        --smem "$smem" --bin "$bin" \
        --seed "$SEED" --runs "$RUNS" --keys "$KEYS" --sampler "$SAMPLER"
}

total=0
failed=0

# Read the CSV on fd 3 so the loop stays in this shell (counters persist).
# Columns: name,memory,store,N,M,R,subgroups,smem,bin
exec 3< "$EXPERIMENTS_CSV"
IFS= read -r _header <&3   # discard header row
while IFS=, read -r name memory store N M R subgroups smem bin <&3; do
    [ -n "$name" ] || continue
    total=$((total + 1))
    root="$ROOT_DIR/$name"

    printf '[%3d] %s\n' "$total" "$name"

    # Validate mode surfaces the kernel's PASSED/FAILED output; benchmark mode
    # stays quiet (per-run progress is noisy across hundreds of configs).
    if [ -n "$validate_flag" ]; then
        run_one && status=0 || status=$?
    else
        run_one >/dev/null 2>&1 && status=0 || status=$?
    fi

    if [ "$status" -ne 0 ]; then
        echo "      FAILED (exit $status)" >&2
        failed=$((failed + 1))
    fi
done
exec 3<&-

echo
if [ -n "$validate_flag" ]; then
    echo "done: $total validated, $failed failed"
else
    echo "done: $total experiments, $failed failed -> $ROOT_DIR"
fi
[ "$failed" -eq 0 ]
