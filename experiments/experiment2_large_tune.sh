#!/usr/bin/env bash
#
# Experiment 2 (large / saturated) calibration: determine sort_iterations per
# cell for a single full-saturation key budget (2^28 = 268,435,456 keys, the
# dataset size used by bb_segsort / Hou et al.). Runs each (condition) once with
# -tune and captures the printed --sort-iterations into a TSV.
#

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-release"
BIN="${BUILD_DIR}/wb_benchmark"

if [ ! -d "${BUILD_DIR}" ]; then
    echo ">> configuring ${BUILD_DIR} (Release)" >&2
    cmake -S . -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release >&2
fi
echo ">> building wb_benchmark (Release)" >&2
cmake --build "${BUILD_DIR}" --target wb_benchmark >&2

SEED=42
KEY_SAMPLER="const(1.0)"
STORE="block"

# Full saturation: 2^28 keys, matching bb_segsort's fixed dataset size.
KEY_BUDGETS=(268435456)

# label : sampler : memory   (same as experiment2.sh)
CONDITIONS=(
    "gpu_subgroups:bin(1):register" "gpu_smem:bin(1):workgroup"
    "gpu_subgroups:bin(2):register" "gpu_smem:bin(2):workgroup"
    "gpu_subgroups:bin(3):register" "gpu_smem:bin(3):workgroup"
    "gpu_subgroups:bin(4):register" "gpu_smem:bin(4):workgroup"
    "gpu_subgroups:bin(5):register" "gpu_smem:bin(5):workgroup"
    "gpu_subgroups:bin(6):register" "gpu_smem:bin(6):workgroup"
    "gpu_subgroups:bin(7):register" "gpu_smem:bin(7):workgroup"
    "gpu_subgroups:bin(8):register" "gpu_smem:bin(8):workgroup"
    "gpu_subgroups:bin(9):register" "gpu_smem:bin(9):workgroup"
    "gpu_subgroups:bin(10):register" "gpu_smem:bin(10):workgroup"
    "gpu_subgroups:bin(11):register" "gpu_smem:bin(11):workgroup"
)

OUT="output/experiment2_large/sort_iterations.tsv"
mkdir -p "$(dirname "${OUT}")"

printf 'label\tsampler\tmemory\tbudget\tsort_iterations\n' | tee "${OUT}"

for budget in "${KEY_BUDGETS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        label="${cond%%:*}"
        rest="${cond#*:}"
        bin_sampler="${rest%%:*}"
        memory="${rest##*:}"

        line=$("${BIN}" \
            --kind wbg --output /tmp/segsort_large_tune \
            --sampler "${bin_sampler}" --key-sampler "${KEY_SAMPLER}" \
            --seed "${SEED}" --keys "${budget}" \
            --runs 1 \
            --memory "${memory}" --store "${STORE}" \
            -tune 2>/dev/null | grep 'Tuning params:')

        si=$(printf '%s\n' "${line}" | sed -E 's/.*--sort-iterations ([0-9]+).*/\1/')

        printf '%s\t%s\t%s\t%s\t%s\n' "${label}" "${bin_sampler}" "${memory}" "${budget}" "${si}" | tee -a "${OUT}"
    done
done

echo ">> wrote ${OUT}" >&2
