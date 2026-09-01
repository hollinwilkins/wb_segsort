#!/usr/bin/env bash
#
# Experiment 1: GPU Subgroups vs Workgroup Memory
# H0: the two sorts take the same amount of time for every bin
#

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-release"
BIN="${BUILD_DIR}/wb_benchmark"

if [ ! -d "${BUILD_DIR}" ]; then
    echo ">> configuring ${BUILD_DIR} (Release)"
    cmake -S . -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release
fi
echo ">> building wb_benchmark (Release)"
cmake --build "${BUILD_DIR}" --target wb_benchmark

SEED=42
KEY_SAMPLER="const(1.0)"

ROUNDS=1
RUNS_PER_ROUND=200
N_WARMUP=5

STORE="block"

KEY_BUDGETS=(1000 10000 100000 1000000 10000000)

# label : sampler : memory
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
NC=${#CONDITIONS[@]}

RESULTS_ROOT="output/experiment2"

run_condition() {
    local label="$1" bin_sampler="$2" memory="$3" budget="$4"
    local results_dir="${RESULTS_ROOT}/keys_${budget}/${label}"
    echo "${BIN}" \
        --kind wbg --output "${results_dir}" \
        --sampler "${bin_sampler}" --key-sampler "${KEY_SAMPLER}" \
        --seed "${SEED}" --keys "${budget}" \
        --runs "${RUNS_PER_ROUND}" --warmup-runs "${N_WARMUP}" \
        --memory "${memory}" --store "${STORE}"
    "${BIN}" \
        --kind wbg --output "${results_dir}" \
        --sampler "${bin_sampler}" --key-sampler "${KEY_SAMPLER}" \
        --seed "${SEED}" --keys "${budget}" \
        --runs "${RUNS_PER_ROUND}" --warmup-runs "${N_WARMUP}" \
        --memory "${memory}" --store "${STORE}"
}

for budget in "${KEY_BUDGETS[@]}"; do
    echo "==== key_budget=${budget} ===="
    for (( round = 0; round < ROUNDS; round++ )); do
        for (( j = 0; j < NC; j++ )); do
            idx=$(( (round + j) % NC ))
            cond="${CONDITIONS[$idx]}"
            label="${cond%%:*}"
            rest="${cond#*:}"
            bin_sampler="${rest%%:*}"
            memory="${rest##*:}"
            echo ">> round $((round + 1))/${ROUNDS}  ${label}  (budget=${budget})"
            run_condition "${label}" "${bin_sampler}" "${memory}" "${budget}"
        done
    done
done

echo ">> done. results under ${RESULTS_ROOT}/"
