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
MAX_KEY=4096
KEY_SAMPLER="const(1.0)"

ROUNDS=1
RUNS_PER_ROUND=200
N_WARMUP=5

STORE="block"

KEY_BUDGETS=(1000 10000 100000 1000000 10000000)

CONDITIONS=(
    "gpu_subgroups:bin(1):1:register" "gpu_smem:bin(1):0:workgroup"
    "gpu_subgroups:bin(2):1:register" "gpu_smem:bin(2):0:workgroup"
    "gpu_subgroups:bin(3):1:register" "gpu_smem:bin(3):0:workgroup"
    "gpu_subgroups:bin(4):1:register" "gpu_smem:bin(4):0:workgroup"
    "gpu_subgroups:bin(5):1:register" "gpu_smem:bin(5):0:workgroup"
    "gpu_subgroups:bin(6):1:register" "gpu_smem:bin(6):0:workgroup"
    "gpu_subgroups:bin(7):1:register" "gpu_smem:bin(7):0:workgroup"
    "gpu_subgroups:bin(8):1:register" "gpu_smem:bin(8):0:workgroup"
    "gpu_subgroups:bin(9):1:register" "gpu_smem:bin(9):0:workgroup"
    "gpu_subgroups:bin(10):1:register" "gpu_smem:bin(10):0:workgroup"
    "gpu_subgroups:bin(11):1:register" "gpu_smem:bin(11):0:workgroup"
)
NC=${#CONDITIONS[@]}

RESULTS_ROOT="output/experiment2"

run_condition() {
    local label="$1" bin_sampler="$2" subgroup="$3" memory="$4" budget="$5"
    local results_dir="${RESULTS_ROOT}/keys_${budget}/${label}"
    echo "${BIN}" \
        wbg "${results_dir}" \
        "${bin_sampler}" "${KEY_SAMPLER}" \
        "${SEED}" "${budget}" \
        "${RUNS_PER_ROUND}" "${N_WARMUP}" \
        "${MAX_KEY}" "${subgroup}" "${memory}" "${STORE}"
    "${BIN}" \
        wbg "${results_dir}" \
        "${bin_sampler}" "${KEY_SAMPLER}" \
        "${SEED}" "${budget}" \
        "${RUNS_PER_ROUND}" "${N_WARMUP}" \
        "${MAX_KEY}" "${subgroup}" "${memory}" "${STORE}"
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
            tmp="${rest#*:}"
            subgroup="${tmp%%:*}"
            memory="${rest##*:}"
            echo ">> round $((round + 1))/${ROUNDS}  ${label}  (budget=${budget})"
            run_condition "${label}" "${bin_sampler}" "${subgroup}" "${memory}" "${budget}"
        done
    done
done

echo ">> done. results under ${RESULTS_ROOT}/"
