#!/usr/bin/env bash
#
# Experiment 1: CPU vs GPU (with subgroups) vs GPU (without subgroups).
# H0: the three sort implementations take the same amount of time.
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
BIN_SAMPLER="uniform"
KEY_SAMPLER="uniform"

ROUNDS=20
RUNS_PER_ROUND=10
N_WARMUP=5

STORE="block"

KEY_BUDGETS=(1000 10000 100000 1000000 10000000)

# label : kind : memory   (memory is ignored by the wbc/CPU path)
CONDITIONS=("cpu:wbc:workgroup" "gpu_subgroups:wbg:register" "gpu_smem:wbg:workgroup")
NC=${#CONDITIONS[@]}

RESULTS_ROOT="output/experiment1"

run_condition() {
    local label="$1" kind="$2" memory="$3" budget="$4"
    local results_dir="${RESULTS_ROOT}/keys_${budget}/${label}"
    "${BIN}" \
        --kind "${kind}" --output "${results_dir}" \
        --sampler "${BIN_SAMPLER}" --key-sampler "${KEY_SAMPLER}" \
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
            kind="${rest%%:*}"
            memory="${rest##*:}"
            echo ">> round $((round + 1))/${ROUNDS}  ${label}  (budget=${budget})"
            run_condition "${label}" "${kind}" "${memory}" "${budget}"
        done
    done
done

echo ">> done. results under ${RESULTS_ROOT}/"
