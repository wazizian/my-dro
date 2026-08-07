#!/usr/bin/env bash
# Run the LSV GPU-study sweep (see lsv.md, lsv_sweep.py) and produce the summary table.
# Every parameter can be overridden via environment variables, e.g.:
#   N_VALUES="10 20" SEEDS="0" STEPS=20 K=2 N_MAX=20 N_TEST=40 CACHE_DIR=/tmp/lsv_smoke \
#     OUTPUT_CSV=/tmp/lsv_smoke.csv TABLE_OUT=/tmp/lsv_smoke.txt ./launch_gpu_sweep.sh
# GPU_ID selects which single GPU to use via CUDA_VISIBLE_DEVICES (default: 1). E.g. to run on
# GPU 0 instead: GPU_ID=0 ./launch_gpu_sweep.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
METHODS="${METHODS:-cvar relaxed_kl_wdro relaxed_kl_wdro_convex}"
N_VALUES="${N_VALUES:-100 250 500 750 1000}"
SEEDS="${SEEDS:-0 1 2}"
K="${K:-4}"
STEPS="${STEPS:-500}"
N_MAX="${N_MAX:-1000}"
N_TEST="${N_TEST:-20000}"
N_TIME="${N_TIME:-100}"
CACHE_DIR="${CACHE_DIR:-lsv_cache}"
OUTPUT_CSV="${OUTPUT_CSV:-lsv_gpu_sweep_results.csv}"
TABLE_OUT="${TABLE_OUT:-lsv_gpu_sweep_table.txt}"

echo "gpu_id=${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "methods=${METHODS}"
echo "n_values=${N_VALUES}"
echo "seeds=${SEEDS}"
echo "k=${K} steps=${STEPS} n_max=${N_MAX} n_test=${N_TEST} n_time=${N_TIME}"
echo "cache_dir=${CACHE_DIR} output_csv=${OUTPUT_CSV} table_out=${TABLE_OUT}"

# shellcheck disable=SC2086
"${PYTHON}" lsv_sweep.py \
    --methods ${METHODS} \
    --n-values ${N_VALUES} \
    --seeds ${SEEDS} \
    --k "${K}" \
    --steps "${STEPS}" \
    --n-max "${N_MAX}" \
    --n-test "${N_TEST}" \
    --n-time "${N_TIME}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${OUTPUT_CSV}"

"${PYTHON}" lsv_table.py "${OUTPUT_CSV}" --output "${TABLE_OUT}"

echo "results csv: ${OUTPUT_CSV}"
echo "results table: ${TABLE_OUT}"
