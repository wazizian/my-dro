#!/usr/bin/env bash
# Shard the LSV GPU-study sweep across two GPUs (default: GPU 2 and GPU 3) by method: cvar +
# relaxed_kl_wdro on one GPU, relaxed_kl_wdro_convex (heavier per-step, extra penalty term) alone
# on the other, for rough compute balance. Precaches all datasets single-process first --
# load_or_simulate only writes its cache file on a miss, so two processes racing to simulate the
# same (seed, tag, n) dataset concurrently against a shared --cache-dir could corrupt/duplicate
# the write; precaching once up front makes every later load_or_simulate call a cache hit
# (read-only), so the two GPU processes never touch the same file at once.
#
# Override any parameter via environment variables, e.g.:
#   GPU_A=0 GPU_B=1 N_VALUES="10 20" SEEDS="0" STEPS=20 K=2 N_MAX=20 N_TEST=40 \
#     CACHE_DIR=/tmp/lsv_smoke ./launch_gpu_sweep_2gpu.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-.venv/bin/python}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
METHODS_A="${METHODS_A:-cvar relaxed_kl_wdro}"
METHODS_B="${METHODS_B:-relaxed_kl_wdro_convex}"
N_VALUES="${N_VALUES:-100 250 500 750 1000}"
SEEDS="${SEEDS:-0 1 2}"
K="${K:-4}"
STEPS="${STEPS:-500}"
N_MAX="${N_MAX:-1000}"
N_TEST="${N_TEST:-20000}"
N_TIME="${N_TIME:-100}"
CACHE_DIR="${CACHE_DIR:-lsv_cache}"
OUTPUT_CSV_A="${OUTPUT_CSV_A:-lsv_gpu_sweep_results_a.csv}"
OUTPUT_CSV_B="${OUTPUT_CSV_B:-lsv_gpu_sweep_results_b.csv}"
OUTPUT_CSV_MERGED="${OUTPUT_CSV_MERGED:-lsv_gpu_sweep_results.csv}"
TABLE_OUT="${TABLE_OUT:-lsv_gpu_sweep_table.txt}"
LOG_A="${LOG_A:-lsv_gpu_sweep_a.log}"
LOG_B="${LOG_B:-lsv_gpu_sweep_b.log}"

echo "gpu_a=${GPU_A} methods_a=${METHODS_A}"
echo "gpu_b=${GPU_B} methods_b=${METHODS_B}"
echo "n_values=${N_VALUES} seeds=${SEEDS}"
echo "k=${K} steps=${STEPS} n_max=${N_MAX} n_test=${N_TEST} n_time=${N_TIME}"
echo "cache_dir=${CACHE_DIR}"

echo "=== precaching datasets (single process, avoids concurrent-write race) ==="
# shellcheck disable=SC2086
"${PYTHON}" lsv_sweep.py --precache \
    --seeds ${SEEDS} \
    --n-max "${N_MAX}" \
    --n-test "${N_TEST}" \
    --n-time "${N_TIME}" \
    --cache-dir "${CACHE_DIR}"

echo "=== launching GPU ${GPU_A} (${METHODS_A}) and GPU ${GPU_B} (${METHODS_B}) ==="
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU_A}" "${PYTHON}" lsv_sweep.py \
    --methods ${METHODS_A} \
    --n-values ${N_VALUES} \
    --seeds ${SEEDS} \
    --k "${K}" \
    --steps "${STEPS}" \
    --n-max "${N_MAX}" \
    --n-test "${N_TEST}" \
    --n-time "${N_TIME}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${OUTPUT_CSV_A}" \
    > "${LOG_A}" 2>&1 &
PID_A=$!

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU_B}" "${PYTHON}" lsv_sweep.py \
    --methods ${METHODS_B} \
    --n-values ${N_VALUES} \
    --seeds ${SEEDS} \
    --k "${K}" \
    --steps "${STEPS}" \
    --n-max "${N_MAX}" \
    --n-test "${N_TEST}" \
    --n-time "${N_TIME}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${OUTPUT_CSV_B}" \
    > "${LOG_B}" 2>&1 &
PID_B=$!

echo "gpu ${GPU_A} pid=${PID_A}, log=${LOG_A}"
echo "gpu ${GPU_B} pid=${PID_B}, log=${LOG_B}"
wait "${PID_A}" "${PID_B}"

echo "=== merging results ==="
head -n 1 "${OUTPUT_CSV_A}" > "${OUTPUT_CSV_MERGED}"
tail -n +2 "${OUTPUT_CSV_A}" >> "${OUTPUT_CSV_MERGED}"
tail -n +2 "${OUTPUT_CSV_B}" >> "${OUTPUT_CSV_MERGED}"

"${PYTHON}" lsv_table.py "${OUTPUT_CSV_MERGED}" --output "${TABLE_OUT}"

echo "results csv: ${OUTPUT_CSV_MERGED}"
echo "results table: ${TABLE_OUT}"
