#!/bin/bash
#
# Per-op cycle-count benchmark for dtln_noise_suppression.tflite, using
# tensorflow/lite/micro/tools/benchmarking/generic_model_benchmark.cc
# (make target: run_tflm_benchmark) instead of test_dtln_test.
#
# Unlike test_dtln_test, this harness wires a tflite::MicroProfiler into
# MicroInterpreter, so its log has real per-op tick counts (not test_dtln_test's
# bare pass/fail) — see doc/performance.md's per-op table and caveat for why
# test_dtln_test alone never shows that data. Must be run with cwd ==
# $TFLM_HOME (tflite-micro/): the Makefile resolves its own internal paths
# relative to the current working directory, not its own location.
#
# Covers all four combinations already recorded in doc/performance.md:
#   scalar FULLY_CONNECTED (TARGET=riscv64_baremetal)        x {gem5, whisper}
#   vectorized FULLY_CONNECTED (TARGET=riscv64_baremetal_vector) x {gem5, whisper}
#
# Optional first arg filters which TARGET to run: "scalar", "vector", or
# omit for both (default, matches doc/performance.md's full table).
#
# Output lands in test/output/gen/<target>_aarch64_default_gcc/bin/tflm_benchmark/
# {logs_gem5.txt,logs_whisper.txt} — each run overwrites its own target's log
# in place, so scalar vs. vector don't collide with each other, but reruns of
# the same TARGET/SIMULATOR combo do.

export TFLM_HOME="/home/ajno5/work/2_pattern/tflm/tflite-micro"
export TOOLCHAIN_ARGS="TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.4.0-1/bin/ TARGET_TOOLCHAIN_PREFIX=riscv-none-elf-"

MODEL=tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite
ARENA_SIZE=16384
FILTER="${1:-both}"

if [[ "$FILTER" == "scalar" || "$FILTER" == "both" ]]; then
  # Scalar baseline FULLY_CONNECTED, gem5 (default SIMULATOR) then whisper:
  make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal $TOOLCHAIN_ARGS \
    BUILD_TYPE=default run_tflm_benchmark \
    GENERIC_BENCHMARK_MODEL_PATH=${MODEL} GENERIC_BENCHMARK_ARENA_SIZE=${ARENA_SIZE}
  make -f ${TFLM_HOME}/tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal SIMULATOR=whisper $TOOLCHAIN_ARGS \
    BUILD_TYPE=default run_tflm_benchmark \
    GENERIC_BENCHMARK_MODEL_PATH=${MODEL} GENERIC_BENCHMARK_ARENA_SIZE=${ARENA_SIZE}
fi

if [[ "$FILTER" == "vector" || "$FILTER" == "both" ]]; then
  # Vectorized FULLY_CONNECTED (-march=...zve64x), gem5 then whisper:
  make -f ${TFLM_HOME}/tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal_vector $TOOLCHAIN_ARGS \
    BUILD_TYPE=default run_tflm_benchmark \
    GENERIC_BENCHMARK_MODEL_PATH=${MODEL} GENERIC_BENCHMARK_ARENA_SIZE=${ARENA_SIZE}
  make -f ${TFLM_HOME}/tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal_vector SIMULATOR=whisper $TOOLCHAIN_ARGS \
    BUILD_TYPE=default run_tflm_benchmark \
    GENERIC_BENCHMARK_MODEL_PATH=${MODEL} GENERIC_BENCHMARK_ARENA_SIZE=${ARENA_SIZE}
fi
