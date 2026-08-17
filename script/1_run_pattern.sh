#!/bin/bash

export TFLM_HOME="/home/ajno5/work/2_pattern/tflm/tflite-micro"
export TOOLCHAIN_ARGS="TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.4.0-1/bin/ TARGET_TOOLCHAIN_PREFIX=riscv-none-elf-"

# SE mode, RV64, qemu (SIMULATOR=gem5 is disabled here — see the
# historical SE-mode section at the bottom of this file):
#make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_generic $TOOLCHAIN_ARGS test_hello_world_test

# FS mode, gem5 (default) or whisper:
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal_vector $TOOLCHAIN_ARGS test_dtln_test
make -f ${TFLM_HOME}/tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal SIMULATOR=whisper $TOOLCHAIN_ARGS test_dtln_test