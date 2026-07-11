#!/bin/bash

# 1. Path to your toolchain
export TOOLCHAIN=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2

# 2. Get the 64-bit sub-directory name (likely rv64imafdc/lp64d)
export MULTI_DIR=$($TOOLCHAIN/bin/riscv-none-elf-gcc -march=rv64gcv -mabi=lp64d -print-multi-directory)

# 3. Define the specific 64-bit paths
export GCC_LIB_DIR=$(dirname $($TOOLCHAIN/bin/riscv-none-elf-gcc -march=rv64gcv -mabi=lp64d -print-libgcc-file-name))
export LIBC_DIR=$TOOLCHAIN/riscv-none-elf/lib/$MULTI_DIR

# 4. check $PATH and append whisper path
export WHISPER_PATH=/home/ajno5/work/0_simulator/whisper/1_build/whisper/build-Linux/

export GEM5_PATH=/home/ajno5/work/0_simulator/gem5/1_build/gem5/build/RISCV/

if [ -z "$PATH_BACK" ]; then
    export PATH_BACK="$PATH"
fi

PATH=$GEM5_PATH:$WHISPER_PATH:$PATH_BACK
