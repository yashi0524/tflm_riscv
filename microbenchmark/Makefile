# Makefile — int8dot_ceiling / fc_bottleneck microbenchmarks for MinorCPU's
# FloatSimd FU. Builds against riscv64_baremetal_vector's own crt0/linker
# script directly -- no TFLM library build needed, matches the by-hand
# commands documented in int8dot_ceiling.c's header comment (kept here
# just so they don't need retyping).
#
# Usage:
#   source ../script/0_env_var_setup.sh
#   make run                                              # GCC 13.4.0-1 (default, authoritative -- see doc/gem5_integration.md)
#   make run TOOLCHAIN_ROOT=../script/clang_wrapper/       # clang-18 cross-check
#   make run-fc-bottleneck                                # builds+runs all 5 fc_bottleneck.c variants, see that file's own header comment
#   make clean

TFLM_VEC_DIR := ../tflite-micro/tensorflow/lite/micro/riscv64_baremetal_vector
SIM_CONFIG   := ../sim_config/gem5_riscv_baremetal_fs.py

# GCC by default; point at ../script/clang_wrapper/ to cross-check with
# clang-18 instead (that wrapper silently drops -mexplicit-relocs, so the
# same flags below work unmodified against either toolchain).
TOOLCHAIN_ROOT ?= $(TOOLCHAIN)/bin/
CC := $(TOOLCHAIN_ROOT)riscv-none-elf-gcc

ARCH_FLAGS := -march=rv64imc_zicsr_zve64x -mabi=lp64 -mcmodel=medany
CFLAGS  := $(ARCH_FLAGS) -mexplicit-relocs -fno-builtin-printf -funsigned-char \
           -fno-delete-null-pointer-checks -fomit-frame-pointer -O2
LDFLAGS := $(ARCH_FLAGS) -mexplicit-relocs -mno-relax -nostartfiles -nostdlib \
           -T $(TFLM_VEC_DIR)/linker_semi.ld

BIN := int8dot_ceiling_riscv

$(BIN): int8dot_ceiling.c $(TFLM_VEC_DIR)/start_semi.S $(TFLM_VEC_DIR)/linker_semi.ld
	$(CC) $(CFLAGS) -c int8dot_ceiling.c -o int8dot_ceiling.o
	$(CC) $(LDFLAGS) -o $(BIN) $(TFLM_VEC_DIR)/start_semi.S int8dot_ceiling.o \
	  -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

run: $(BIN)
	gem5.opt -d m5out $(SIM_CONFIG) ./$(BIN)

# --- fc_bottleneck.c: 5 variants (0=FULL,1=NO_BIAS,2=NO_REQUANT,3=NO_CLAMP,4=DOT_ONLY),
# each its own fully-specialized binary (compile-time flag, not a runtime
# branch -- see fc_bottleneck.c's header comment for why) ---
FC_VARIANTS := 0 1 2 3 4
FC_BINS := $(addprefix fc_bottleneck_riscv_v,$(FC_VARIANTS))

fc_bottleneck_riscv_v%: fc_bottleneck.c $(TFLM_VEC_DIR)/start_semi.S $(TFLM_VEC_DIR)/linker_semi.ld
	$(CC) $(CFLAGS) -DFC_VARIANT=$* -c fc_bottleneck.c -o fc_bottleneck_v$*.o
	$(CC) $(LDFLAGS) -o $@ $(TFLM_VEC_DIR)/start_semi.S fc_bottleneck_v$*.o \
	  -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

run-fc-bottleneck: $(FC_BINS)
	@for v in $(FC_VARIANTS); do \
	  echo "=== FC_VARIANT=$$v ==="; \
	  gem5.opt -d m5out_fc_v$$v $(SIM_CONFIG) ./fc_bottleneck_riscv_v$$v; \
	done

clean:
	rm -f *.o $(BIN) $(FC_BINS)
	rm -rf m5out m5out_fc_v*

.PHONY: run run-fc-bottleneck clean
