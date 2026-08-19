# Makefile — int8dot_ceiling / fc_bottleneck microbenchmarks for MinorCPU's
# FloatSimd FU. Builds against riscv64_baremetal_vector's own crt0/linker
# script directly -- no TFLM library build needed, matches the by-hand
# commands documented in int8dot_ceiling.c's header comment (kept here
# just so they don't need retyping).
#
# Build artifacts and gem5 output logs land in two independent
# subdirectories (one per source benchmark), keeping the two benchmark
# families' .o/binaries/m5out* from being interleaved in this directory:
#   int8dot_ceiling/   -- int8dot_ceiling.c's .o, binary, m5out
#   fc_bottleneck/     -- fc_bottleneck.c's 10 variant .o/binaries, their
#                         m5out_fc_v*[_cold], and m5out_cold_ceiling
#
# Usage:
#   source ../script/0_env_var_setup.sh
#   make run                                              # GCC 13.4.0-1 (default, authoritative -- see doc/gem5_integration.md)
#   make run TOOLCHAIN_ROOT=../script/clang_wrapper/       # clang-18 cross-check
#   make run-fc-bottleneck                                # builds+runs all 5 fc_bottleneck.c variants, warm cache (FC_CACHE_MODE=0)
#   make run-fc-bottleneck-cold                            # same 5 variants, cold cache (FC_CACHE_MODE=1) -- see fc_bottleneck.c's header comment
#   make run-cold-ceiling                                  # single-binary version of the above's FC_VARIANT=4 cold run --
#                                                           # what script/4_roofline_report.py's --measure-cold-ceiling invokes
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

INT8DOT_DIR := int8dot_ceiling
FC_DIR      := fc_bottleneck

BIN := $(INT8DOT_DIR)/int8dot_ceiling_riscv

$(INT8DOT_DIR) $(FC_DIR):
	mkdir -p $@

$(BIN): int8dot_ceiling.c $(TFLM_VEC_DIR)/start_semi.S $(TFLM_VEC_DIR)/linker_semi.ld | $(INT8DOT_DIR)
	$(CC) $(CFLAGS) -c int8dot_ceiling.c -o $(INT8DOT_DIR)/int8dot_ceiling.o
	$(CC) $(LDFLAGS) -o $(BIN) $(TFLM_VEC_DIR)/start_semi.S $(INT8DOT_DIR)/int8dot_ceiling.o \
	  -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

run: $(BIN)
	gem5.opt -d $(INT8DOT_DIR)/m5out $(SIM_CONFIG) ./$(BIN)

# --- fc_bottleneck.c: 5 variants (0=FULL,1=NO_BIAS,2=NO_REQUANT,3=NO_CLAMP,4=DOT_ONLY),
# each its own fully-specialized binary (compile-time flag, not a runtime
# branch -- see fc_bottleneck.c's header comment for why) ---
FC_VARIANTS := 0 1 2 3 4
FC_BINS := $(addprefix $(FC_DIR)/fc_bottleneck_riscv_v,$(FC_VARIANTS))

$(FC_DIR)/fc_bottleneck_riscv_v%: fc_bottleneck.c $(TFLM_VEC_DIR)/start_semi.S $(TFLM_VEC_DIR)/linker_semi.ld | $(FC_DIR)
	$(CC) $(CFLAGS) -DFC_VARIANT=$* -c fc_bottleneck.c -o $(FC_DIR)/fc_bottleneck_v$*.o
	$(CC) $(LDFLAGS) -o $@ $(TFLM_VEC_DIR)/start_semi.S $(FC_DIR)/fc_bottleneck_v$*.o \
	  -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

run-fc-bottleneck: $(FC_BINS)
	@for v in $(FC_VARIANTS); do \
	  echo "=== FC_VARIANT=$$v (warm cache) ==="; \
	  gem5.opt -d $(FC_DIR)/m5out_fc_v$$v $(SIM_CONFIG) ./$(FC_DIR)/fc_bottleneck_riscv_v$$v; \
	done

# --- same 5 variants, built with FC_CACHE_MODE=1 (cold cache: evict then
# a single timed pass, reproducing the real kernel's actual memory
# environment -- see fc_bottleneck.c's header comment) ---
FC_COLD_BINS := $(addsuffix _cold,$(FC_BINS))

$(FC_DIR)/fc_bottleneck_riscv_v%_cold: fc_bottleneck.c $(TFLM_VEC_DIR)/start_semi.S $(TFLM_VEC_DIR)/linker_semi.ld | $(FC_DIR)
	$(CC) $(CFLAGS) -DFC_VARIANT=$* -DFC_CACHE_MODE=1 -c fc_bottleneck.c -o $(FC_DIR)/fc_bottleneck_v$*_cold.o
	$(CC) $(LDFLAGS) -o $@ $(TFLM_VEC_DIR)/start_semi.S $(FC_DIR)/fc_bottleneck_v$*_cold.o \
	  -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

run-fc-bottleneck-cold: $(FC_COLD_BINS)
	@for v in $(FC_VARIANTS); do \
	  echo "=== FC_VARIANT=$$v (cold cache) ==="; \
	  gem5.opt -d $(FC_DIR)/m5out_fc_v$${v}_cold $(SIM_CONFIG) ./$(FC_DIR)/fc_bottleneck_riscv_v$${v}_cold; \
	done

# --- single-binary cold-ceiling run (FC_VARIANT=4/DOT_ONLY, cold cache) --
# the cold-cache analog of `run` above (int8dot_ceiling.c's warm ceiling),
# used as the "empirical cold-cache ceiling" in the roofline report/SVG
# (script/4_roofline_report.py --measure-cold-ceiling) ---
run-cold-ceiling: $(FC_DIR)/fc_bottleneck_riscv_v4_cold
	gem5.opt -d $(FC_DIR)/m5out_cold_ceiling $(SIM_CONFIG) ./$(FC_DIR)/fc_bottleneck_riscv_v4_cold

# --- single-binary fc_bottleneck warm-ceiling run (FC_VARIANT=4/DOT_ONLY,
# warm cache, FC_ITERS=1 same as the cold run above -- the third ceiling
# in the roofline report/SVG, script/4_roofline_report.py
# --measure-fc-warm-ceiling) ---
run-fc-warm-ceiling: $(FC_DIR)/fc_bottleneck_riscv_v4
	gem5.opt -d $(FC_DIR)/m5out_fc_warm_ceiling $(SIM_CONFIG) ./$(FC_DIR)/fc_bottleneck_riscv_v4

clean:
	rm -rf $(INT8DOT_DIR) $(FC_DIR)

.PHONY: run run-fc-bottleneck run-fc-bottleneck-cold run-cold-ceiling run-fc-warm-ceiling clean
