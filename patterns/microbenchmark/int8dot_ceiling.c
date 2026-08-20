#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

/* Compute-roof ceiling probe for `Int8DotProductRvv`'s actual instruction
 * mix (tensorflow/lite/kernels/internal/reference/integer_ops/
 * fully_connected.h), against MinorCPU's FloatSimd FU (opLat=6,
 * issueLat=1, 1 instance -- see gem5 src/cpu/minor/BaseMinorCPU.py).
 *
 * Answers: "how fast could this simulated CPU possibly do this op mix,
 * with independent-chain unrolling to hide the FU's 6-cycle latency" --
 * mirroring the sibling gemm project's fmacc.c methodology (measure the
 * ceiling, don't assume it), adapted to this kernel's specific
 * multi-instruction (widen/widen/multiply/reduce) op sequence instead of
 * a single fused FMA. See "Why efficiency stays low even after
 * vectorization" and the ceiling-probe writeup in
 * ../doc/gem5_integration.md for the full derivation and how this result
 * changes the roofline's efficiency numbers.
 *
 * First attempt used 8 independent chains (mirroring gemm's fmacc.c
 * 8-way unroll) -- but at this kernel's LMUL=2 base (e8m2 -> e16m4 ->
 * e32m8), 8 simultaneously-live chains need up to 8x8=64 vector
 * registers at the multiply stage (RVV only has 32 total), forcing heavy
 * spill/reload traffic (confirmed via disassembly: 20 whole-register
 * spill/reload instructions) that swamped the actual FU latency being
 * measured. Register pressure, not FU latency, was the bottleneck in
 * that version -- a real finding, but not what this probe is trying to
 * isolate. NCHAINS=3 below keeps peak simultaneous vector register use
 * at 3x8=24 (multiply stage) / 3x2x4=24 (widen stage), comfortably under
 * 32 with no spilling.
 *
 * Measured results (both correct, sink=-361804616 for both):
 *   GCC 13.4.0-1:  206 cycles/iter, 3.723 GFLOP/s, 4 residual spills
 *   clang-18:      278 cycles/iter, 2.760 GFLOP/s, 3 residual spills
 *
 * Build + run: `source ../script/0_env_var_setup.sh && make run` (see
 * ./Makefile -- GCC 13.4.0-1 by default; `make run
 * TOOLCHAIN_ROOT=../script/clang_wrapper/` for the clang-18 cross-check).
 *
 * The Makefile just automates the by-hand commands below (bare-metal
 * RISC-V, links against the riscv64_baremetal_vector target's own
 * crt0/linker script -- no TFLM build needed), kept here for reference:
 *
 *   source ../script/0_env_var_setup.sh
 *   TFLM_VEC_DIR=../tflite-micro/tensorflow/lite/micro/riscv64_baremetal_vector
 *
 *   # GCC 13.4.0-1:
 *   $TOOLCHAIN/bin/riscv-none-elf-gcc \
 *     -march=rv64imc_zicsr_zve64x -mabi=lp64 -mcmodel=medany -mexplicit-relocs \
 *     -fno-builtin-printf -funsigned-char -fno-delete-null-pointer-checks \
 *     -fomit-frame-pointer -O2 -c int8dot_ceiling.c -o int8dot_ceiling.o
 *   $TOOLCHAIN/bin/riscv-none-elf-gcc \
 *     -march=rv64imc_zicsr_zve64x -mabi=lp64 -mcmodel=medany -mexplicit-relocs \
 *     -mno-relax -nostartfiles -nostdlib -T $TFLM_VEC_DIR/linker_semi.ld \
 *     -o int8dot_ceiling_riscv $TFLM_VEC_DIR/start_semi.S int8dot_ceiling.o \
 *     -Wl,--start-group -lc -lm -lgcc -Wl,--end-group
 *
 *   # clang-18 (via ../script/clang_wrapper/, see its own header comment):
 *   ../script/clang_wrapper/riscv-none-elf-gcc \
 *     -march=rv64imc_zicsr_zve64x -mabi=lp64 -mcmodel=medany -O2 \
 *     -c int8dot_ceiling.c -o int8dot_ceiling_clang.o
 *   ../script/clang_wrapper/riscv-none-elf-gcc \
 *     -march=rv64imc_zicsr_zve64x -mabi=lp64 -mcmodel=medany -nostartfiles \
 *     -nostdlib -Wl,--no-relax -T $TFLM_VEC_DIR/linker_semi.ld \
 *     -o int8dot_ceiling_clang_riscv $TFLM_VEC_DIR/start_semi.S \
 *     int8dot_ceiling_clang.o -lc -lm -lgcc
 *
 *   # Either binary, under gem5 (cycle-accurate; whisper has no timing
 *   # model, not useful for this measurement):
 *   gem5.opt -d /tmp/m5out ../sim_config/gem5_riscv_baremetal_fs.py ./int8dot_ceiling_riscv
 */

#define NCHAINS 3
#define ITERS 2000

static uint32_t rng_state = 12345u;
static int8_t rand_int8(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return (int8_t)((rng_state >> 16) & 0xFFu);
}

static inline uint64_t ReadMcycle(void) {
  uint64_t v;
  asm volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

int main(void) {
  static int8_t bufA[NCHAINS][128];
  static int8_t bufB[NCHAINS][128];
  for (int c = 0; c < NCHAINS; ++c) {
    for (int i = 0; i < 128; ++i) {
      bufA[c][i] = rand_int8();
      bufB[c][i] = rand_int8();
    }
  }

  size_t vl = __riscv_vsetvl_e8m2(128);
  int32_t acc0 = 0, acc1 = 0, acc2 = 0;

  uint64_t t0 = ReadMcycle();
  for (int it = 0; it < ITERS; ++it) {
    /* Perturb one byte per chain with an optimizer-opaque value so the
     * compiler can't recognize repeated iterations as identical and
     * hoist/CSE the whole computation out of the loop (confirmed this
     * happens without it: >800 GFLOP/s, a physically impossible result
     * given only 1 FU with issueLat=1). */
    int8_t pert;
    asm volatile("addi %0, %1, 0" : "=r"(pert) : "r"(it));
    for (int c = 0; c < NCHAINS; ++c) {
      bufA[c][0] = pert;
      bufB[c][0] = pert;
    }

    /* Stage 1: all loads */
    vint8m2_t a0 = __riscv_vle8_v_i8m2(bufA[0], vl);
    vint8m2_t a1 = __riscv_vle8_v_i8m2(bufA[1], vl);
    vint8m2_t a2 = __riscv_vle8_v_i8m2(bufA[2], vl);
    vint8m2_t b0 = __riscv_vle8_v_i8m2(bufB[0], vl);
    vint8m2_t b1 = __riscv_vle8_v_i8m2(bufB[1], vl);
    vint8m2_t b2 = __riscv_vle8_v_i8m2(bufB[2], vl);

    /* Stage 2: all widens */
    vint16m4_t wa0 = __riscv_vwadd_vx_i16m4(a0, 0, vl);
    vint16m4_t wa1 = __riscv_vwadd_vx_i16m4(a1, 0, vl);
    vint16m4_t wa2 = __riscv_vwadd_vx_i16m4(a2, 0, vl);
    vint16m4_t wb0 = __riscv_vwadd_vx_i16m4(b0, 0, vl);
    vint16m4_t wb1 = __riscv_vwadd_vx_i16m4(b1, 0, vl);
    vint16m4_t wb2 = __riscv_vwadd_vx_i16m4(b2, 0, vl);

    /* Stage 3: all multiplies */
    vint32m8_t p0 = __riscv_vwmul_vv_i32m8(wa0, wb0, vl);
    vint32m8_t p1 = __riscv_vwmul_vv_i32m8(wa1, wb1, vl);
    vint32m8_t p2 = __riscv_vwmul_vv_i32m8(wa2, wb2, vl);

    /* Stage 4: all reduces */
    vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);
    acc0 += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(p0, zero32, vl));
    acc1 += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(p1, zero32, vl));
    acc2 += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(p2, zero32, vl));
  }
  uint64_t t1 = ReadMcycle();

  uint64_t total_cycles = t1 - t0;
  uint64_t cycles_per_iter = total_cycles / ITERS;
  long total_flops = (long)NCHAINS * 128L * 2L * ITERS;
  long gflops_milli = (total_flops * 1000L) / (long)total_cycles;

  printf("total_cycles=%llu\n", (unsigned long long)total_cycles);
  printf("cycles_per_iter=%llu\n", (unsigned long long)cycles_per_iter);
  printf("cycles_per_chain_op=%llu (cycles_per_iter/%d)\n",
         (unsigned long long)(cycles_per_iter / NCHAINS), NCHAINS);
  printf("GFLOP/s=%ld.%03ld\n", gflops_milli / 1000, gflops_milli % 1000);
  printf("sink=%d\n", acc0 + acc1 + acc2);
  return 0;
}
