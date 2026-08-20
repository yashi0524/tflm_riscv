#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

/* Realistic FULLY_CONNECTED benchmark, at dtln_noise_suppression.tflite's
 * actual FC shape (M=1, K=128, N=257) and actual quantization params
 * (pulled straight from the .tflite flatbuffer -- see below), instead of
 * int8dot_ceiling.c's isolated dot-product-only chain. Where the ceiling
 * probe answers "what's the best this hardware could do for just the
 * vector chain," this answers "where do the real kernel's cycles
 * actually go" -- by building the whole per-output-channel pipeline
 * (dot product -> bias add -> requantize -> output-offset -> clamp ->
 * store, exactly mirroring FullyConnected()'s out_c loop in
 * tensorflow/lite/kernels/internal/reference/integer_ops/
 * fully_connected.h) and then removing one stage at a time via
 * FC_VARIANT, to measure each stage's actual cycle contribution instead
 * of just asserting "the real kernel does more scalar work" the way
 * ../doc/gem5_integration.md's Amdahl's-Law paragraphs have so far.
 *
 * Int8DotProductRvv and MultiplyByQuantizedMultiplier below are direct
 * copies of the real kernel's versions (fully_connected.h and
 * tensorflow/lite/kernels/internal/common.cc respectively) -- copied
 * rather than #included, since common.h pulls in fixedpoint.h/
 * runtime_shape.h/types.h and the rest of TFLM's normal include graph,
 * which this bare-metal probe deliberately avoids linking against (same
 * reasoning as int8dot_ceiling.c: build directly against
 * riscv64_baremetal_vector's own crt0/linker script, no TFLM library
 * build needed). Keep both in sync with their real-kernel originals if
 * either changes.
 *
 * Quantization params: real values from dtln_noise_suppression.tflite's
 * FULLY_CONNECTED op (extracted via a one-off flatbuffer read, not
 * guessed) -- input zero_point=-4 (input_offset=+4), filter zero_point=0
 * (filter_offset=0, confirmed symmetric weight quantization -- see
 * ../doc/gem5_integration.md's "Tried: interleaving..." section), output
 * zero_point=-2 (output_offset=-2). output_multiplier/output_shift
 * (1820954201, -7) computed from the real effective_scale
 * (input_scale * filter_scale / output_scale = 0.00736330496147275 *
 * 0.0348852202296257 / 0.03877529129385948) via TFLite's standard
 * QuantizeMultiplier algorithm (frexp-based fixed-point decomposition) --
 * not read directly from the flatbuffer, since TFLite computes this at
 * Prepare() time rather than storing it. activation_min/max = -128/127
 * (int8 full range -- this op's FusedActivationFunction is NONE).
 *
 * FC_VARIANT (set via -DFC_VARIANT=N at compile time -- a compile-time
 * flag, not a runtime branch: a runtime branch guarding one of these
 * stages already measured *worse* than doing the work unconditionally,
 * see "Tried" in gem5_integration.md -- so a runtime toggle here would
 * risk re-introducing that same scheduling artifact into what's supposed
 * to be a clean per-stage attribution):
 *   0 = FULL       -- every stage, matches the real kernel exactly
 *   1 = NO_BIAS     -- skip the bias add
 *   2 = NO_REQUANT  -- skip MultiplyByQuantizedMultiplier
 *   3 = NO_CLAMP    -- skip the activation_min/max clamp
 *   4 = DOT_ONLY    -- only Int8DotProductRvv, no scalar post-processing
 *                      at all (closest analog to int8dot_ceiling.c, but
 *                      at this op's real N=257/K=128 shape instead of an
 *                      interleaved synthetic chain)
 *
 * Build + run (all 5 variants): `source ../script/0_env_var_setup.sh &&
 * make run-fc-bottleneck` (see ./Makefile). Results and analysis: see
 * "Realistic FULLY_CONNECTED bottleneck decomposition" in
 * ../doc/gem5_integration.md.
 *
 * ============================================================================
 * KNOWN FIDELITY GAP -- read before trusting this file's absolute numbers.
 * ============================================================================
 * This benchmark's FC_VARIANT=0 (full pipeline) measures ~25,000
 * cycles/pass. The real kernel measures 84,311 cycles for the identical
 * shape and op mix (confirmed via direct instrumentation of the real
 * FullyConnected() -- see the doc section above). That gap was originally
 * much worse (~19,500 cycles, 4.3x off) because the first version of this
 * file passed `filter_offset` in as a plain `const int32_t filter_offset =
 * 0` local: a compile-time-visible literal zero let GCC prove
 * `filter_offset * input_sum` in Int8DotProductRvv is always exactly 0 and
 * delete the entire input_sum reduction at compile time -- silently
 * measuring a *cheaper, different* computation than the real kernel, which
 * sees filter_offset as a genuine runtime value (params.weights_offset,
 * only known after the model loads) it can't fold away. That's why every
 * quantization param below is declared `volatile`, not `const` -- forces a
 * real runtime read GCC can't constant-propagate, closing most (not all)
 * of the gap and restoring the correct 3-reduction instruction sequence
 * (verified via disassembly).
 *
 * A real gap remained even so (~25,000 vs. 84,311 for FC_VARIANT=0), and
 * an earlier version of this comment attributed it to register-pressure/
 * instruction-scheduling differences from the real FullyConnected()'s
 * larger surrounding scope. That theory was investigated further and
 * disproven (2026-08-19): a fresh disassembly of a clean rebuild showed
 * the real kernel's per-channel loop has clean register allocation, no
 * spilling, for the scalar quantization constants -- see "Realistic
 * FULLY_CONNECTED bottleneck decomposition" in gem5_integration.md.
 *
 * The actual cause, confirmed the same day: this file's own cache-warmth
 * artifact, not anything about the real kernel. The init loops above write
 * `filter`/`input`/`bias` immediately before the timed region, and
 * ITERS=20 reuses that same ~33 KB array every pass -- so after the first
 * touch it's essentially permanently resident in the 64 KB L1. The real
 * kernel's FullyConnected() never gets that: its filter tensor was last
 * touched at model-load time, and the LSTM ops that run immediately
 * before it in the same dtln inference have a much larger combined
 * working set that evicts it from L1 first -- a genuinely cold cache on
 * every invocation. Reproducing that cold state here (touching a 256 KiB
 * `volatile` distractor buffer right before timing, in a scratch copy)
 * brought FC_VARIANT=4's 74 cycles/channel up to 200 -- within 2% of the
 * real kernel's directly-instrumented 204. Not chased down further into a
 * permanent fix in this file, since direct instrumentation of the real
 * kernel already gives reliable ground truth (the table in
 * gem5_integration.md). Formalized as a permanent build-time mode instead
 * of a one-off scratch copy -- see FC_CACHE_MODE below -- so both the
 * warm-cache and cold-cache ceilings can be reproduced directly from this
 * file instead of hand-patching a throwaway copy each time.
 *
 * What this file IS still good for: fast iteration on relative,
 * within-itself comparisons (e.g. "does skipping the bias add measurably
 * change cycles at all") without needing to touch and revert the real
 * kernel each time -- the non-dot stages (bias/requant/clamp) aren't
 * memory-bound the way the dot product is, so their relative deltas are
 * more trustworthy than the dot product's absolute number even in
 * FC_CACHE_MODE=0 (warm). For absolute, real-kernel-comparable numbers,
 * use FC_CACHE_MODE=1 (cold) -- see below.
 *
 * FC_CACHE_MODE (set via -DFC_CACHE_MODE=N at compile time, default 0):
 *   0 = WARM -- filter/input/bias are written by the init loops
 *       immediately before timing, and timing covers exactly one pass
 *       over them (FC_ITERS=1, same as COLD below) -- so they're
 *       L1-resident from that init touch, with nothing else to evict
 *       them before the timed pass runs. This is a *compute* ceiling:
 *       the best this op's instruction mix can do assuming its operands
 *       are already in cache. It is NOT representative of the real
 *       kernel's actual memory environment -- see the fidelity-gap
 *       discussion above.
 *   1 = COLD -- touches a 256 KiB `volatile` distractor buffer (4x this
 *       target's real 64 KiB L1) immediately after the init loops but
 *       before timing starts, evicting filter/input/bias back out, then
 *       runs the same single timed pass (FC_ITERS=1) as WARM. This
 *       reproduces the real kernel's actual condition: every weight byte
 *       read exactly once, never previously cached (confirmed to land
 *       within 2% of the real kernel's directly-instrumented
 *       cycles/channel -- see "Realistic FULLY_CONNECTED bottleneck
 *       decomposition" in gem5_integration.md). This is the
 *       *memory-latency* ceiling: the best this op can do given its
 *       actual, unavoidable per-invocation cache-miss pattern.
 * FC_ITERS is fixed at 1 for both modes -- originally WARM ran 20
 * reps (reusing the same resident array) while COLD was forced to 1,
 * which meant the two ceilings differed in iteration count as well as
 * cache state. Pinning both to a single timed pass isolates cache state
 * as the only variable between them (the array is already warm from the
 * init-loop touch by the first pass either way, so this doesn't change
 * WARM's measured cycles -- it just removes the confound).
 * Build both: `make run-fc-bottleneck` (warm) and
 * `make run-fc-bottleneck-cold` (cold) -- see ./Makefile.
 */

#define N_CHANNELS 257
#define K_DEPTH 128
#define FC_ITERS 1

#ifndef FC_VARIANT
#define FC_VARIANT 0
#endif

#ifndef FC_CACHE_MODE
#define FC_CACHE_MODE 0
#endif

#if FC_CACHE_MODE == 1
/* 256 KiB: 4x this target's real 64 KiB L1 (sim_config/
 * gem5_riscv_baremetal_fs.py) -- confirmed sufficient to fully evict
 * filter/input/bias (see gem5_integration.md). `volatile` so GCC can't
 * prove the writes are dead and elide the whole loop. */
static volatile int8_t g_evict_buf[262144];
#endif

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

/* Copy of fully_connected.h's Int8DotProductRvv -- see that file's own
 * header comment for the offset-truncation-bug history and why the
 * gemmlowp-style algebraic expansion is used instead of folding offsets
 * into vwadd.vx. Unchanged here: this benchmark is measuring what's
 * *around* this function, not this function itself (see
 * int8dot_ceiling.c for that). */
static inline int32_t Int8DotProductRvv(const int8_t* input,
                                        const int8_t* filter, int accum_depth,
                                        int32_t input_offset,
                                        int32_t filter_offset) {
  int32_t dot = 0;
  int32_t filter_sum = 0;
  int32_t input_sum = 0;
  int n = accum_depth;
  const int8_t* a = input;
  const int8_t* w = filter;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);
    vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
    vint8m2_t vw = __riscv_vle8_v_i8m2(w, vl);
    vint16m4_t va16 = __riscv_vwadd_vx_i16m4(va, 0, vl);
    vint16m4_t vw16 = __riscv_vwadd_vx_i16m4(vw, 0, vl);
    vint32m8_t prod32 = __riscv_vwmul_vv_i32m8(vw16, va16, vl);
    vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);

    vint32m1_t dot_v = __riscv_vredsum_vs_i32m8_i32m1(prod32, zero32, vl);
    dot += __riscv_vmv_x_s_i32m1_i32(dot_v);

    vint32m1_t fsum_v = __riscv_vwredsum_vs_i16m4_i32m1(vw16, zero32, vl);
    filter_sum += __riscv_vmv_x_s_i32m1_i32(fsum_v);

    vint32m1_t isum_v = __riscv_vwredsum_vs_i16m4_i32m1(va16, zero32, vl);
    input_sum += __riscv_vmv_x_s_i32m1_i32(isum_v);

    a += vl;
    w += vl;
    n -= (int)vl;
  }
  return dot + input_offset * filter_sum + filter_offset * input_sum +
         accum_depth * filter_offset * input_offset;
}

/* Copy of the int32_t overload from
 * tensorflow/lite/kernels/internal/common.cc, used by FullyConnected()'s
 * int32-accumulator requantization path. common.cc actually selects
 * between two different algorithms via #if TFLITE_SINGLE_ROUNDING, and
 * this project's build never defines that macro -- so the real kernel
 * (and this copy, to match) uses double-rounding (gemmlowp
 * SaturatingRoundingDoublingHighMul + RoundingDivideByPOT), not the
 * simpler single-rounding shift-based formula. Confirmed
 * 2026-08-20: this file originally hardcoded single-rounding
 * unconditionally, silently out of sync with the real (double-rounding)
 * scalar path -- same bug independently found and fixed in
 * fully_connected.h's MultiplyByQuantizedMultiplierInlined via
 * ../../doc/anomaly_detection/performance_anomaly_detection.md's
 * root-cause investigation. Reimplemented here from gemmlowp/fixedpoint/
 * fixedpoint.h's actual int32_t scalar specializations (can't just
 * #include the real gemmlowp header: this file builds as plain C against
 * no TFLM include path at all -- see this file's own top-of-file
 * comment -- and gemmlowp's fixedpoint.h is a C++ template header) --
 * this is the same reimplementation already cross-validated in
 * requant_correctness_probe.c. */
#if TFLITE_SINGLE_ROUNDING
static inline int32_t MultiplyByQuantizedMultiplier(int32_t x,
                                                     int32_t quantized_multiplier,
                                                     int shift) {
  const int64_t total_shift = 31 - shift;
  const int64_t round = (int64_t)1 << (total_shift - 1);
  int64_t result = x * (int64_t)quantized_multiplier + round;
  result = result >> total_shift;
  return (int32_t)result;
}
#else   /* !TFLITE_SINGLE_ROUNDING (the default, and what this project's
         * build actually uses) */
static inline int32_t SaturatingRoundingDoublingHighMul_(int32_t a, int32_t b) {
  int overflow = (a == b) && (a == INT32_MIN);
  int64_t ab_64 = (int64_t)a * (int64_t)b;
  int32_t nudge = (ab_64 >= 0) ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (1LL << 31));
  return overflow ? INT32_MAX : ab_x2_high32;
}

static inline int32_t RoundingDivideByPOT_(int32_t x, int exponent) {
  if (exponent == 0) return x;
  int32_t mask = (1 << exponent) - 1;
  int32_t remainder = x & mask;
  int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
  return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

static inline int32_t MultiplyByQuantizedMultiplier(int32_t x,
                                                     int32_t quantized_multiplier,
                                                     int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT_(
      SaturatingRoundingDoublingHighMul_(x * (1 << left_shift), quantized_multiplier),
      right_shift);
}
#endif  /* TFLITE_SINGLE_ROUNDING */

int main(void) {
  static int8_t input[K_DEPTH];
  static int8_t filter[N_CHANNELS][K_DEPTH];
  static int32_t bias[N_CHANNELS];
  static int8_t output[N_CHANNELS];

  for (int i = 0; i < K_DEPTH; ++i) input[i] = rand_int8();
  for (int n = 0; n < N_CHANNELS; ++n) {
    for (int i = 0; i < K_DEPTH; ++i) filter[n][i] = rand_int8();
    bias[n] = (int32_t)rand_int8() * 1000;
  }

#if FC_CACHE_MODE == 1
  /* Evict filter/input/bias back out of L1 before timing starts, so the
   * timed pass below sees the same genuinely-cold state the real kernel's
   * single Invoke() call does -- without this, the init loops just above
   * leave this data resident, silently reproducing the warm-cache
   * artifact this mode exists to avoid. */
  for (int i = 0; i < (int)sizeof(g_evict_buf); ++i) g_evict_buf[i] = (int8_t)i;
#endif

  /* volatile, not const: the real kernel reads these from a
   * FullyConnectedParams struct populated at Prepare() time from the
   * model's flatbuffer -- genuine runtime values GCC cannot see the
   * contents of at compile time, even though filter_offset happens to
   * equal 0 for this real model. A plain `const int32_t filter_offset = 0`
   * here would let GCC prove filter_offset*input_sum is always 0 and
   * delete the whole input_sum reduction at compile time -- confirmed via
   * disassembly this actually happened in an earlier version of this
   * benchmark (only 2 of Int8DotProductRvv's 3 reductions survived), which
   * is why FC_VARIANT=4 measured ~4x fewer cycles than instrumenting the
   * real kernel directly showed for the same "257 calls" quantity. `volatile`
   * forces a genuine runtime read at every use, closing that gap. */
  volatile int32_t input_offset = 4;
  volatile int32_t filter_offset = 0;
  volatile int32_t output_offset = -2;
  volatile int32_t output_multiplier = 1820954201;
  volatile int output_shift = -7;
  volatile int32_t activation_min = -128;
  volatile int32_t activation_max = 127;

  int32_t sink = 0;

  uint64_t t0 = ReadMcycle();
  for (int it = 0; it < FC_ITERS; ++it) {
    /* Same anti-hoisting trick as int8dot_ceiling.c: without this, the
     * compiler can prove every outer iteration computes the same 257
     * outputs and hoist the whole loop out. */
    int8_t pert;
    asm volatile("addi %0, %1, 0" : "=r"(pert) : "r"(it));
    input[0] = pert;

    for (int n = 0; n < N_CHANNELS; ++n) {
      int32_t acc = Int8DotProductRvv(input, filter[n], K_DEPTH, input_offset,
                                      filter_offset);

#if FC_VARIANT == 4 /* DOT_ONLY */
      sink += acc;
#else
#if FC_VARIANT != 1 /* not NO_BIAS */
      acc += bias[n];
#endif

#if FC_VARIANT != 2 /* not NO_REQUANT */
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
#else
      int32_t acc_scaled = acc;
#endif
      acc_scaled += output_offset;

#if FC_VARIANT != 3 /* not NO_CLAMP */
      if (acc_scaled < activation_min) acc_scaled = activation_min;
      if (acc_scaled > activation_max) acc_scaled = activation_max;
#endif

      output[n] = (int8_t)acc_scaled;
      sink += output[n];
#endif /* FC_VARIANT == 4 */
    }
  }
  uint64_t t1 = ReadMcycle();

  uint64_t total_cycles = t1 - t0;
  uint64_t cycles_per_iter = total_cycles / FC_ITERS;
  uint64_t cycles_per_channel = cycles_per_iter / N_CHANNELS;

  /* FLOPs = 2 per MAC (matching this project's roofline methodology --
   * see analysis/roofline_log.txt -- counts only the dot product's MACs,
   * not the scalar bias/requant/clamp stages, so this is comparable
   * across all 5 FC_VARIANTs and against the real kernel's own numbers). */
  long total_flops = (long)N_CHANNELS * K_DEPTH * 2L * FC_ITERS;
  long gflops_milli = (total_flops * 1000L) / (long)total_cycles;

  printf("variant=%d\n", FC_VARIANT);
  printf("cache_mode=%s\n", (FC_CACHE_MODE == 1) ? "cold" : "warm");
  printf("total_cycles=%llu\n", (unsigned long long)total_cycles);
  printf("cycles_per_iter=%llu\n", (unsigned long long)cycles_per_iter);
  printf("cycles_per_channel=%llu\n", (unsigned long long)cycles_per_channel);
  printf("GFLOP/s=%ld.%03ld\n", gflops_milli / 1000, gflops_milli % 1000);
  printf("sink=%d\n", sink);
  return 0;
}
