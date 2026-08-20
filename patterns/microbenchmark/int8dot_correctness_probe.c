#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

/* Standalone correctness cross-check for Int8DotProductRvv (copied
 * verbatim from ../../tflite-micro/tensorflow/lite/kernels/internal/
 * reference/integer_ops/fully_connected.h, same convention as
 * fc_bottleneck.c -- see that file's header comment for why a copy
 * instead of an #include) against a plain int64_t scalar reference, at
 * every (K, input_offset) combination anomaly_detection_int8.tflite's 10
 * FULLY_CONNECTED layers actually use. Root-causing the reproducible
 * output CRC32 mismatch found running that model through
 * run_tflm_benchmark (see ../../doc/anomaly_detection/
 * performance_anomaly_detection.md's "Correctness" section) -- this
 * isolates whether Int8DotProductRvv itself is wrong at these shapes,
 * independent of the rest of FullyConnected()'s bias/requant/clamp
 * pipeline.
 *
 * Build + run: `source ../../script/0_env_var_setup.sh && make -f
 * ../Makefile.probe run` (or by hand -- see the by-hand command in
 * int8dot_ceiling.c's own header comment, same pattern).
 */

/* Verbatim copy -- keep in sync with fully_connected.h if it changes. */
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

/* Plain, unoptimized int64_t scalar reference -- no widening tricks, no
 * vector ops, just the literal definition, to catch anything Int8Dot
 * ProductRvv's algebraic expansion might get wrong. */
static int32_t ScalarDotProductRef(const int8_t* input, const int8_t* filter,
                                   int accum_depth, int32_t input_offset,
                                   int32_t filter_offset) {
  int64_t acc = 0;
  for (int d = 0; d < accum_depth; ++d) {
    acc += (int64_t)(filter[d] + filter_offset) * (int64_t)(input[d] + input_offset);
  }
  return (int32_t)acc;
}

static uint32_t rng_state = 12345u;
static int8_t rand_int8(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return (int8_t)((rng_state >> 16) & 0xFFu);
}

#define MAX_K 640
static int8_t g_input[MAX_K];
static int8_t g_filter[MAX_K];

/* Every (K, input_offset) combination anomaly_detection_int8.tflite's 10
 * FULLY_CONNECTED layers actually use (filter_offset is 0 for all 10 --
 * symmetric weight quantization throughout), plus dtln's K=128,
 * input_offset=4 as a known-good control. */
typedef struct { int k; int32_t input_offset; const char* label; } TestCase;
static const TestCase kCases[] = {
    {640, -89, "call1 K=640 (real offset)"},
    {128, 128, "call2-4,7-9 K=128 (real offset)"},
    {128, 128, "call5 K=128 (real offset, N=8 out)"},
    {8,   128, "call6 K=8 (real offset)"},
    {128, 4,   "dtln control K=128 (known-good offset)"},
    {640, 128, "K=640 with the +/-128 edge-case offset"},
    {8,   -89, "K=8 with call1's offset (cross-check)"},
    {1,   128, "K=1 (degenerate, smaller than a single dot)"},
    {639, 128, "K=639 (K=640 minus 1, odd remainder vs. VLMAX=128)"},
    {641, 128, "K=641 (K=640 plus 1, odd remainder vs. VLMAX=128)"},
};
#define N_CASES (sizeof(kCases) / sizeof(kCases[0]))

int main(void) {
  int n_pass = 0, n_fail = 0;
  for (size_t c = 0; c < N_CASES; ++c) {
    int k = kCases[c].k;
    int32_t input_offset = kCases[c].input_offset;
    for (int d = 0; d < k; ++d) {
      g_input[d] = rand_int8();
      g_filter[d] = rand_int8();
    }
    int32_t got = Int8DotProductRvv(g_input, g_filter, k, input_offset, 0);
    int32_t want = ScalarDotProductRef(g_input, g_filter, k, input_offset, 0);
    int ok = (got == want);
    if (ok) n_pass++; else n_fail++;
    printf("case=%d k=%d input_offset=%d got=%d want=%d %s -- %s\n",
           (int)c, k, (int)input_offset, (int)got, (int)want,
           ok ? "PASS" : "FAIL", kCases[c].label);
  }
  printf("SUMMARY: %d/%d passed\n", n_pass, (int)N_CASES);
  return n_fail;
}
