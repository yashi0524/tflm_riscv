#include <stdio.h>
#include <stdint.h>

/* Root-cause probe: is the RVV-only MultiplyByQuantizedMultiplierInlined
 * (fully_connected.h) actually the same rounding algorithm as the real
 * scalar-path MultiplyByQuantizedMultiplier (common.cc)? common.cc has
 * TWO algorithms gated by TFLITE_SINGLE_ROUNDING -- single-rounding
 * (shift-based, what MultiplyByQuantizedMultiplierInlined copies) vs.
 * double-rounding (gemmlowp SaturatingRoundingDoublingHighMul +
 * RoundingDivideByPOT, the #else/default branch when
 * TFLITE_SINGLE_ROUNDING is unset). This project's build never defines
 * TFLITE_SINGLE_ROUNDING (grep confirmed), so the scalar target uses
 * double-rounding -- but the RVV-only inlined copy hardcodes
 * single-rounding unconditionally. This probe reimplements both exactly
 * (double-rounding copied from gemmlowp's actual SaturatingRoundingDoublingHighMul/
 * RoundingDivideByPOT, int32_t semantics) and sweeps real (acc,
 * output_multiplier, output_shift) combinations from anomaly_detection_
 * int8.tflite's own layers to quantify how often/by how much they
 * diverge.
 */

/* Single-rounding: verbatim copy of MultiplyByQuantizedMultiplierInlined
 * (fully_connected.h) / TFLITE_SINGLE_ROUNDING's int32_t overload
 * (common.cc). */
static int32_t MultiplyByQuantizedMultiplier_SingleRounding(
    int32_t x, int32_t quantized_multiplier, int shift) {
  const int64_t total_shift = 31 - shift;
  const int64_t round = (int64_t)1 << (total_shift - 1);
  int64_t result = x * (int64_t)quantized_multiplier + round;
  result = result >> total_shift;
  return (int32_t)result;
}

/* Double-rounding: gemmlowp::SaturatingRoundingDoublingHighMul +
 * RoundingDivideByPOT, reimplemented from gemmlowp/fixedpoint/fixedpoint.h's
 * actual int32_t scalar definitions (not the SIMD specializations). This
 * is what common.cc's #else (default, TFLITE_SINGLE_ROUNDING unset)
 * branch actually calls. */
static int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  int overflow = (a == b) && (a == INT32_MIN);
  int64_t a_64 = (int64_t)a;
  int64_t b_64 = (int64_t)b;
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = (ab_64 >= 0) ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (1LL << 31));
  return overflow ? INT32_MAX : ab_x2_high32;
}

static int32_t RoundingDivideByPOT(int32_t x, int exponent) {
  if (exponent == 0) return x;
  int32_t mask = (1 << exponent) - 1;
  int32_t zero = 0;
  int32_t one = 1;
  int32_t remainder = x & mask;
  int32_t threshold = (mask >> 1) + ((x < zero) ? one : zero);
  return (x >> exponent) + ((remainder > threshold) ? one : zero);
}

static int32_t MultiplyByQuantizedMultiplier_DoubleRounding(
    int32_t x, int32_t quantized_multiplier, int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),
      right_shift);
}

/* (output_multiplier, output_shift) for each of anomaly_detection_int8.
 * tflite's 10 FC layers -- computed from the flatbuffer's effective_scale
 * = input_scale*filter_scale/output_scale via TFLite's standard
 * QuantizeMultiplier (frexp-based fixed-point decomposition), same
 * technique fc_bottleneck.c's header comment documents for dtln. */
typedef struct { int32_t multiplier; int shift; const char* label; } QuantParam;
static const QuantParam kLayers[] = {
    {1638001719, -8, "call1 K=640,N=128"},
    {1442659867, -5, "call2 K=128,N=128"},
    {1185020333, -2, "call3 K=128,N=128"},
    {1439819856, -4, "call4 K=128,N=128"},
    {1085889731, -6, "call5 K=128,N=8"},
    {1442237646, -5, "call6 K=8,N=128"},
    {1315670656, -5, "call7 K=128,N=128"},
    {1994356874, -6, "call8 K=128,N=128"},
    {1105921578, -6, "call9 K=128,N=128"},
    {1462485049, -9, "call10 K=128,N=640"},
};
#define N_LAYERS (sizeof(kLayers) / sizeof(kLayers[0]))

static uint32_t rng_state = 777u;
static int32_t rand_acc(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  int32_t v = (int32_t)rng_state;
  return v / 4;  /* keep well within the x*(1<<left_shift) safe range */
}

int main(void) {
  int n_pass = 0, n_fail = 0, n_tested = 0;
  int32_t max_abs_diff = 0;
  for (size_t l = 0; l < N_LAYERS; ++l) {
    int mismatches_this_layer = 0;
    for (int t = 0; t < 2000; ++t) {
      int32_t acc = rand_acc();
      int32_t single = MultiplyByQuantizedMultiplier_SingleRounding(
          acc, kLayers[l].multiplier, kLayers[l].shift);
      int32_t doubler = MultiplyByQuantizedMultiplier_DoubleRounding(
          acc, kLayers[l].multiplier, kLayers[l].shift);
      n_tested++;
      int32_t diff = single - doubler;
      if (diff < 0) diff = -diff;
      if (diff > max_abs_diff) max_abs_diff = diff;
      if (single != doubler) {
        n_fail++;
        mismatches_this_layer++;
      } else {
        n_pass++;
      }
    }
    printf("layer=%s multiplier=%d shift=%d mismatches=%d/2000\n",
           kLayers[l].label, (int)kLayers[l].multiplier, kLayers[l].shift,
           mismatches_this_layer);
  }
  printf("SUMMARY: %d/%d single-vs-double-rounding MATCHED, %d MISMATCHED, max_abs_diff=%d\n",
         n_pass, n_tested, n_fail, (int)max_abs_diff);
  return n_fail > 0 ? 1 : 0;
}
