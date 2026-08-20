# Work note: MLPerf Tiny on gem5

## Resource

- Repo: [mlcommons/tiny](https://github.com/mlcommons/tiny) — MLPerf Tiny's
  benchmark suite, reference implementations, and training/conversion
  scripts (`benchmark/training/<benchmark>/`).
- Anomaly detection (`ad01`) specifically:
  [`benchmark/training/anomaly_detection`](https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection),
  golden model at
  [`trained_models/ad01_int8.tflite`](https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite)
  (MD5 `361fa1b1b871e2068b2ab38d9805ef56`, verified against
  `trained_models/ad01_int8.tflite.md5` at fetch time).
- Reference implementation is TensorFlow Lite for Microcontrollers (TFLM)
  + Mbed, though MLPerf Tiny's own rules don't hard-require TFLM — other
  frameworks (CMSIS-NN, microTVM, etc.) are allowed for a real submission.
  This project uses TFLM because the entire existing gem5 harness
  (`run_tflm_benchmark`, `MicroProfiler`-based per-op cycle counting,
  `4_roofline_report.py`) is already built on top of it — see the
  `dtln`-vs-CARRV'19 comparison discussion earlier in this project's
  history for the fuller methodology writeup.

## Implementation: `ad01` running on gem5

Added as `2_pattern/tflm/patterns/anomaly_detection/` (real files) with a
symlink from `tflite-micro/tensorflow/lite/micro/examples/anomaly_detection`
back to it, so TFLM's Makefile (which auto-discovers examples via a
wildcard over `tflite-micro/tensorflow/lite/micro/examples/*/Makefile.inc`)
still finds it, while the source itself is owned by the `tflm_riscv` repo
rather than the `tflite-micro` fork.

- **Model**: `ad01_int8.tflite`, a `640`-in/`640`-out fully int8-quantized
  dense autoencoder — `640 -> 128x4 -> 8 (bottleneck) -> 128x4 -> 640`, 10
  back-to-back `FULLY_CONNECTED` layers, no conv/LSTM (confirmed by
  walking the flatbuffer's operator list directly).
- **Harness**: `anomaly_detection_test.cc`, same shape as `../dtln`'s
  test — `MicroMutableOpResolver<1>` (`AddFullyConnected()` only), 8 KiB
  tensor arena, `Invoke()` + shape/type assertions.
- **Input**: `feature_data` in `anomaly_detection_inout_data.cc` is a
  fixed-seed synthetic int8 stream (same LCG generator as
  `microbenchmark/fc_bottleneck.c`'s `rand_int8()`), **not** a real
  DCASE2020/ToyADMOS log-mel-spectrogram sample. This is a build/link/
  `Invoke()` smoke test, not an accuracy evaluation — no golden-output
  check exists yet (see TODO below).
- **Build/run** (from `tflite-micro/`, after sourcing
  `../script/0_env_var_setup.sh`):
  ```
  make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=riscv64_baremetal[_vector] TARGET_TOOLCHAIN_ROOT=$TOOLCHAIN/bin/ \
    TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- BUILD_TYPE=default \
    test_anomaly_detection_test
  ```
- **Results** (gem5 `MinorCPU`, both `~~~ALL TESTS PASSED~~~`):
  - `riscv64_baremetal` (scalar): 3,252,758 cycles
  - `riscv64_baremetal_vector`: 657,290 cycles (~4.95x) — no new
    vectorization work needed, since `FULLY_CONNECTED` already has the
    RVV `Int8DotProductRvv` fast path from the `dtln` work.
  - Verified identical after the symlink move (clean rebuild, both
    targets) — confirms the symlink reorg didn't change build inputs.

## Roofline analysis (2026-08-20)

Attempted, reusing `dtln`'s exact tooling (`4_roofline_report.py` +
`5_gen_roofline_svg.py`, generic already — no anomaly_detection-specific
code needed beyond one real bug fix, see below) — full writeup in
[`performance_anomaly_detection.md`](performance_anomaly_detection.md) in
this same folder. Headline: **this surfaced a real, reproducible,
root-caused correctness bug** — the vectorized `FULLY_CONNECTED` path's
output doesn't match scalar for this model (`Output CRC32:
scalar=0xC6F70B6E vector=0xFA7AD6B9`). Root cause:
`MultiplyByQuantizedMultiplierInlined` (the RVV-only requantize copy in
`fully_connected.h`) hardcodes the single-rounding algorithm, but this
build's `TFLITE_SINGLE_ROUNDING` is unset, so the real scalar path uses
`common.cc`'s double-rounding (gemmlowp) algorithm instead — confirmed
and quantified with a standalone probe at this model's own 10
`(multiplier, shift)` pairs: 455/20,000 sample mismatches, each off by
exactly `±1`, up to 12.1% of values on the worst layer. This is
model-independent (latent in every RVV `FULLY_CONNECTED`/LSTM call in
this project, including `dtln`'s — just not yet observed to flip a final
CRC32 there). An earlier hypothesis in this note (untested `K=8`/`K=640`
shapes) is **ruled out** — `Int8DotProductRvv` itself is correct at every
shape and offset this model uses, confirmed by a second standalone probe.
See the performance doc's "Correctness" section for full detail. Also
fixed a real, model-independent bug in `4_roofline_report.py` along the
way: it read only `OperatorCode.BuiltinCode()`, which is `0` (mistagged
as `ADD`) for
this model's older-converter `.tflite` — `dtln`'s newer-converter model
happened to never hit this.

## TODO

- [ ] **Fix the vectorized `FULLY_CONNECTED` correctness bug** — root
      cause found and confirmed (see above): `MultiplyByQuantizedMultiplierInlined`
      needs to either match `TFLITE_SINGLE_ROUNDING`'s actual selection
      (mirror `common.cc`'s `#if`/`#else` instead of hardcoding
      single-rounding) or just call the shared `MultiplyByQuantizedMultiplier`
      directly and re-measure whether the inlining win (29.34% whole-model,
      per `../gem5_integration.md`) survives losing the force-inline.
      Higher priority than the dataset fetch below, since it's a real,
      currently-reproducible wrong answer affecting every RVV
      `FULLY_CONNECTED`/LSTM call in this project, not just this model.
- [ ] **Move `dtln` out of the `tflite-micro` submodule too**, same
      symlink treatment as `anomaly_detection`, for consistency (both
      example patterns should live in the same place). See "Implementation"
      above for the pattern to follow.
- [ ] **Fetch the real ToyCar dataset for a golden-output accuracy check**
      — currently untaken, scoped separately since it's a materially
      bigger lift than the build-only smoke test above:
      - `dev_data_ToyCar.zip` (1.7 GiB) +
        `eval_data_train_ToyCar.zip` (0.82 GiB) from Zenodo
        ([mlcommons/tiny's `get_dataset.sh`](https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/get_dataset.sh)),
        ~2.55 GB compressed total.
      - Feature-extraction pipeline pins old deps
        (`librosa==0.6.0`, `numpy==1.16.0`, `tensorflow==2.3.0`,
        `numba==0.48.0` — 2019-era) that will likely need a dedicated venv
        and may not install cleanly against a current toolchain.
      - Now secondary to the correctness bug above: even a real,
        DCASE-derived input would still run through the currently-broken
        vectorized path.
- [ ] **Build a `fc_bottleneck.c`-style ceiling probe at this model's own
      shapes** (`K=8` and `K=640` particularly) — the roofline currently
      reuses `dtln`'s `K=128,N=257`-calibrated ceilings as an
      approximation. See "Ceiling shape mismatch" in
      `performance_anomaly_detection.md`.
