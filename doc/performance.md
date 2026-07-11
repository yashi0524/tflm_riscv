# TFLM RISC-V Performance Log

Consolidated numbers from every verified run across this project's
simulator integrations. For the *why* behind each target/fix, see
[`gem5_integration.md`](gem5_integration.md) — this file is just the
numbers, kept as a single scannable reference.

**Caveat that applies to every row below:** TFLM's per-op software timing
instrumentation isn't wired up on any of these targets — every op always
prints `0 ticks (0 ms)` regardless of target/simulator. The only
trustworthy timing figures are the simulator's own whole-run counters:
gem5's `tick` count (cycle-accurate, `RiscvMinorCPU`) or whisper's
instruction count (functional-only, no timing model — **not**
comparable to gem5 ticks, just useful as a relative "how much work did
this do" signal and for fast iteration).

## `riscv{32,64}_generic` (SE / syscall-emulation mode, gem5)

| RV width | Test/benchmark | Model | Arena | gem5 ticks | Simulated time |
|---|---|---|---|---|---|
| RV32 | `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 1,376 B | not recorded | not recorded |
| RV64 | `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 2,408 B | not recorded | not recorded |
| RV64 | `tflm_benchmark` | `person_detect.tflite` (30 ops) | 89,248 B | 6,179,398,577,000 | 6.179 s |

## `riscv64_baremetal` (FS / bare-metal mode, gem5 — cycle-accurate `RiscvMinorCPU`)

| Test/benchmark | Model | Arena | Binary size | gem5 ticks | Simulated time | Notes |
|---|---|---|---|---|---|---|
| `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 2,408 B | 147,232 B | 672,762,000 | ~0.67 ms | FLASH usage ~87.4 KB (of the original 128 KB budget, since bumped to 4 MB) |
| `micro_utils_test` | n/a (unit test, 8 cases) | — | — | 126,455,000 | ~0.13 ms | Genuine result, post-`.init_array` fix (was silently vacuous before) |
| `dtln_test` | `dtln_noise_suppression.tflite` (LSTM + 1 `FULLY_CONNECTED`, `M=1,K=128,N=257`) | — | — | 5,996,072,000 | ~6.0 ms | Genuine result, post-`.init_array` fix (was silently vacuous before). Slowest/biggest `riscv64_baremetal` run so far. |
| `tflm_benchmark` | `person_detect.tflite` (30 ops) | 89,248 B | — | 429,485,090,000 | ~429 ms | Post-`.init_array`-fix regression check; matches pre-fix 422,108,725,000 (~422 ms) within normal run-to-run variance |

Note the FS-mode `tflm_benchmark` run (~422–429 ms) is roughly **14×
faster simulated time** than the equivalent SE-mode run above (6.179 s)
for the identical model/op sequence — not yet investigated why (could be
FS mode's simpler system/cache config among other differences), so no
conclusions should be drawn from that gap yet.

## `riscv64_baremetal` (whisper — functional-only RISC-V ISS, no timing model)

| Test/benchmark | Model | Instructions | Wall-clock | Throughput | Notes |
|---|---|---|---|---|---|
| `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 449,804 | 0.04 s | ~10.2M inst/s | Identical arena/output to the gem5 run; unaffected by the `.init_array` fix either way (older test-macro style) |
| `dtln_test` | `dtln_noise_suppression.tflite` | 4,711,964 | 0.38 s | ~12.5M inst/s | Genuine result, post-`.init_array` fix. ~10× the instruction count of `hello_world_test`, consistent with the much larger model |

`whisper_rv64gcv_config.json` declares vector/float support
(`rv64imafdcv_zfh_zvfh_...`, VLEN=512/ELEN=64) that no current TFLM build
actually uses — `riscv64_baremetal` still compiles `rv64imc_zicsr` (no
`v`/`f`/`d`). Its `Fp`/`Vector`/`VectorLoad`/`VectorStore` HPM counters
will read 0 for every run above; it's a placeholder for future
vectorized-kernel comparisons, not a current data source.

## Benchmark candidate comparison (FC/Conv layer shapes)

Pulled directly from each model's flatbuffer via the vendored
`flatbuffers` Python package + `schema_py_generated.py` (no `pip`/full TF
install needed) — see the matrix-optimization discussion in the main
session history. Kept here since it's the basis for picking `dtln_test`
as the benchmark target above.

| Model | Size | Largest FC/Conv layer (input × weight) | M×K×N | MACs |
|---|---|---|---|---|
| `hello_world_float` | 3.2 KB | `[1,16]×[16,16]` | 1×16×16 | 256 |
| `dtln_noise_suppression` | 364 KB | `[1,1,128]×[257,128]` | **1×128×257** | **32,896** |
| `micro_speech_quantized` | 18.4 KB | `[1,25,20,8]→flat×[4,4000]` | 1×4000×4 | 16,000 |
| `memory_footprint` | 976 B | none (`ADD` only) | — | — |
| `person_detect` | 294 KB | none (`CONV_2D`/`DEPTHWISE_CONV_2D` only, no FC layer) | — | — |

`dtln_noise_suppression` was picked as the standing benchmark target:
biggest model, most total FC compute, and a "square-ish" GEMM shape
(K=128, N=257) rather than `micro_speech`'s extreme deep-K/narrow-N shape
(K=4000, N=4).

## Reproducing

```bash
source /home/ajno5/work/2_pattern/tflm/script/0_env_var_setup.sh
cd /home/ajno5/work/2_pattern/tflm/tflite-micro
TOOLCHAIN_ARGS="TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ TARGET_TOOLCHAIN_PREFIX=riscv-none-elf-"

# SE mode, RV64, gem5:
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_generic SIMULATOR=gem5 $TOOLCHAIN_ARGS test_hello_world_test

# FS mode, gem5 (default) or whisper:
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal $TOOLCHAIN_ARGS test_dtln_test
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal SIMULATOR=whisper $TOOLCHAIN_ARGS test_dtln_test

# Generic benchmark harness with a chosen model:
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal $TOOLCHAIN_ARGS \
  BUILD_TYPE=default run_tflm_benchmark \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=153600
```
