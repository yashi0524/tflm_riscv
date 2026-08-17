# TFLM RISC-V Performance Log

Consolidated numbers from every verified run across this project's
simulator integrations. For the *why* behind each target/fix, see
[`gem5_integration.md`](gem5_integration.md) — this file is just the
numbers, kept as a single scannable reference.

**Caveat that applies to every row below:** TFLM's per-op software timing
instrumentation isn't wired up by default — every op prints `0 ticks (0
ms)` unless both (a) a profiler is explicitly passed into
`MicroInterpreter` (only `run_tflm_benchmark` does this;
`hello_world_test`/`dtln_test`/etc. don't) and (b) the target has its own
`micro_time.cc` reading a real cycle counter (only `riscv64_baremetal`
has one, added specifically for this). Where neither holds, the only
trustworthy timing figures are the simulator's own whole-run counters:
gem5's `tick` count (cycle-accurate, `RiscvMinorCPU`) or whisper's
instruction count (functional-only, no timing model — **not**
comparable to gem5 ticks, just useful as a relative "how much work did
this do" signal and for fast iteration).

**For real per-op cycle counts, the vectorized `FULLY_CONNECTED` kernel,
and a full roofline analysis — all specific to `dtln_noise_suppression`
— see [`performance_dtln.md`](performance_dtln.md).** This file stays the
consolidated whole-run log across every benchmark in the project;
`dtln_test`/`dtln_noise_suppression` below is just one row/one model among
several.

## `riscv64_baremetal` (FS / bare-metal mode, gem5 — cycle-accurate `RiscvMinorCPU`)

| Test/benchmark | Model | Arena | Binary size | gem5 ticks | Simulated time | Notes |
|---|---|---|---|---|---|---|
| `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 2,408 B | 147,232 B | 672,762,000 | ~0.67 ms | FLASH usage ~87.4 KB (of the original 128 KB budget, since bumped to 4 MB) |
| `micro_utils_test` | n/a (unit test, 8 cases) | — | — | 126,455,000 | ~0.13 ms | Genuine result, post-`.init_array` fix (was silently vacuous before) |
| `dtln_test` | `dtln_noise_suppression.tflite` (LSTM + 1 `FULLY_CONNECTED`, `M=1,K=128,N=257`) | — | — | 5,996,072,000 | ~6.0 ms | Genuine result, post-`.init_array` fix (was silently vacuous before). Slowest/biggest `riscv64_baremetal` run so far. Per-op breakdown + roofline: [`performance_dtln.md`](performance_dtln.md). |
| `tflm_benchmark` | `person_detect.tflite` (30 ops) | 89,248 B | — | 429,485,090,000 | ~429 ms | Post-`.init_array`-fix regression check; matches pre-fix 422,108,725,000 (~422 ms) within normal run-to-run variance |

Note the FS-mode `tflm_benchmark` run (~422–429 ms) is roughly **14×
faster simulated time** than the equivalent SE-mode run (6.179 s, see the
historical SE-mode table at the bottom of this file) for the identical
model/op sequence — not yet investigated why (could be FS mode's simpler
system/cache config among other differences), so no conclusions should be
drawn from that gap yet.

## `riscv64_baremetal` (whisper — functional-only RISC-V ISS, no timing model)

| Test/benchmark | Model | Instructions | Wall-clock | Throughput | Notes |
|---|---|---|---|---|---|
| `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 449,804 | 0.04 s | ~10.2M inst/s | Identical arena/output to the gem5 run; unaffected by the `.init_array` fix either way (older test-macro style) |
| `dtln_test` | `dtln_noise_suppression.tflite` | 4,711,964 | 0.38 s | ~12.5M inst/s | Genuine result, post-`.init_array` fix. ~10× the instruction count of `hello_world_test`, consistent with the much larger model. Per-op breakdown + roofline: [`performance_dtln.md`](performance_dtln.md). |

`whisper_rv64gcv_config.json` declares vector/float support
(`rv64imafdcv_zfh_zvfh_...`, VLEN=512/ELEN=64) that no current TFLM build
actually uses — `riscv64_baremetal` still compiles `rv64imc_zicsr` (no
`v`/`f`/`d`). Its `Fp`/`Vector`/`VectorLoad`/`VectorStore` HPM counters
will read 0 for every run above; it's a placeholder for future
vectorized-kernel comparisons, not a current data source.

## Reproducing

```bash
source /home/ajno5/work/2_pattern/tflm/script/0_env_var_setup.sh
cd /home/ajno5/work/2_pattern/tflm/tflite-micro
TOOLCHAIN_ARGS="TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.4.0-1/bin/ TARGET_TOOLCHAIN_PREFIX=riscv-none-elf-"

# SE mode, RV64, qemu (SIMULATOR=gem5 is disabled here — see the
# historical SE-mode section at the bottom of this file):
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_generic $TOOLCHAIN_ARGS test_hello_world_test

# FS mode, gem5 (default) or whisper — pass/fail smoke test, whole-run
# ticks only (no per-op cycles; for those and the roofline analysis, see
# performance_dtln.md's Reproducing section instead):
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal $TOOLCHAIN_ARGS test_dtln_test
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal SIMULATOR=whisper $TOOLCHAIN_ARGS test_dtln_test
```

## `riscv{32,64}_generic` (SE / syscall-emulation mode, gem5) — historical, now disabled

> `SIMULATOR=gem5` on these targets is now disabled (`$(error ...)`) — see
> "gem5 SE mode disabled" in [`gem5_integration.md`](gem5_integration.md).
> `riscv64_baremetal` + `SIMULATOR=whisper` covers the same
> fast/functional-simulator role these numbers represent. Kept here as a
> historical record of what was measured before the switch.

| RV width | Test/benchmark | Model | Arena | gem5 ticks | Simulated time |
|---|---|---|---|---|---|
| RV32 | `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 1,376 B | not recorded | not recorded |
| RV64 | `hello_world_test` | hello_world (1 `FULLY_CONNECTED`) | 2,408 B | not recorded | not recorded |
| RV64 | `tflm_benchmark` | `person_detect.tflite` (30 ops) | 89,248 B | 6,179,398,577,000 | 6.179 s |
