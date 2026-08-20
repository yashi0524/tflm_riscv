# Microbenchmark: build settings, test config, results

Reference doc for [`../../patterns/microbenchmark/`](../../patterns/microbenchmark/)
(`int8dot_ceiling.c` + `fc_bottleneck.c`) — the standalone, TFLM-library-free
probes this project uses to measure empirical compute-roof and cache-state
ceilings for `MinorCPU`'s single shared `FloatSimd` functional unit,
instead of assuming an idealized figure. Written so **other patterns can
build their own ceiling probes against the same settings** without having
to re-derive them from `doc/gem5_integration.md`'s full investigation
narrative (that doc has the "how we found this" story; this one has the
"how to build/run it and what you get" reference).

Currently only [`../dtln/performance_dtln.md`](../dtln/performance_dtln.md)
consumes these ceilings (via
[`../../script/4_roofline_report.py`](../../script/4_roofline_report.py)'s
`--measure-ceiling`/`--measure-fc-warm-ceiling`/`--measure-cold-ceiling`),
but nothing here is dtln-specific — `fc_bottleneck.c`'s shape (`K=128,N=257`)
happens to match dtln's `FULLY_CONNECTED` layer, and a future pattern with
a different FC/GEMV shape would want its own variant of `fc_bottleneck.c`
at that shape, built the same way.

## Build settings

Both probes build standalone against `riscv64_baremetal_vector`'s own
crt0/linker script directly — **no TFLM library build needed**, so these
numbers are reproducible independent of the `run_tflm_benchmark` pipeline
the rest of this project's patterns use.

- **Toolchain**: GCC 13.4.0-1 by default (`TOOLCHAIN_ROOT ?=
  $(TOOLCHAIN)/bin/`, same `$TOOLCHAIN` env var as everywhere else in this
  project — see `script/0_env_var_setup.sh`); pass
  `TOOLCHAIN_ROOT=../../script/clang_wrapper/` for a clang-18 cross-check
  (see "Investigating clang as an alternative toolchain" in
  [`../gem5_integration.md`](../gem5_integration.md) for why clang isn't
  the default here).
- **Arch/ABI**: `-march=rv64imc_zicsr_zve64x -mabi=lp64 -mcmodel=medany` —
  same `zve64x` vector extension subset as the real kernel build, at
  `VLEN=512`/`ELEN=64` per `sim_config/gem5_riscv_baremetal_fs.py`.
- **CFLAGS**: `-mexplicit-relocs -fno-builtin-printf -funsigned-char
  -fno-delete-null-pointer-checks -fomit-frame-pointer -O2`.
- **LDFLAGS**: `-mexplicit-relocs -mno-relax -nostartfiles -nostdlib -T
  $(TFLM_VEC_DIR)/linker_semi.ld` — links directly against
  `riscv64_baremetal_vector/{start_semi.S,linker_semi.ld}`, the same
  semihosting startup files `dtln_test` itself uses.
- **Layout**: build artifacts and gem5 `m5out*` logs land in two
  independent subdirectories, `int8dot_ceiling/` and `fc_bottleneck/` (not
  interleaved) — see the `Makefile`'s own header comment.
- **Simulator config**: `sim_config/gem5_riscv_baremetal_fs.py` —
  `RiscvMinorCPU`, 1 GHz, `DDR3_1600_8x8`, 64 kB L1 I/D, no L2/L3 — the
  identical config the real `dtln`/`anomaly_detection` benchmarks run
  under, so ceilings and achieved-performance numbers are directly
  comparable.

## Test config

### `int8dot_ceiling.c` (warm ceiling)

Independent-3-chain-unrolled probe replicating `Int8DotProductRvv`'s
actual instruction sequence (`vle8` → `vwadd` ×2 → `vwmul` → `vredsum`,
`LMUL=2`), enough parallel chains to overlap the `FloatSimd` FU's 6-cycle
latency without over-subscribing RVV's 32 physical vector registers (an
8-chain attempt needed up to 64 registers at `LMUL=2`, forcing spill/reload
traffic that swamped the measurement — 3 chains was the largest that fit
cleanly). Small operand buffer stays permanently L1-resident across many
iterations — a *compute/FU-throughput* ceiling, not representative of a
real single-shot invocation's memory environment. Target: `make run`.

### `fc_bottleneck.c` (fc warm / cold ceilings)

Builds the whole real per-output-channel pipeline (dot product → bias add
→ requantize → output-offset → clamp → store), mirroring
`FullyConnected()`'s `out_c` loop exactly, at `dtln`'s real FC shape
(`M=1,K=128,N=257`) and real quantization params pulled from the
flatbuffer (not guessed). Two compile-time flags select what's measured:

- **`FC_VARIANT`** (`-DFC_VARIANT=N`, compile-time — a runtime branch here
  already measured *worse*, see "Tried" in
  [`../gem5_integration.md`](../gem5_integration.md)):
  - `0` = `FULL` — every stage, matches the real kernel exactly
  - `1` = `NO_BIAS` — skip the bias add
  - `2` = `NO_REQUANT` — skip `MultiplyByQuantizedMultiplier`
  - `3` = `NO_CLAMP` — skip the `activation_min`/`max` clamp
  - `4` = `DOT_ONLY` — only `Int8DotProductRvv`, no scalar post-processing
    (closest analog to `int8dot_ceiling.c`, but at this op's real
    `N=257/K=128` shape instead of a synthetic interleaved chain) — **this
    is the variant used for the roofline's fc-warm/cold ceilings**.
- **`FC_CACHE_MODE`** (`-DFC_CACHE_MODE=N`, default `0`):
  - `0` = `WARM` — filter/input/bias written by the init loops immediately
    before timing, one timed pass (`FC_ITERS=1`) — L1-resident from that
    init touch, nothing evicts them before the timed pass runs. A
    *compute* ceiling: the best this op's instruction mix can do assuming
    operands are already cached.
  - `1` = `COLD` — touches a 256 KiB `volatile` distractor buffer (4x this
    target's real 64 KiB L1, confirmed sufficient to fully evict) after
    the init loops but before timing, then the same single timed pass. A
    *memory-latency* ceiling: the best this op can do given its actual,
    unavoidable per-invocation cache-miss pattern — confirmed within 2% of
    the real kernel's directly-instrumented cycles/channel.
  - `FC_ITERS` is fixed at `1` for both modes (as of 2026-08-19 — see
    "Follow-up" in [`../gem5_integration.md`](../gem5_integration.md)):
    originally `WARM` ran `ITERS=20` (reusing the same resident array)
    while `COLD` was forced to `1`, confounding cache state with
    iteration count. Pinning both to a single pass isolates cache state as
    the only variable between the two.

Targets: `make run-fc-bottleneck` (all 5 variants, warm) /
`make run-fc-bottleneck-cold` (all 5, cold) / `make run-fc-warm-ceiling`
and `make run-cold-ceiling` (single-binary `DOT_ONLY`, what
`script/4_roofline_report.py`'s `--measure-fc-warm-ceiling`/
`--measure-cold-ceiling` invoke).

## Results

GCC 13.4.0-1, current as of 2026-08-20:

| Probe | Config | cyc/ch (or /iter) | GFLOP/s | Role |
|---|---|---|---|---|
| `int8dot_ceiling.c` | 3-chain, `LMUL=2` | 206/iter | **3.723** | Warm ceiling |
| `fc_bottleneck.c` | `DOT_ONLY`, `FC_CACHE_MODE=0` | 76 | **3.365** | FC-warm ceiling |
| `fc_bottleneck.c` | `DOT_ONLY`, `FC_CACHE_MODE=1` | 200 | **1.278** | Cold ceiling (the one that applies to a real single-shot `Invoke()`) |

`fc_bottleneck.c`'s full 5-variant warm-vs-cold comparison, same shape
(`FC_ITERS=1` both modes). Numbers below are post the 2026-08-20
requantize-rounding fix (see "Correctness regression probes" below) —
`DOT_ONLY` and `NO_REQUANT` are unaffected (neither calls
`MultiplyByQuantizedMultiplier` at all), `FULL`/`NO_BIAS`/`NO_CLAMP` all
got slightly slower (the corrected double-rounding path has one more
conditional than the previously-hardcoded, and wrong, single-rounding
path):

| Variant | Warm cyc/ch | Warm GFLOP/s | Cold cyc/ch | Cold GFLOP/s | Cold/warm |
|---|---|---|---|---|---|
| `FULL` (0) | 131 | 1.946 | 256 | 0.999 | 1.95x |
| `NO_BIAS` (1) | 123 | 2.068 | 254 | 1.005 | 2.07x |
| `NO_REQUANT` (2) | 92 | 2.767 | 224 | 1.138 | 2.43x |
| `NO_CLAMP` (3) | 115 | 2.208 | 243 | 1.050 | 2.11x |
| `DOT_ONLY` (4) | 76 | 3.365 | 200 | 1.278 | 2.63x |

Every variant slows ~2-2.6x cold vs. warm — consistent with the dot
product (memory-bound either way) dominating all of them. See "Realistic
`FULLY_CONNECTED` bottleneck decomposition" in
[`../gem5_integration.md`](../gem5_integration.md) for the full derivation
(including why the cold `FULL` ceiling lands close to, and is briefly
*beaten* by, the real kernel's directly-instrumented performance) and
"Known limitations / follow-ups not yet done" there for an open
discrepancy in the real kernel's own vectorized `FULLY_CONNECTED` cycle
count still being investigated.

## Correctness regression probes

Two standalone cross-checks against a plain scalar reference (not
ceilings — added while root-causing `anomaly_detection_int8.tflite`'s
vectorized-output CRC32 mismatch, see
[`../anomaly_detection/performance_anomaly_detection.md`](../anomaly_detection/performance_anomaly_detection.md)'s
"Correctness" section and [`../gem5_integration.md`](../gem5_integration.md)'s
"Known limitations" for the full story):

- **`int8dot_correctness_probe.c`** (`make run-int8dot-correctness`):
  `Int8DotProductRvv` (copied verbatim, same convention as
  `fc_bottleneck.c`) vs. a plain `int64_t` scalar reference, at every
  `(K, input_offset)` combination `anomaly_detection_int8.tflite`'s 10
  `FULLY_CONNECTED` layers actually use, plus edge cases (`K=1`, `VLMAX±1`
  at `K=639`/`641`). **All pass** — rules `Int8DotProductRvv` itself out
  as the bug source.
- **`requant_correctness_probe.c`** (`make run-requant-correctness`):
  `MultiplyByQuantizedMultiplierInlined`'s hardcoded single-rounding
  algorithm vs. a reimplementation of `common.cc`'s actual double-rounding
  (gemmlowp) algorithm, at `anomaly_detection_int8.tflite`'s 10 real
  `(multiplier, shift)` pairs (computed from the flatbuffer's scales via
  TFLite's `QuantizeMultiplier`), 2000 random accumulator values each.
  **455/20,000 mismatch, every one off by exactly `±1`** — confirmed root
  cause of the CRC32 mismatch, and latent in every RVV
  `FULLY_CONNECTED`/LSTM call in this project (not just this model).
  **Fixed 2026-08-20**: `MultiplyByQuantizedMultiplierInlined` now
  mirrors `common.cc`'s `#if TFLITE_SINGLE_ROUNDING`/`#else` structure
  exactly, calling the real `gemmlowp` functions directly instead of
  hardcoding one branch — this probe (and the mismatch counts above) is
  kept as-is, describing the pre-fix bug, as a permanent regression test
  against reintroducing it.
- **`fc_bottleneck.c` had the identical bug independently, in its own
  copy of `MultiplyByQuantizedMultiplier`** (not this probe — the probe
  only tested the real kernel's copy) — same hardcoded single-rounding,
  never checked against fixing the real kernel first, so it went out of
  sync in the opposite direction until also fixed 2026-08-20 (same
  `#if`/`#else` structure, reimplemented from gemmlowp's scalar
  specializations since this file builds as plain C with no TFLM include
  path — can't `#include` the real, C++-template, `fixedpoint.h`). Only
  affects the `FULL`/`NO_BIAS`/`NO_CLAMP` variants below (`DOT_ONLY` and
  `NO_REQUANT` never call this function at all) — the three ceilings
  above were never affected either way.
