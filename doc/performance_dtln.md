# `dtln_noise_suppression.tflite` Performance & Roofline Analysis

Deep-dive performance analysis for `dtln_noise_suppression.tflite`, split
out of [`performance.md`](performance.md) (which stays the consolidated
whole-run log across every benchmark in this project — `dtln_test` is just
one row there). This file covers everything specific to `dtln`: real
per-op cycle counts, the vectorized `FULLY_CONNECTED` **and** LSTM kernels
(both `int8` gate matmuls now take the RVV fast path — see below), the
roofline analysis built on top of it, and why `dtln_noise_suppression` was
picked as the standing benchmark target in the first place.

**Toolchain: GCC 13.4.0-1** (`1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.4.0-1/`),
the project default as of the LSTM vectorization fix below — GCC 13.2.0-2
(the previous default, still installed alongside it) miscompiles the fixed
`Int8DotProductRvv` at `-O2`. See "LSTM vectorization" in
[`gem5_integration.md`](gem5_integration.md) for the full story.

**Same caveat as `performance.md`'s intro applies:** TFLM's per-op software
timing instrumentation isn't wired up by default — every op prints `0
ticks (0 ms)` unless both (a) a profiler is explicitly passed into
`MicroInterpreter` (only `run_tflm_benchmark` does this; `dtln_test`
doesn't) and (b) the target has its own `micro_time.cc` reading a real
cycle counter (`riscv64_baremetal`/`riscv64_baremetal_vector` both do).
Every number in this file comes from `run_tflm_benchmark`, not `dtln_test`
directly — see the callout in "Vectorized `FULLY_CONNECTED`" below for why.

## Per-op cycle counts: `dtln_noise_suppression.tflite` on `riscv64_baremetal`

Real per-op profiling, not `0 ticks` — see "Per-op cycle counts on
`riscv64_baremetal`" in [`gem5_integration.md`](gem5_integration.md) for
how (`micro_time.cc` reading `mcycle`, plus running via
`run_tflm_benchmark` instead of `dtln_test` directly, since only the
former wires a `MicroProfiler`). `GENERIC_BENCHMARK_ARENA_SIZE=16384`.

| Op | gem5 cycles | whisper cycles |
|---|---|---|
| `UNIDIRECTIONAL_SEQUENCE_LSTM` (1st call) | 2,685,618 | 2,479,287 |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` (2nd call) | 1,845,791 | 1,688,145 |
| `FULLY_CONNECTED` | 378,379 | 311,697 |
| `LOGISTIC` | 89,537 | 88,405 |
| **Total (profiled ops only)** | **4,999,325** | **4,567,534** |

Output CRC32 (`0x7E578D1C`) identical between simulators. gem5's numbers
are the trustworthy ones for actual performance comparison (models real
pipeline stalls); whisper's are close but not cycle-accurate (functional
simulator, closer to an idealized-IPC assumption) — good for fast relative
comparison, not absolute numbers.

**The LSTM, not the `FULLY_CONNECTED` layer, dominates** — ~91% of total
profiled cycles either way. Relevant if/when comparing a vectorized
`FULLY_CONNECTED` kernel against this baseline: the FC layer alone is a
small fraction of this model's total cost.

`person_detect.tflite` hasn't been run through this same per-op profiling
path yet — deliberately skipped given its ~7–9 minute gem5 wall-clock cost.

## Vectorized `FULLY_CONNECTED` and LSTM (`riscv64_baremetal_vector`) vs. scalar baseline

See "A vectorized `FULLY_CONNECTED` kernel" and "LSTM vectorization" in
[`gem5_integration.md`](gem5_integration.md) for the full implementation
history, including two real correctness bugs found/fixed along the way:
(1) an `if constexpr` type guard that didn't check `OutputType`, which let
the fast path incorrectly apply to `lstm_eval.cc`'s internal `int16_t`
gate matmuls before the underlying bug (below) was actually fixed — caught
via an Output CRC32 mismatch, `0x50433D2B` vs. the correct `0x7E578D1C`;
and (2) the *real* bug that guard was masking: `Int8DotProductRvv` folded
`input_offset`/`filter_offset` into a widening-add whose scalar operand is
only 8 bits wide, silently truncating a legal offset value of exactly
`128` (e.g. `zero_point=-128`) — latent in the FC kernel too, just never
triggered by FC's own zero-point (`+4`). Fixed via the standard
gemmlowp-style algebraic expansion (offsets applied via scalar `int32_t`
arithmetic after vector reduction, never passed to a narrow vector
intrinsic). That fix also uncovered a GCC 13.2 `-O2` codegen bug blocking
it from working at all — resolved by upgrading to **GCC 13.4.0-1** (see
above).

| | gem5 (cycle-accurate) | whisper (functional, no timing model) |
|---|---|---|
| Baseline `FULLY_CONNECTED` | 378,379 | 311,697 |
| Vectorized `FULLY_CONNECTED` | 87,788 | 30,873 |
| **Speedup** | **4.31×** | **10.09×** (not representative — see below) |
| Baseline `UNIDIRECTIONAL_SEQUENCE_LSTM` (1st call) | 2,685,618 | 2,479,287 |
| Vectorized `UNIDIRECTIONAL_SEQUENCE_LSTM` (1st call) | 621,580 | 221,910 |
| **Speedup** | **4.32×** | **11.17×** |
| Baseline `UNIDIRECTIONAL_SEQUENCE_LSTM` (2nd call) | 1,845,791 | 1,688,145 |
| Vectorized `UNIDIRECTIONAL_SEQUENCE_LSTM` (2nd call) | 435,055 | 183,513 |
| **Speedup** | **4.24×** | **9.20×** |

Output CRC32 (`0x7E578D1C`) identical to baseline in every case — verified
correct, not just faster; also cross-validated against 3 other
FC/LSTM shapes entirely (`mnist_lstm`'s `trained_lstm_int8` — a genuinely
different LSTM with a 28-timestep sequential call, not just a different
shape; `micro_speech_quantized`'s extreme `K=4000,N=4` FC;
`hello_world_int8`'s tiny `K=16,N=16` FC) — all matched their respective
scalar-target baselines exactly. gem5's ~4.2–4.3× is the number to trust
for each op; whisper's much larger speedups are inflated by having no
cycle-accurate memory/pipeline model (can't capture the real cost of the
vector loads), so don't read them as real-hardware expectations.

Whole-model effect — now both `FULLY_CONNECTED` **and** the LSTM
(~91% of total cycles) are vectorized:

| | gem5 total ticks | whisper total ticks |
|---|---|---|
| Baseline | 4,999,325 | 4,567,534 |
| With vectorized FC only (historical intermediate step) | 4,369,141 | 4,281,314 |
| With vectorized FC + LSTM | **1,233,600** | **524,701** |
| **Whole-model speedup (FC + LSTM vs. baseline)** | **~4.05×** | **~8.71×** |

Vectorizing FC alone only bought ~12.6% whole-model speedup, since the
LSTM — the actual bulk of the work — was untouched. Vectorizing both is
the win the roofline analysis below predicted was available.

Target: `riscv64_baremetal_vector` (`-march=rv64imc_zicsr_zve64x`) — a
separate `TARGET` from plain `riscv64_baremetal` deliberately, to avoid
`GENDIR` cache collisions between the two `-march=` variants (this build
has no `.d` header-dependency tracking at all — a real gotcha discovered
along the way, see the doc — so mixing arches under one `TARGET` risks
silently linking stale objects).

> **If you want to do roofline analysis, build/run with
> `TARGET=riscv64_baremetal_vector`, not plain `riscv64_baremetal`.** The
> roofline's peak-compute ceiling below is defined by the RVV vector unit's
> int8 throughput (`VLEN=512`) — only `riscv64_baremetal_vector` binaries
> (`-march=...zve64x`) actually contain RVV instructions. Plain
> `riscv64_baremetal` is pure scalar (`rv64imc_zicsr`, no `v` extension) and
> structurally can't approach that ceiling regardless of optimization, so
> its achieved-performance points aren't meaningful to plot against a
> vector-unit-based peak line. (Concrete illustration of the same fact:
> whisper's `Vector`/`VectorLoad`/`VectorStore` HPM counters read 0 on the
> scalar target — there are no RVV instructions to count at all.)
>
> **Also run `run_tflm_benchmark`, not `test_dtln_test`, as the binary.**
> `dtln_test.cc` constructs `MicroInterpreter` without a `MicroProfiler`
> (see the caveat at the top of this file), so it never calls
> `GetCurrentTimeTicks()` and its log has no per-op cycle counts at all —
> only whole-run pass/fail. `run_tflm_benchmark`
> (`generic_model_benchmark.cc`) is the one harness that wires a real
> `MicroProfiler` into the interpreter, which is where every per-op number
> in this section (and the roofline's achieved-performance points) comes
> from — see [`script/2_run_benchmark.sh`](../script/2_run_benchmark.sh)
> (or the `/run_benchmark_tflm` command, which defaults to this target).

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal_vector $TOOLCHAIN_ARGS \
  BUILD_TYPE=default run_tflm_benchmark \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=16384
```

## Roofline analysis

Shapes pulled straight from the `.tflite` flatbuffer via
[`script/3_extract_lstm_shapes.py`](../script/3_extract_lstm_shapes.py) (same
technique as the "Benchmark candidate comparison" table below, extended to
`UNIDIRECTIONAL_SEQUENCE_LSTM`'s 24 fixed input operand slots — gate
weights/biases/state/peephole/projection/layer-norm — instead of
`FULLY_CONNECTED`'s plain `[M,K]x[K,N]`). `dtln_noise_suppression.tflite`'s
LSTM has no peephole/projection/layer-norm tensors populated (plain LSTM),
so each call is just 8 gate matmuls: 4× `input_to_*_weights` + 4×
`recurrent_to_*_weights`.

### Machine parameters

Same board as the sibling `gemm` project's
(`sim_config/gem5_riscv_baremetal_fs.py`): `RiscvMinorCPU`, 1 GHz,
`VLEN=512`/`ELEN=64`, `DDR3_1600_8x8`, no L2/L3 (64 kB L1 I/D only).

- Peak BW = 1600 MT/s × 8 B = **12.8 GB/s**
- Peak int8 compute (widening MAC, `vl = VLEN/SEW = 512/8 = 64` int8
  elements/instr, idealized 1 vector-MAC-instruction/cycle ceiling — same
  simplifying assumption the `gemm` project's roofline uses for its FP64
  case): `64 × 2 FLOP/MAC × 1 GHz` = **128 GFLOP/s**
- Ridge point = 128 / 12.8 = **10 FLOP/byte**

### Arithmetic intensity

Unlike the `gemm` project (which reads real `VectorLoad`/`VectorStore` HPM
counts from whisper, since its tiled/blocked kernel has no closed-form
memory-traffic formula), `Q` here is computed directly from the flatbuffer
weight shapes rather than measured HPM counters — these are all
batch-1 (`M=1`) int8 GEMVs with no blocking, so every weight byte is read
exactly once and the memory traffic has an exact closed form (this also
sidesteps gem5 MinorCPU's `mhpmcounterN` being unusable — like
`TimingSimpleCPU` in the `gemm` project's notes, `RiscvMinorCPU` has no
configurable HPM event model and just aliases every `mhpmcounterN` to the
cycle counter). Activation/bias/output bytes are omitted (a few hundred
bytes vs. tens of thousands of weight bytes — under 2% effect on AI).

| | MACs | FLOPs (2×MACs) | Weight bytes (int8) | AI (FLOP/byte) |
|---|---|---|---|---|
| `FULLY_CONNECTED` (`M=1,K=128,N=257`) | 32,896 | 65,792 | 32,896 | 2.0 |
| LSTM 1st call (`in=257→hid=128`) | 4×(128×257)+4×(128×128) = 197,120 | 394,240 | 197,120 | 2.0 |
| LSTM 2nd call (`in=128→hid=128`) | 4×(128×128)+4×(128×128) = 131,072 | 262,144 | 131,072 | 2.0 |

**All three land at exactly the same AI = 2.0 FLOP/byte** — not a
coincidence, every one of these is a batch-1 int8 GEMV with zero weight
reuse, so the FLOP/byte ratio is fixed by the arithmetic alone regardless of
which gate or op it is. `AI (2.0) << ridge (10.0)` → solidly **memory-bound**
for all three; the memory-bound ceiling (attainable performance) is
`AI × peak_BW = 2.0 × 12.8 = 25.6 GFLOP/s` for every op here.

### Achieved performance vs. the roofline (gem5, cycle-accurate)

`T = cycles / 1e9 s` (1 GHz clock); `P = FLOPs / T`; `efficiency = P /
attainable`. Cycles from the per-op `MicroProfiler` table above — scalar
rows from the `riscv64_baremetal` build, vectorized rows from
`riscv64_baremetal_vector` with both the `FULLY_CONNECTED` and LSTM RVV
fixes applied (GCC 13.4.0-1; see "Vectorized `FULLY_CONNECTED` and LSTM"
above).

| | Cycles | T (µs) | P (MFLOP/s) | Efficiency vs. 25.6 GFLOP/s ceiling | Cycles/weight-byte |
|---|---|---|---|---|---|
| `FULLY_CONNECTED` (scalar baseline) | 378,176 | 378.18 | 173.97 | 0.68% | 11.50 |
| `FULLY_CONNECTED` (vectorized) | 87,788 | 87.79 | 749.44 | 2.93% | 2.67 |
| LSTM 1st call (scalar) | 2,683,720 | 2683.72 | 146.90 | 0.57% | 13.61 |
| LSTM 1st call (vectorized) | 621,580 | 621.58 | 634.25 | 2.48% | 3.15 |
| LSTM 2nd call (scalar) | 1,845,580 | 1845.58 | 142.04 | 0.55% | 14.08 |
| LSTM 2nd call (vectorized) | 435,055 | 435.06 | 602.55 | 2.35% | 3.32 |

**Verdict: real ~4.2–4.3× cycle-count wins per op, but still nowhere close
to saturating the memory-bound ceiling.** Even fully vectorized, every op
here reaches only ~2.3–2.9% of the 12.8 GB/s bandwidth-bound ceiling. This
confirms the bottleneck being fixed here was never really DDR bandwidth;
it's the in-order scalar pipeline's per-byte overhead (~11.5–14.1
cycles/weight-byte scalar → ~2.7–3.3 cycles/byte vectorized, processing 64
int8 elements/instruction instead of one). That still leaves the vast
majority of headroom against peak DRAM bandwidth unused across the board —
substantial further room (loop unrolling, prefetching, wider `LMUL`,
precomputing the filter-derived row-sum once per `Invoke()` instead of
recomputing it — see "Known limitations of this specific kernel" in
`gem5_integration.md`) before bandwidth itself becomes the binding
constraint for any of these ops.

**LSTM vectorization: achieved.** What blocked this in an earlier pass —
a genuine correctness bug in `Int8DotProductRvv` (an offset-value-`128`
truncation, latent in the FC kernel too) plus a GCC 13.2 `-O2` codegen bug
— is now fixed. See "LSTM vectorization" in
[`gem5_integration.md`](gem5_integration.md) for the full root-cause
writeup, the fix, the GCC 13.4.0-1 toolchain upgrade that unblocked it,
and cross-shape validation against 3 other models.

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
TOOLCHAIN_ARGS="TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.4.0-1/bin/ TARGET_TOOLCHAIN_PREFIX=riscv-none-elf-"

# Generic benchmark harness, with real per-op cycle counts (gem5 shown; add
# SIMULATOR=whisper for the fast functional-only path). This is what
# produces every number in this file:
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=riscv64_baremetal_vector $TOOLCHAIN_ARGS \
  BUILD_TYPE=default run_tflm_benchmark \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=16384

# person_detect.tflite works the same way, but budget ~7-9 min gem5 wall-clock:
# GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite
# GENERIC_BENCHMARK_ARENA_SIZE=153600
```

Or via the wrapper scripts (equivalent, less typing):

```bash
# All 4 combinations (scalar/vector x gem5/whisper):
/home/ajno5/work/2_pattern/tflm/script/2_run_benchmark.sh both

# Vector target only (what the roofline section above needs) — also the
# default if no argument is given, and what `/run_benchmark_tflm` runs:
/home/ajno5/work/2_pattern/tflm/script/2_run_benchmark.sh vector

# Pass/fail smoke test only (test_dtln_test, no per-op cycles — see
# performance.md's whole-run tables for what this produces):
/home/ajno5/work/2_pattern/tflm/script/1_run_pattern.sh
```
