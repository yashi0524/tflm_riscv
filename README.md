# tflm_riscv

RISC-V simulator integration for [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro):
runs TFLM tests/benchmarks under [gem5](https://www.gem5.org/) and
[whisper](https://github.com/chipsalliance/whisper) instead of QEMU —
built specifically to develop and empirically validate an **RVV
(RISC-V Vector extension) int8 kernel** for TFLM's `FULLY_CONNECTED`/
`UNIDIRECTIONAL_SEQUENCE_LSTM` ops. That kernel work is this project's
actual focus; the simulator integration exists to measure it.

**RVV is the core contribution.** TFLM's reference int8 `FULLY_CONNECTED`
kernel is scalar; this project adds an RVV `Int8DotProductRvv` fast path
(`tflite-micro/tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h`,
gated on the `riscv64_baremetal_vector` target's `-march=...zve64x`) and
reuses the identical fast path for LSTM's internal gate matmuls, since
both reduce to the same batch-1 int8 GEMV shape. Every change to it is
checked two ways before being trusted: **correctness** — output CRC32
matched against the untouched scalar baseline, cross-checked across 5
different model/shape combinations — and **performance** — measured
against ceilings from standalone, TFLM-library-free probes
([`patterns/microbenchmark/`](patterns/microbenchmark/)) that directly
exercise `MinorCPU`'s single `FloatSimd` functional unit, not
assumed/idealized peak-FLOPs numbers. Both run under gem5 FS mode
(`RiscvMinorCPU`, cycle-accurate, the number to trust) and whisper
(functional only, no timing model, ~10-20x faster wall-clock, good for
fast iteration).

## Performance summary

Two models measured so far, gem5 cycle-accurate (`RiscvMinorCPU`) —
whisper's instruction counts aren't comparable to these, see
`doc/performance.md`'s caveat:

| | Vectorized vs. scalar | Efficiency vs. measured cold-cache ceiling |
|---|---|---|
| `dtln` — whole model (`FULLY_CONNECTED` + 2x LSTM) | **~6.20×** (806,800 vs. 4,999,325 cycles) | — |
| `dtln` — `FULLY_CONNECTED` alone | **6.90×** | **87.32%** |
| `dtln` — LSTM, 1st / 2nd call | **6.79× / 6.83×** | 70.57% / 65.37% |
| `anomaly_detection` — whole test (10x `FULLY_CONNECTED`) | **~4.95×** (657,290 vs. 3,252,758 cycles) | — |

**Performance vs. roofline ceiling** (`anomaly_detection`, `K=128,N=128`
layers — 6 of its 10 `FULLY_CONNECTED` calls share this shape):

| | Efficiency vs. cold-cache ceiling |
|---|---|
| Scalar | ~14.6-14.9% |
| Vectorized | ~82-87% |

(One of the six, call 9, measured 110.80% in the build behind that
range — flagged ⚠ in the per-call table as the same build-to-build cycle
variance noted below, not a stable per-layer result, so excluded here.
Full per-call table, all 10 shapes:
[`doc/anomaly_detection/performance_anomaly_detection.md`](doc/anomaly_detection/performance_anomaly_detection.md).)

"Efficiency vs. cold-cache ceiling" is against a ceiling this project
measured itself (`patterns/microbenchmark/fc_bottleneck.c`, same op, same
real single-shot cache behavior), not a naive peak-FLOPs figure — 87%
means the vectorized kernel is already close to what this hardware can
physically do for this op, not that there's an easy 13% left on the
table. `dtln`'s full per-op breakdown and roofline plot:
[`doc/dtln/performance_dtln.md`](doc/dtln/performance_dtln.md).

The single biggest win **wasn't the vector ISA at all**: force-inlining
the int32 requantize step (`MultiplyByQuantizedMultiplier`) onto the hot
path, instead of an out-of-line call, cut `dtln`'s whole-model cycles by
**29.34%** on its own (`FULLY_CONNECTED` alone: **-34.93%**) — found by
directly instrumenting where the vectorized kernel's cycles were actually
going, not by guessing. This work also surfaced and fixed two real
correctness bugs along the way (a requantize rounding-algorithm mismatch
affecting every vectorized `FULLY_CONNECTED`/LSTM call project-wide, and
an 8-bit offset-truncation bug) — see "Known limitations" in
[`doc/gem5_integration.md`](doc/gem5_integration.md) for both. Cycle
counts vary a small amount build-to-build (observed, exact trigger not
yet identified — see the same doc); the figures above are from a build
confirmed stable across 5 independent rebuilds.

**Backup / historical: gem5 SE mode**, on `riscv32_generic`/`riscv64_generic`
(ordinary newlib+Linux-ABI binaries, the same execution model QEMU
linux-user already covers for these targets) — **now disabled**
(`SIMULATOR=gem5` errors on those targets with a `$(error ...)` pointing
here) once `riscv64_baremetal` + whisper started covering the same
fast/functional role. Kept in the repo, not deleted, in case it's ever
revisited; `riscv{32,64}_generic` still works fine under the default
`SIMULATOR=qemu`.

See [`doc/gem5_integration.md`](doc/gem5_integration.md) for the full
writeup: design decisions, bugs found/fixed along the way (a newlib
`.sdata`/`.sbss` linker-script bug that silently broke `printf`, an
RWX-segment issue under `-Wl,--fatal-warnings`, FLASH/RAM sizing for
`tflm_benchmark`, a silent-vacuous-test-pass bug from a missing
`.init_array` call, the vectorized-kernel work, etc.), and verified
commands/output for every target.

## Layout

```
tflm_riscv/
├── tflite-micro/       · Git submodule → fork of tensorflow/tflite-micro,
│                         branch `gem5-riscv-integration` (the actual
│                         TFLM-side changes: new/modified Makefile
│                         targets, crt0, linker script, test-runner
│                         scripts, plus the RVV `Int8DotProductRvv`
│                         kernel). `examples/dtln/` lives natively here;
│                         `examples/anomaly_detection/` is a symlink out
│                         to `patterns/anomaly_detection/` below (see
│                         "Patterns")
├── patterns/           · Example patterns kept in the outer repo instead
│   ├── anomaly_detection/    of inside the submodule, symlinked into
│   │                         tflite-micro's examples/ so its Makefile
│   │                         still finds them — MLPerf Tiny's `ad01`
│   │                         anomaly-detection autoencoder (10x
│   │                         FULLY_CONNECTED, no conv/LSTM)
│   └── microbenchmark/     · Standalone, TFLM-library-free C probes
│                             (`int8dot_ceiling.c`, `fc_bottleneck.c` +
│                             two correctness-regression probes) that
│                             measure this project's roofline ceilings
│                             directly against `MinorCPU`'s `FloatSimd`
│                             unit — own Makefile, no `make
│                             third_party_downloads` needed
├── sim_config/         · Gem5 board configs (Python) + whisper config
│                         (JSON) — kept outside the tflite-micro tree
│                         since these aren't TFLM source, they're
│                         simulator environment config
├── script/             · Local dev-environment setup (toolchain/gem5/
│                         whisper paths, run/benchmark helper scripts,
│                         roofline tooling — see "Running" below)
└── doc/
    ├── gem5_integration.md  · The full writeup (design decisions, bugs
    │                          found/fixed, known limitations)
    ├── performance.md       · Consolidated whole-run numbers across
    │                          every benchmark/target in the project
    ├── dtln/                · performance_dtln.md (per-op cycle counts,
    │                          vectorized-kernel results, roofline
    │                          analysis) + its generated roofline SVG/log
    ├── anomaly_detection/    · Same, for the anomaly_detection pattern,
    │                          plus work_note_mlperf_tiny.md
    └── microbenchmark/       · Shared reference: build settings, test
                               config, and results for the probes in
                               patterns/microbenchmark/, so other
                               patterns can build their own ceiling
                               probes without re-deriving them
```

## Setup

1. Clone with submodules:
   ```
   git clone --recurse-submodules https://github.com/yashi0524/tflm_riscv.git
   ```
2. You'll need, separately:
   - A RISC-V GCC toolchain with `rv32imc`/`rv64imc_zicsr`/`rv64imc_zicsr_zve64x`
     multilib support (this was built/tested against the
     [xPack RISC-V Embedded GCC](https://xpack.github.io/dev-tools/riscv-none-elf-gcc/)
     13.2.0 distribution).
   - [gem5](https://github.com/gem5/gem5) built for the `RISCV` ISA
     (`build/RISCV/gem5.opt`) — `riscv64_baremetal`'s default simulator (FS
     mode, cycle-accurate).
   - [whisper](https://github.com/chipsalliance/whisper) built with an
     `RV64` config, for `SIMULATOR=whisper` on `riscv64_baremetal`/
     `riscv64_baremetal_vector` — not strictly required to use gem5 alone,
     but core to the current fast-iteration workflow, not just a nice-to-have.
3. Edit `script/0_env_var_setup.sh` — it currently has this machine's paths
   hardcoded (`TOOLCHAIN`, `GEM5_PATH`, `WHISPER_PATH`). Point them at your
   own toolchain/gem5/whisper builds (`WHISPER_PATH` only matters if you're
   using `SIMULATOR=whisper`).
4. `source script/0_env_var_setup.sh` before any `make`/`gem5.opt`/`whisper`
   invocation below — it puts `gem5.opt`/`whisper` on `PATH` and computes
   toolchain library paths.

## Running

```bash
source script/0_env_var_setup.sh
cd tflite-micro

# FS bare-metal mode, RV64, via gem5 (default, cycle-accurate) -- primary target:
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal \
  TARGET_TOOLCHAIN_ROOT=<your-toolchain>/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test

# Same target/binary, via whisper instead (functional-only, much faster):
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal SIMULATOR=whisper \
  TARGET_TOOLCHAIN_ROOT=<your-toolchain>/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test

# The RVV-vectorized FULLY_CONNECTED kernel, benchmarked against
# dtln_noise_suppression.tflite via the generic benchmark harness (gem5
# shown; add SIMULATOR=whisper for the faster functional-only path):
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal_vector \
  TARGET_TOOLCHAIN_ROOT=<your-toolchain>/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  BUILD_TYPE=default run_tflm_benchmark \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=16384

# Backup/historical: SE mode, RV64 -- SIMULATOR=gem5 is disabled here (see
# above); riscv{32,64}_generic still works under the default SIMULATOR=qemu:
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_generic \
  TARGET_TOOLCHAIN_ROOT=<your-toolchain>/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test
```

`GEM5_FS_CONFIG`/`WHISPER_CONFIG` (consumed by
`tensorflow/lite/micro/testing/test_with_{gem5_fs,whisper}.sh` inside
the submodule) default to `sim_config/gem5_riscv_baremetal_fs.py` /
`sim_config/whisper_rv64gcv_config.json` relative to the submodule's own
location — i.e. they resolve correctly as
long as this repo's layout above is preserved, no extra configuration
needed. Override any of these env vars to point elsewhere if you want a
different board/simulator config.

See `doc/gem5_integration.md` for the RV32 command form, `tflm_benchmark`
usage, and everything else that's been verified. See
`doc/performance.md` for a consolidated table of every measured
run (gem5 tick counts, whisper instruction counts, arena sizes).

`script/1_run_pattern.sh` and `script/2_run_benchmark.sh` wrap the two
command forms above (`test_<pattern>_test` and `run_tflm_benchmark`) with
this project's own paths pre-filled — edit the `TOOLCHAIN_ARGS`/`TFLM_HOME`
lines at the top before use, same as `0_env_var_setup.sh`.

## Patterns

Three example patterns exercise the RVV-vectorized `FULLY_CONNECTED`
kernel, each integrated a different way:

| Pattern | Where it lives | How it's wired in |
|---|---|---|
| `dtln` (noise suppression) | `tflite-micro/tensorflow/lite/micro/examples/dtln/` | native to the submodule |
| `anomaly_detection` (MLPerf Tiny `ad01`) | [`patterns/anomaly_detection/`](patterns/anomaly_detection/) | symlinked into the submodule's `examples/` so its own Makefile still finds it — see its [README](patterns/anomaly_detection/README.md) |
| `microbenchmark` (roofline ceiling probes) | [`patterns/microbenchmark/`](patterns/microbenchmark/) | standalone, no TFLM library build at all — own `Makefile`, builds straight against the target's crt0/linker script; see [`doc/microbenchmark/README.md`](doc/microbenchmark/README.md) |

Each of `dtln`/`anomaly_detection` has a roofline analysis under
`doc/<pattern>/`, generated from `microbenchmark`'s measured ceilings via:

```bash
source script/0_env_var_setup.sh
cd tflite-micro
python3 ../script/4_roofline_report.py --measure-ceiling --measure-fc-warm-ceiling --measure-cold-ceiling \
  --model tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite \
  --output ../doc/dtln/roofline_log.txt
python3 ../script/5_gen_roofline_svg.py --log ../doc/dtln/roofline_log.txt --out ../doc/dtln/dtln_roofline.svg
```

(swap `--model`/`--log`/`--out` for `anomaly_detection`'s paths under
`doc/anomaly_detection/` to regenerate that one instead). `4_roofline_report.py --help`
lists the flags to reuse an already-measured ceiling instead of
re-running the probes.

## Development / git workflow

This repo is two independent git repositories, one nested inside the
other's working tree — that's what a git submodule is. `tflite-micro/` has
its own `.git`, its own remotes, its own history; the outer repo
(`tflm_riscv`) doesn't store its content, only a pointer to one specific
commit of it.

```
tflm_riscv/          outer repo — patterns/, sim_config/, script/, doc/
│                     (this README's own remote: origin → tflm_riscv)
└── tflite-micro/     inner repo — the actual TFLM source patches
                       (own remotes: origin → your tflite-micro fork,
                        upstream → the real tensorflow/tflite-micro)
```

Remotes, as set up:

| Repo | `origin` | `upstream` |
|---|---|---|
| outer (`tflm_riscv`) | `https://github.com/<you>/tflm_riscv.git` | *(none)* |
| inner (`tflite-micro`) | `https://github.com/<you>/tflite-micro.git` (your fork) | `https://github.com/tensorflow/tflite-micro.git` (real upstream) |

Branch tracking: inner repo's `main` tracks `upstream/main` (stays a clean
mirror — don't commit to it directly); do your own work on
`gem5-riscv-integration` (or another feature branch), which tracks
`origin/...` on your fork. Outer repo's `main` tracks `origin/main`.

**To push a TFLM source change** (crt0, linker script, Makefile targets,
test-runner scripts — anything under `tflite-micro/`):

```bash
cd tflite-micro
git checkout gem5-riscv-integration   # not main
# ... edit, git add, git commit ...
git push                              # → your fork
```

**To push a config/doc/pattern change** (`patterns/`, `sim_config/`,
`script/`, `doc/`, or this README):

```bash
# from the outer repo root
# ... edit, git add, git commit ...
git push
```

**If you did both** (changed TFLM source *and* want the outer repo to
reference the new commit), push the inner repo first, then stage the
updated submodule pointer in the outer repo:

```bash
cd tflite-micro && git push && cd ..
git add tflite-micro     # stages the new commit pointer, not file content
git commit -m "Bump tflite-micro submodule"
git push
```

Until that last step, `git status` in the outer repo will show
`tflite-micro` as having "new commits" — that's expected, it just means the
outer repo's pointer hasn't been bumped yet, not an error.

**Syncing the fork with real upstream** (pulling in new tflite-micro
releases): fetch/merge `upstream/main` into your feature branch from inside
`tflite-micro/` — normal git, no submodule-specific steps needed:

```bash
cd tflite-micro
git fetch upstream
git checkout gem5-riscv-integration
git merge upstream/main   # or rebase, if you prefer
git push
```

## License

`tflite-micro` (the submodule) is Apache 2.0, from
[tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro). The
files here (`patterns/`, `sim_config/`, `script/`, `doc/`) are this
project's own gem5-integration work built on top of it.
