# tflm_riscv

RISC-V simulator integration for [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro):
runs TFLM tests/benchmarks under [gem5](https://www.gem5.org/) and
[whisper](https://github.com/chipsalliance/whisper) instead of QEMU.

**Current focus: `riscv64_baremetal` under gem5 FS mode + whisper.** Both
run the exact same bare-metal binaries (custom crt0, no host OS, RISC-V
semihosting for I/O) — gem5 FS mode (`RiscvMinorCPU`, cycle-accurate, the
number to trust for real performance comparisons) and whisper (functional
only, no timing model, ~10-20x faster wall-clock, good for fast iteration
and correctness checks). This pairing is also what's behind the
`riscv64_baremetal_vector` target's RVV-vectorized `FULLY_CONNECTED`
kernel — a 4.74x cycle-count speedup on gem5, correctness-verified via
matching output CRC32 against the scalar baseline. See
[`doc/performance.md`](doc/performance.md) for the numbers and
[`doc/gem5_integration.md`](doc/gem5_integration.md) for how it's built.

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
├── tflite-micro/    git submodule → fork of tensorflow/tflite-micro,
│                    branch `gem5-riscv-integration` (the actual TFLM-side
│                    changes: new/modified Makefile targets, crt0, linker
│                    script, test-runner scripts)
├── sim_config/      gem5 board configs (Python) + whisper config (JSON) —
│                    kept outside the tflite-micro tree since these aren't
│                    TFLM source, they're simulator environment config
├── script/          local dev-environment setup (toolchain/gem5/whisper
│                    paths — edit before use, see below)
└── doc/             gem5_integration.md (the full writeup, see above) +
                     performance.md (consolidated numbers from every run)
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

## Development / git workflow

This repo is two independent git repositories, one nested inside the
other's working tree — that's what a git submodule is. `tflite-micro/` has
its own `.git`, its own remotes, its own history; the outer repo
(`tflm_riscv`) doesn't store its content, only a pointer to one specific
commit of it.

```
tflm_riscv/          outer repo — sim_config/, script/, doc/ (this README's
│                     own remote: origin → tflm_riscv)
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

**To push a config/doc change** (`sim_config/`, `script/`, `doc/`, or this
README):

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
files here (`sim_config/`, `script/`, `doc/`) are this project's own
gem5-integration work built on top of it.
