# tflm_riscv

RISC-V simulator integration for [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro):
runs TFLM tests/benchmarks under [gem5](https://www.gem5.org/) and
[whisper](https://github.com/chipsalliance/whisper) instead of (or in
addition to) QEMU, across three target/mode combinations:

- **SE (syscall-emulation) mode, gem5** — `riscv32_generic`/`riscv64_generic`,
  ordinary newlib+Linux-ABI binaries, same execution model QEMU linux-user
  already uses for these targets, just via gem5's SE mode instead.
- **FS (full-system) bare-metal mode, gem5** — a new `riscv64_baremetal`
  target that boots the ELF directly at its entry point with no host OS
  underneath at all: a custom crt0, its own linker script, and RISC-V
  semihosting for I/O. Cycle-accurate (`RiscvMinorCPU`), so it's slower
  to run than the next option.
- **Same `riscv64_baremetal` binaries, whisper** — whisper is a
  functional-only RISC-V ISS (no timing model) that also supports
  semihosting directly, so it runs the exact same bare-metal binaries gem5
  FS mode does, just much faster (no cycle-accurate simulation) — useful
  for quick iteration. Its config already declares vector/float support
  (`rv64gcv`-class ISA) even though no current TFLM build here compiles
  with it yet — ready for whenever a vectorized kernel build exists.

See [`doc/gem5_integration.adoc`](doc/gem5_integration.adoc) for the full
writeup: design decisions, bugs found/fixed along the way (a newlib
`.sdata`/`.sbss` linker-script bug that silently broke `printf`, an
RWX-segment issue under `-Wl,--fatal-warnings`, FLASH/RAM sizing for
`tflm_benchmark`, why the whisper config declares extensions the binary
doesn't use, etc.), and verified commands/output for each target.

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
└── doc/             the full writeup (see above)
```

## Setup

1. Clone with submodules:
   ```
   git clone --recurse-submodules https://github.com/yashi0524/tflm_riscv.git
   ```
2. You'll need, separately:
   - A RISC-V GCC toolchain with `rv32imc`/`rv64imc_zicsr` multilib support
     (this was built/tested against the
     [xPack RISC-V Embedded GCC](https://xpack.github.io/dev-tools/riscv-none-elf-gcc/)
     13.2.0 distribution).
   - [gem5](https://github.com/gem5/gem5) built for the `RISCV` ISA
     (`build/RISCV/gem5.opt`), for `SIMULATOR=gem5` (the default).
   - [whisper](https://github.com/chipsalliance/whisper) built with an
     `RV64` config, for `SIMULATOR=whisper` on `riscv64_baremetal`
     (optional — only needed if you want the faster functional-only path).
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

# SE mode, RV64:
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_generic SIMULATOR=gem5 \
  TARGET_TOOLCHAIN_ROOT=<your-toolchain>/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test

# FS bare-metal mode, RV64, via gem5 (default, cycle-accurate):
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
```

`GEM5_SE_CONFIG`/`GEM5_FS_CONFIG`/`WHISPER_CONFIG` (consumed by
`tensorflow/lite/micro/testing/test_with_{gem5,gem5_fs,whisper}.sh` inside
the submodule) default to `sim_config/gem5_riscv_se.py` /
`sim_config/gem5_riscv_baremetal_fs.py` / `sim_config/whisper_rv64gcv_config.json`
relative to the submodule's own location — i.e. they resolve correctly as
long as this repo's layout above is preserved, no extra configuration
needed. Override any of these env vars to point elsewhere if you want a
different board/simulator config.

See `doc/gem5_integration.adoc` for the RV32 command form, `tflm_benchmark`
usage, and everything else that's been verified.

## License

`tflite-micro` (the submodule) is Apache 2.0, from
[tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro). The
files here (`sim_config/`, `script/`, `doc/`) are this project's own
gem5-integration work built on top of it.
