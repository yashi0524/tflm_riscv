# tflm_riscv

gem5 RISC-V integration for [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro):
runs TFLM tests/benchmarks under [gem5](https://www.gem5.org/) instead of
(or in addition to) QEMU, in two modes:

- **SE (syscall-emulation) mode** — `riscv32_generic`/`riscv64_generic`,
  ordinary newlib+Linux-ABI binaries, same execution model QEMU linux-user
  already uses for these targets, just via gem5's SE mode instead.
- **FS (full-system) bare-metal mode** — a new `riscv64_baremetal` target
  that boots the ELF directly at its entry point with no host OS
  underneath at all: a custom crt0, its own linker script, and RISC-V
  semihosting for I/O.

See [`doc/gem5_integration.adoc`](doc/gem5_integration.adoc) for the full
writeup: design decisions, bugs found/fixed along the way (a newlib
`.sdata`/`.sbss` linker-script bug that silently broke `printf`, an
RWX-segment issue under `-Wl,--fatal-warnings`, FLASH/RAM sizing for
`tflm_benchmark`, etc.), and verified commands/output for each target.

## Layout

```
tflm_riscv/
├── tflite-micro/    git submodule → fork of tensorflow/tflite-micro,
│                    branch `gem5-riscv-integration` (the actual TFLM-side
│                    changes: new/modified Makefile targets, crt0, linker
│                    script, test-runner scripts)
├── sim_config/      gem5 board configs (Python, gem5's own config API) —
│                    kept outside the tflite-micro tree since these aren't
│                    TFLM source, they're gem5 environment config
├── script/          local dev-environment setup (toolchain/gem5 paths —
│                    edit before use, see below)
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
     (`build/RISCV/gem5.opt`).
3. Edit `script/0_env_var_setup.sh` — it currently has this machine's paths
   hardcoded (`TOOLCHAIN`, `GEM5_PATH`, `WHISPER_PATH`). Point `TOOLCHAIN`
   and `GEM5_PATH` at your own toolchain/gem5 build; `WHISPER_PATH` is
   optional (a different RISC-V simulator, not required for the gem5 flow
   here).
4. `source script/0_env_var_setup.sh` before any `make`/`gem5.opt`
   invocation below — it puts `gem5.opt` on `PATH` and computes toolchain
   library paths.

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

# FS bare-metal mode, RV64:
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal \
  TARGET_TOOLCHAIN_ROOT=<your-toolchain>/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test
```

`GEM5_SE_CONFIG`/`GEM5_FS_CONFIG` (consumed by
`tensorflow/lite/micro/testing/test_with_gem5{,_fs}.sh` inside the
submodule) default to `sim_config/gem5_riscv_se.py` /
`sim_config/gem5_riscv_baremetal_fs.py` relative to the submodule's own
location — i.e. they resolve correctly as long as this repo's layout above
is preserved, no extra configuration needed. Override either env var to
point elsewhere if you want a different board config.

See `doc/gem5_integration.adoc` for the RV32 command form, `tflm_benchmark`
usage, and everything else that's been verified.

## License

`tflite-micro` (the submodule) is Apache 2.0, from
[tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro). The
files here (`sim_config/`, `script/`, `doc/`) are this project's own
gem5-integration work built on top of it.
