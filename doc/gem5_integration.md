# TFLite Micro: gem5 Support for riscv32_generic / riscv64_generic

## Table of contents

- [Overview](#overview)
- [Environment blockers fixed first](#environment-blockers-fixed-first)
- [gem5 SE-mode design](#gem5-se-mode-design)
  - [Two non-obvious gem5 API requirements hit along the way](#two-non-obvious-gem5-api-requirements-hit-along-the-way)
- [Files changed](#files-changed)
  - [New, outside the tflite-micro clone (`/home/ajno5/work/2_pattern/tflm/sim_config/`)](#new-outside-the-tflite-micro-clone-homeajno5work2_patterntflmsim_config)
  - [New, inside the tflite-micro clone (tracked)](#new-inside-the-tflite-micro-clone-tracked)
  - [Modified](#modified)
- [Verified working commands](#verified-working-commands)
  - [`run_tflm_benchmark` (generic model benchmark) on `riscv64_generic` + gem5](#run_tflm_benchmark-generic-model-benchmark-on-riscv64_generic--gem5)
- [gem5 Full-System (FS) bare-metal mode: `hello_world_test`](#gem5-full-system-fs-bare-metal-mode-hello_world_test)
  - [New files](#new-files)
  - [Bug found and fixed: `.sdata`/`.sbss` were silently excluded from copy/zero ranges](#bug-found-and-fixed-sdatasbss-were-silently-excluded-from-copyzero-ranges)
  - [Bug found and fixed: `.init_array`/`.fini_array`/`.preinit_array` are emitted writable, producing an RWX FLASH segment](#bug-found-and-fixed-init_arrayfini_arraypreinit_array-are-emitted-writable-producing-an-rwx-flash-segment)
  - [Wired into the Makefile as a proper target: `riscv64_baremetal`](#wired-into-the-makefile-as-a-proper-target-riscv64_baremetal)
  - [Verified working commands](#verified-working-commands-1)
  - [`tflm_benchmark` on `riscv64_baremetal`, and two more linker-script fixes it surfaced](#tflm_benchmark-on-riscv64_baremetal-and-two-more-linker-script-fixes-it-surfaced)
- [Whisper support on `riscv64_baremetal`: a second, faster simulator](#whisper-support-on-riscv64_baremetal-a-second-faster-simulator)
  - [Why the whisper config declares vector/float support the binary doesn't use](#why-the-whisper-config-declares-vectorfloat-support-the-binary-doesnt-use)
- [The `.init_array` bug: silent vacuous test passes on `riscv64_baremetal`](#the-init_array-bug-silent-vacuous-test-passes-on-riscv64_baremetal)
- [gem5 SE mode disabled](#gem5-se-mode-disabled)
- [Per-op cycle counts on `riscv64_baremetal`: the two gaps that made every op read "0 ticks"](#per-op-cycle-counts-on-riscv64_baremetal-the-two-gaps-that-made-every-op-read-0-ticks)
- [A vectorized `FULLY_CONNECTED` kernel: `riscv64_baremetal_vector`](#a-vectorized-fully_connected-kernel-riscv64_baremetal_vector)
  - [The math, and why it needed a genuinely new target](#the-math-and-why-it-needed-a-genuinely-new-target)
  - [The fast path itself](#the-fast-path-itself)
  - [Bug found and fixed: the `if constexpr` guard didn't check `OutputType`](#bug-found-and-fixed-the-if-constexpr-guard-didnt-check-outputtype)
  - [Results](#results)
  - [Known limitations of this specific kernel](#known-limitations-of-this-specific-kernel)
  - [LSTM vectorization: a real bug found and fixed in `Int8DotProductRvv`, unblocked by upgrading GCC](#lstm-vectorization-a-real-bug-found-and-fixed-in-int8dotproductrvv-unblocked-by-upgrading-gcc)
  - [`Int8DotProductRvv`: widened from base `LMUL=1` to `LMUL=2`](#int8dotproductrvv-widened-from-base-lmul1-to-lmul2)
  - [Why efficiency stays low even after vectorization: `MinorCPU`'s single shared `FloatSimd` functional unit](#why-efficiency-stays-low-even-after-vectorization-minorcpus-single-shared-floatsimd-functional-unit)
  - [Measuring the real ceiling: `microbenchmark/int8dot_ceiling.c`](#measuring-the-real-ceiling-microbenchmarkint8dot_ceilingc)
  - [Correction: the 2-FU experiment helps the real kernel much less than the synthetic probe suggested](#correction-the-2-fu-experiment-helps-the-real-kernel-much-less-than-the-synthetic-probe-suggested)
  - [The ceiling saturates at 2 FUs: `MinorCPU`'s 2-wide issue front end, not FU count, is the real limiter](#the-ceiling-saturates-at-2-fus-minorcpus-2-wide-issue-front-end-not-fu-count-is-the-real-limiter)
  - [Realistic FULLY_CONNECTED bottleneck decomposition: vector work dominates, not scalar overhead](#realistic-fully_connected-bottleneck-decomposition-vector-work-dominates-not-scalar-overhead)
  - [Applied: inlining `MultiplyByQuantizedMultiplier` — the single biggest win found in this project](#applied-inlining-multiplybyquantizedmultiplier--the-single-biggest-win-found-in-this-project)
  - [Tried: interleaving independent output-channel dot-product chains — mixed result, not applied](#tried-interleaving-independent-output-channel-dot-product-chains--mixed-result-not-applied)
  - [Investigating clang as an alternative toolchain: builds clean, but crashes gem5 on the full `dtln` model](#investigating-clang-as-an-alternative-toolchain-builds-clean-but-crashes-gem5-on-the-full-dtln-model)
- [Known limitations / follow-ups not yet done](#known-limitations--follow-ups-not-yet-done)

## Overview

This memo documents the changes made to add gem5 as an alternative to QEMU
for running `tflite-micro` tests/benchmarks on RISC-V, plus the environment
fixes needed to get the existing Makefile-based build working at all on
this host. Scope: `hello_world_test`, verified working under gem5 for both
`riscv32_generic` (RV32IMC) and `riscv64_generic` (RV64IMC, a new target —
did not exist upstream before this work).

Repo: `/home/ajno5/work/2_pattern/tflm/tflite-micro` (a real git clone of
https://github.com/tensorflow/tflite-micro). Per-environment config that
shouldn't be added to that clone lives in `/home/ajno5/work/2_pattern/tflm/`
instead (`sim_config/`, `script/`).

See [`performance.md`](performance.md) for a consolidated table of every
measured run (gem5 tick counts, whisper instruction counts, arena sizes) —
this file covers the *why*, that one has just the numbers.

> **Update:** gem5 SE mode (the `riscv{32,64}_generic` sections below) has
> since been **disabled** — `SIMULATOR=gem5` now hits a `$(error ...)` on
> those targets. Reasoning: `riscv64_baremetal` + `SIMULATOR=whisper` covers
> the fast/functional-simulator role gem5 SE mode was filling, and gem5 FS
> mode (`riscv64_baremetal`'s default) covers the cycle-accurate role — so
> gem5 SE mode was redundant. The sections below are kept as-is (accurate
> history of what was built/verified); see "gem5 SE mode disabled" further
> down for the actual change.

## Environment blockers fixed first

None of these are specific to gem5 — they blocked the existing
QEMU-based `make ... test_hello_world_test` flow too, and had to be
resolved before any RISC-V target would build at all on this host.

| Symptom | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` | System Python had no `numpy`; `pip3` wasn't installed either. | `sudo apt install python3-numpy` |
| `ModuleNotFoundError: No module named 'PIL'` | Same, for Pillow. | `sudo apt install python3-pil` |
| `unzip: command not found` while downloading the `ruy` third-party dep | `unzip` wasn't installed. | `sudo apt install unzip` |
| Re-running after installing `unzip` still failed on a missing header (`fixedpoint/fixedpoint.h`) | `gemmlowp` and `ruy` had each been downloaded *before* `unzip` was available; the Makefile's download step created the target directory before failing, so on retry it saw the directory already existed and skipped re-downloading — leaving both directories present but empty. | `rmdir tensorflow/lite/micro/tools/make/downloads/{gemmlowp,ruy}`, then re-run so they get re-fetched properly. |
| `riscv64-unknown-elf-g++: 1: ELF: not found` / `Syntax error: word unexpected` | The Makefile's default `RISCV_TOOLCHAIN_URL` (a 2018-era SiFive package) is an **x86_64** binary; this host is **aarch64** (`uname -m` → `aarch64`). The shell tried to `execve()` an incompatible-architecture ELF, which failed and fell through to interpreting the raw bytes as a shell script. | Override `TARGET_TOOLCHAIN_ROOT` / `TARGET_TOOLCHAIN_PREFIX` on the `make` command line to point at the aarch64-native toolchain already used for the sibling `gemm` project (`xpack-riscv-none-elf-gcc-13.2.0-2`, prefix `riscv-none-elf-` instead of the default `riscv64-unknown-elf-`). Confirmed this toolchain has `rv32imc`/`rv64imc` multilib support (`-imultilib rv32imc/ilp32` / `.../rv64imc/lp64` showed up in the actual `cc1plus` invocation). |

## gem5 SE-mode design

The existing RISC-V targets run tests via **QEMU linux-user mode**
(`qemu-riscv32`/`qemu-riscv64`), *not* full-system emulation: the toolchain
produces ordinary statically-linked ELF binaries whose `write()`/`exit()`/
etc. calls are serviced as Linux syscalls, with QEMU translating them to
the host. This is confirmed by `debug_log.cc`'s implementation
(`std::fputs` via plain libc `stdio`, no semihosting/UART driver in sight)
and by `test_with_qemu.sh` invoking plain `qemu-riscv32` (not
`qemu-system-riscv32`).

gem5's direct equivalent of QEMU linux-user mode is **SE (syscall
emulation) mode** — `Root(full_system=False, ...)` plus a `Process()`
object, no kernel/bootloader/UART model needed. This is a materially
different (and much simpler) setup than the bare-metal M-mode +
semihosting configs built earlier for the `gemm` project's RVV work, which
solved a different problem (a custom linker script + semihosting console,
because that toolchain/runtime target was genuinely bare-metal).

### Two non-obvious gem5 API requirements hit along the way

1. **A CPU's `isa` must explicitly match the process's word size.**
   gem5 auto-selects `RiscvProcess32`/`RiscvProcess64` from the ELF's
   `EI_CLASS`, but separately `fatal_if(isa->rvType() != RV32/RV64, ...)`
   checks the *CPU's* configured ISA against it — the two aren't
   automatically kept in sync. Fix: `gem5_riscv_se.py` reads the ELF header's
   `EI_CLASS` byte itself (offset 4: `1`→RV32, `2`→RV64) and sets
   `RiscvISA(riscv_type=...)` to match, so one config file works for both
   `riscv32_generic` and `riscv64_generic` without a manual flag that could
   be set wrong.
2. **SE mode needs an explicit `SEWorkload`, separate from `cpu.workload`.**
   First attempt (`system.cpu.workload = process` only) failed at
   instantiation with:

   ```
   fatal: fatal condition !seWorkload occurred: Couldn't find appropriate workload object.
   ```

   Fix: `system.workload = SEWorkload.init_compatible(binary_path)` — a
   factory that inspects the ELF and returns the right `SEWorkload` subclass
   (`RiscvEmuLinux` here). Found by reading gem5's own (deprecated but still
   functional) `configs/deprecated/example/se.py` reference script.

## Files changed

### New, outside the tflite-micro clone (`/home/ajno5/work/2_pattern/tflm/sim_config/`)

**`gem5_riscv_se.py`**
gem5 SE-mode board config. Auto-detects RV32 vs RV64 from the target ELF
(see above); `--cpu={atomic,timing,minor}` selects the CPU model, default
`minor` (`RiscvMinorCPU`, matching the `gemm` project's convention).
`enable_rvv=False` always, since both `riscv{32,64}_generic` targets build
for `*imc` (no vector extension).

Kept outside the repo deliberately (per instruction) since it's
environment-specific (hardcodes this host's `gem5.opt`/paths), unlike the
wrapper script below which is a portable, project-integrated file.

**`gem5_riscv_baremetal_fs.py`**
gem5 FS-mode board config for the `riscv64_baremetal` target (see the FS
mode section further below) — `RiscvMinorCPU`, `RiscvBareMetal` workload,
`RiscvSemihosting()`. Fixed single-CPU-model config, no `--cpu=` switch
(unlike the SE-mode config above), since bare-metal/semihosting boot is a
different enough execution model that CPU-model flexibility wasn't worth
the complexity here yet. Also kept outside the repo for the same reason as
`gem5_riscv_se.py`.

### New, inside the tflite-micro clone (tracked)

**`tensorflow/lite/micro/testing/test_with_gem5.sh`**
Drop-in alternative to `test_with_qemu.sh` — same argument shape
(`arch-suffix cpu binary pass-string target-name`), same behavior (run,
tee to a log, grep for the pass string, `exit 0`/`1`). Internally invokes
`gem5.opt -d <m5out> <GEM5_SE_CONFIG> --cpu=<cpu> <binary>`.
`GEM5_BIN`/`GEM5_SE_CONFIG` env vars are overridable; the latter defaults
to the path above.

**`tensorflow/lite/micro/riscv64_generic/debug_log.cc`**
Copy of `riscv32_generic`'s (XLEN-agnostic — just `vsnprintf_` + `fputs`).
Needed because the Makefile's `specialize_files.py` step picks per-target
overrides from `tensorflow/lite/micro/$(TARGET)/`, keyed on the `TARGET`
name.

**`tensorflow/lite/micro/tools/make/targets/riscv64_generic_makefile.inc`**
New target (did not exist upstream). Mirrors `riscv32_generic_makefile.inc`
with `RISCV_ARCH := rv64imc`, `RISCV_ABI := lp64`; otherwise identical
(same `SIMULATOR` switch, same `--specs=nano.specs`/`-mno-relax` flags,
same excluded-tests list).

**`tensorflow/lite/micro/riscv64_baremetal/{start_semi.S,linker_semi.ld,debug_log.cc}`**
The bare-metal crt0, linker script, and `DebugLog()` override for the new
`riscv64_baremetal` target — see the FS mode section below for the full
story (`.sdata`/`.sbss` copy/zero fix, RWX-segment/`PHDRS` fix, `exit()`
vs. `_exit()`, `putchar_` stub).

**`tensorflow/lite/micro/testing/test_with_gem5_fs.sh`**
Counterpart to `test_with_gem5.sh` for FS mode — invokes
`gem5_riscv_baremetal_fs.py` instead of the SE-mode config, no `--cpu=`
flag (fixed `RiscvMinorCPU`). `GEM5_BIN`/`GEM5_FS_CONFIG` env vars
overridable, same convention as the SE-mode script.

**`tensorflow/lite/micro/tools/make/targets/riscv64_baremetal_makefile.inc`**
New target (did not exist upstream, and unlike `riscv{32,64}_generic` has
no upstream SE/QEMU-mode analog to mirror). Full bare-metal FS-mode target:
custom crt0 linked in via `MICROLITE_CC_SRCS +=`, custom linker script,
`-nostartfiles -nostdlib` + explicit `-lc -lm -lgcc`, `test_with_gem5_fs.sh`
as `TEST_SCRIPT`. See the dedicated section below for the full rationale.

### Modified

**`tensorflow/lite/micro/tools/make/targets/riscv32_generic_makefile.inc`**
Added a `SIMULATOR ?= qemu` variable; `TEST_SCRIPT` now branches on it
(`gem5` → `test_with_gem5.sh riscv32 minor`, anything else → the original
`test_with_qemu.sh riscv32 rv32` unchanged). Default behavior is
unaffected — existing QEMU-based CI/workflows keep working as before.

## Verified working commands

```bash
source /home/ajno5/work/2_pattern/tflm/script/0_env_var_setup.sh   # puts gem5.opt on PATH
cd /home/ajno5/work/2_pattern/tflm/tflite-micro

# RV32, under gem5:
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv32_generic SIMULATOR=gem5 \
  TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test

# RV64, under gem5 (same command, TARGET swapped):
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_generic SIMULATOR=gem5 \
  TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test
```

Both end in `~~~ALL TESTS PASSED~~~` / `Pass` / exit code 0.

Sanity-check numbers (arena size scales with pointer width, as expected —
not a bug):

| Target | `RecordingMicroAllocator` arena total |
|---|---|
| `riscv32_generic` (4-byte pointers) | 1,376 bytes |
| `riscv64_generic` (8-byte pointers) | 2,408 bytes |

Also still confirmed working, unaffected by any of the above: QEMU-based
`SIMULATOR=qemu` (the default) for `riscv32_generic`, and the plain default
(`linux`/`aarch64`) target's `test_hello_world_test`, `tflm_benchmark`, and
`run_keyword_benchmark`.

### `run_tflm_benchmark` (generic model benchmark) on `riscv64_generic` + gem5

Also verified, using the repo's built-in `person_detect.tflite` model:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_generic SIMULATOR=gem5 \
  TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  BUILD_TYPE=default run_tflm_benchmark \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=153600
```

Ran clean — all 30 ops (13× `DEPTHWISE_CONV_2D`, 14× `CONV_2D`,
`AVERAGE_POOL_2D`, `RESHAPE`, `SOFTMAX`) executed, arena usage
`89,248 B` total (`61.96%` non-persistent / `38.04%` persistent), gem5
exited cleanly at `tick 6,179,398,577,000` (6.179 s of simulated time —
much larger than `hello_world_test`'s ~11 ms, expected for a real
conv-based model vs. a single `FULLY_CONNECTED` op).

Two caveats noticed, neither gem5-related:

- Every per-op timing prints `0 ticks (0 ms)` — a pre-existing TFLM
  limitation: this target's software-side profiling clock isn't wired up
  for `riscv{32,64}_generic`. gem5's own tick count (`Exiting @ tick ...`)
  is the only trustworthy timing figure here.
- `collect_meta_data.sh`'s metadata step logs
  `/usr/bin/python3: No module named pip` and falls back to
  `Model analysis not available` — non-fatal, same `pip`-not-installed gap
  noted earlier in this session.

## gem5 Full-System (FS) bare-metal mode: `hello_world_test`

Separate from SE mode above, this section covers running `hello_world_test`
under gem5's **FS (full-system) mode** — booting an ELF directly at its
linked entry point with no host OS underneath at all (no kernel, no Linux
syscall emulation). This reuses the bare-metal crt0/linker-script/semihosting
approach built for the sibling `gemm` project's RVV work
(`/home/ajno5/work/2_pattern/gemm/{start_semi.S,linker_semi.ld}` and its FS
gem5 config), adapted for TFLM.

### New files

**`tensorflow/lite/micro/riscv64_baremetal/start_semi.S`**
Copy of gemm's `start_semi.S` (crt0: stack/gp/`mtvec`/BSS-zero/`.data`-copy
setup, then `call main`), with three changes for TFLM:

- Assembled with `-march=rv64imc_zicsr` — TFLM's plain `rv64imc` lacks the
  `zicsr` extension needed for the `csrw`/`csrr` instructions this crt0 uses
  (`mtvec`, `mstatus`, `mcause`, `mepc`, `mtval`).
- `_start` now does `call main` then **`call exit`** (libc's real `exit()`,
  which flushes all open stdio streams via `_fwalk` before internally
  calling `_exit()`) instead of falling straight through into the raw
  `_exit` stub as gemm's copy does. Gemm's kernel happens to always end on a
  flushed line so the gap never mattered there; TFLM's test output isn't
  guaranteed to, so skipping `exit()` risked losing buffered output.
- Added a `putchar_` stub (routes one byte through this file's own
  `_write`). TFLM bundles the `eyalroz_printf` library; linking pulls in
  its `printf.o` whole, and that object also defines `putchar_wrapper`
  (backing `putchar()`/`puts()`, which TFLM doesn't actually call) that
  references an external `putchar_` — unresolved otherwise.

**`tensorflow/lite/micro/riscv64_baremetal/linker_semi.ld`**
Copy of gemm's linker script, with one **correctness fix** (see below) —
not TFLM-specific, latent in the original gemm script too.

### Bug found and fixed: `.sdata`/`.sbss` were silently excluded from copy/zero ranges

First attempt: relinking `hello_world_test`'s objects against
`start_semi.o`/`linker_semi.ld` produced a binary that ran to a clean
semihosting exit under gem5 FS mode — but printed **nothing**, not even a
trap message. A raw `_write(1, "boot ok\n", 8)` call inserted directly into
`_start` (bypassing libc entirely) printed fine, proving crt0 + semihosting
+ the gem5 FS config were all correct; the bug was specifically in
buffered stdio (`printf`/`puts`/`fputs`, even `fflush`) never reaching
`_write` at all.

Root cause, found via a gem5 `--debug-flags=Exec` instruction trace:
`_impure_ptr` — newlib's global reentrancy-struct pointer, which every
buffered-stdio call dereferences first — read back as `0` at runtime
despite being a properly-initialized `D` (data) symbol in the ELF. Cause:
this RISC-V toolchain places *small* globals (`_impure_ptr`, `errno`,
`__lock_*` mutexes, `__malloc_*` bookkeeping, the `__stdio` init flag —
anything eligible for gp-relative addressing) into `.sdata`/`.sbss`
sections, not `.data`/`.bss`. The linker script's `.data`/`.bss` output
sections only matched `*(.data*)`/`*(.bss*)` patterns, so the linker fell
back to its orphan-section placement heuristic for `.sdata*`/`.sbss*` —
which put them in their own sections immediately adjacent to `.data`/`.bss`,
but **outside** the `[_data_start,_data_end)`/`[_bss_start,_bss_end)` ranges
that `start_semi.S`'s copy-from-FLASH and zero-BSS loops actually cover.
Net effect: those symbols kept whatever gem5's simulated DRAM happened to
already contain (zero, in this case) — `_impure_ptr` never got its real
`&_impure_data` value, silently breaking every libc stdio call it touches
without ever faulting (usually; forcing `setvbuf()` down a different
code path with the same corrupted pointer did produce a genuine misaligned
load trap, which is what made the bug legible).

Fix: added `*(.sdata) *(.sdata.*) *(.gnu.linkonce.s.*)` into the `.data`
output section (before `_data_end`) and `*(.sbss) *(.sbss.*)
*(.gnu.linkonce.sb.*) *(.scommon)` into the `.bss` output section (before
`_bss_end`) in `linker_semi.ld`. Confirmed fix with a minimal standalone
`printf("...")` test binary before reapplying to the full TFLM link.

The same latent bug was present in gemm's original
`/home/ajno5/work/2_pattern/gemm/linker_semi.ld` and has since been fixed
there too (same edit); rebuilt `test/dgemm_riscv` and reran under gemm's own
`gem5_riscv_demo_riscv_baremetal_semihost_minor.py` FS config to confirm no
regression — output (`misa`, `"Starting Scalar DGEMM..."`, `vl`,
`mcycle`/`minstret`/vector-counter dump) still prints correctly. It simply
hadn't been *tripped* before: gemm's own runs apparently didn't exercise
whichever stdio code path first dereferences the corrupted `_impure_ptr`
badly enough to lose output, but the underlying exclusion of `.sdata`/`.sbss`
from the copy/zero ranges was there regardless.

### Bug found and fixed: `.init_array`/`.fini_array`/`.preinit_array` are emitted writable, producing an RWX FLASH segment

TFLM's Makefile adds `-Wl,--fatal-warnings` for every gcc-toolchain target
(`tools/make/Makefile`, unconditionally for `TOOLCHAIN=gcc` + non-osx), so
any linker warning is a hard build failure — not just cosmetic here. Linking
through the normal `%_test` rule surfaced:
`warning: ... has a LOAD segment with RWX permissions`, which
`--fatal-warnings` turns into `collect2: error: ld returned 1 exit status`.

Just changing `MEMORY`'s `RAM (rwx)` to `RAM (rw)` in `linker_semi.ld`
didn't fix it — `readelf -l` showed the *FLASH* segment (`.text.init`/
`.text`/`.rodata`/…) was the one marked `RWE`, not RAM. Cause: GNU ld
computes each `PT_LOAD` segment's permission flags as the **union of every
input section's `SHF_*` flags** it contains, not from the `MEMORY` region's
declared attributes. `.init_array`/`.fini_array`/`.preinit_array` are
emitted `SHF_WRITE` by gcc (conventionally, in case a loader ever needs to
relocate the constructor-pointer array at load time) even though nothing in
this bare-metal, no-relocation static build ever writes to them — but
because they share FLASH's `PT_LOAD` segment with `.text`, that `W` bit
unions into the whole segment.

Fix: added an explicit `PHDRS` block (`flash PT_LOAD FLAGS(5)` = R+X,
`ram PT_LOAD FLAGS(6)` = R+W) and assigned every output section to `:flash`
or `:ram` explicitly, which overrides the automatic union-of-inputs
computation entirely. One follow-on issue this surfaced: assigning `.heap`/
`.stack` (pure runtime reservations with zero real ELF content) to the same
`:ram` PHDR as `.data` made the linker try to compute a contiguous LMA for
them by chasing `.data`'s `AT>FLASH` load address across a nonsensical
half-megabyte range, producing `section .heap lma ... adjusted to ...`
warnings (also fatal). Fixed by marking `.heap`/`.stack` `(NOLOAD)` — they
were always pure address-space reservations, never meant to have file
content or a load address in the first place.

Confirmed clean (`readelf -l` shows `R E` for the FLASH segment, `RW` for
RAM, no `X`) with both `-Wl,--fatal-warnings` and `-Wl,--gc-sections`
active — the same flags the Makefile always passes.

### Wired into the Makefile as a proper target: `riscv64_baremetal`

Unlike the `SIMULATOR=gem5` switch on `riscv{32,64}_generic` (same target,
alternate test runner), FS mode needed a genuinely different target: a
different crt0 (no default startfiles), a different linker script, no
`--specs=nano.specs`, and explicit `-lc -lm -lgcc` (since `-nostdlib` drops
the default lib auto-linking too, not just startfiles). There's no
`SIMULATOR` choice on this target — QEMU linux-user mode fundamentally
cannot run a bare-metal semihosting binary (no ELF interpreter, no Linux
syscalls to emulate), so gem5 FS mode is the only supported runner.

New target file: `tensorflow/lite/micro/tools/make/targets/riscv64_baremetal_makefile.inc`.
Notable pieces:

- `RISCV_ARCH := rv64imc_zicsr` (not plain `rv64imc` — needed for
  `start_semi.S`'s CSR instructions).
- `LDFLAGS += -nostartfiles -nostdlib -T .../riscv64_baremetal/linker_semi.ld`.
- `MICROLITE_LIBS := -Wl,--start-group -lc -lm -lgcc -Wl,--end-group`
  (overrides the default `-lm`).
- `MICROLITE_CC_SRCS += .../riscv64_baremetal/start_semi.S` — this is the
  key trick that avoids any manual relink step or core-Makefile change:
  the crt0 gets compiled via the *existing* `$(CORE_OBJDIR)%.o: %.S`
  pattern rule (using this target's own `-march=rv64imc_zicsr` etc.,
  since `CCFLAGS` is global per-target) and archived straight into
  `libtensorflow-microlite.a`, which the standard `%_test` link rule
  already links against — no separate "prepend the startup object" step
  needed, unlike the ad hoc manual relink used while debugging above.
- `tensorflow/lite/micro/riscv64_baremetal/debug_log.cc` — copy of
  `riscv64_generic`'s (`vsnprintf_` + `fputs`), needed because
  `specialize_files.py` keys per-target overrides on the exact `TARGET`
  name; without it the build would fall back to the generic top-level
  `debug_log.cc` (a different, heavier libc-`vfprintf`-based
  implementation) instead of the one already verified against the
  `.sdata`/`.sbss` fix above.
- New `tensorflow/lite/micro/testing/test_with_gem5_fs.sh` — counterpart
  to `test_with_gem5.sh`, invoking `sim_config/gem5_riscv_baremetal_fs.py`
  (no `--cpu=` flag, since the FS config's `RiscvMinorCPU` is fixed) instead
  of the SE-mode config. `TEST_SCRIPT := ... test_with_gem5_fs.sh riscv64 minor`
  keeps the same positional-argument shape as the SE-mode `TEST_SCRIPT`
  definitions purely for visual/interface consistency; the `riscv64`/`minor`
  words are unused by the script itself.

### Verified working commands

```bash
source /home/ajno5/work/2_pattern/tflm/script/0_env_var_setup.sh
cd /home/ajno5/work/2_pattern/tflm/tflite-micro

make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal \
  TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test
```

Builds the entire `libtensorflow-microlite.a` from scratch for this target
(first build takes a few minutes — full kernel library, not just the one
test) and runs it under gem5 FS mode via `test_with_gem5_fs.sh`. Output:
`~~~ALL TESTS PASSED~~~`, `Pass`, clean semihosting exit at
`tick 672762000` (~0.67 ms simulated). `readelf -l` on the resulting
`gen/riscv64_baremetal_aarch64_default_gcc/bin/hello_world_test`
(147,232 bytes) confirms `R E`/`RW` segments — no RWX, no linker warnings,
`-Wl,--fatal-warnings` and `-Wl,--gc-sections` both satisfied. FLASH usage
is `0x15588` (~87.4 KB) of the 128 KB budget.

Also re-verified with a second, unrelated test (`test_micro_utils_test`) to
confirm the target works generally, not just for `hello_world_test` —
passed cleanly (`0.8 s` simulated wall time vs. `hello_world_test`'s
`1.77 s`, consistent with it being a much smaller test).

> **Correction, added later:** this specific check was itself a false
> positive — see "The `.init_array` bug" section further below.
> `test_micro_utils_test` uses TFLM's newer GTest-style `TEST()` macro, which
> turned out to silently register zero tests under this crt0 at the time
> (`0 tests ran` / `[PASSED] 0 tests` — a vacuous pass, not a real one). The
> crt0 bug has since been fixed and this test now genuinely passes 8 real
> tests; the "confirms the target works generally" claim above wasn't
> actually established until that fix.

No more manual relinking against `riscv64_generic`'s build artifacts is
needed — this target builds and links itself end-to-end through the normal
Makefile flow, same as any other TFLM target.

### `tflm_benchmark` on `riscv64_baremetal`, and two more linker-script fixes it surfaced

`hello_world_test` is TFLM's smallest possible binary (one `FULLY_CONNECTED`
op) — it never exercised whether `linker_semi.ld`'s memory budget or section
handling would hold up for anything bigger. Trying
`run_tflm_benchmark` with the `person_detect.tflite` model (13×
`DEPTHWISE_CONV_2D`, 14× `CONV_2D`, `AVERAGE_POOL_2D`, `RESHAPE`, `SOFTMAX` —
the same model used for the `riscv64_generic` SE-mode benchmark documented
above) hit two more issues:

1. **FLASH overflow.** `tflm_benchmark` links in the *entire* kernel library
   (every op TFLM ships, not just the ones a given model uses — the
   benchmark harness registers a generic op resolver) plus the model itself
   compiled in as a `.rodata` C array. First link attempt:
   `region 'FLASH' overflowed by 701600 bytes` — the 128 KB FLASH region
   sized for `hello_world_test` was never going to hold this. Fixed by
   bumping `linker_semi.ld`'s `MEMORY` block: `FLASH` `128 KB → 4 MB` (same
   `ORIGIN = 0x00010000`), `RAM` `2 MB → 4 MB` (moved to
   `ORIGIN = 0x00410000`, right after the now-larger FLASH). Both comfortably
   fit under gem5 FS config's `system.mem_ranges = [AddrRange("512MB")]`
   (`sim_config/gem5_riscv_baremetal_fs.py`), so no sim-config change was
   needed — only the linker script.
2. **`.bss` segment allocation error**, only after fixing FLASH: `section
   '.bss' can't be allocated in segment 1`. Same root cause as the
   `.heap`/`.stack` LMA-adjustment issue found while wiring up
   `hello_world_test` (see the `PHDRS`/RWX-segment section above) — .bss has
   no real ELF file content (it's zero-filled by `start.S`'s own zeroing
   loop at runtime, not the loader), but without `(NOLOAD)` the linker tries
   to give it a real LMA continuing from `.data`'s `AT>FLASH` address inside
   the shared `:ram` segment. That computation only broke once `.bss` grew
   large enough (`tflm_benchmark`'s `static uint8_t tensor_arena[153600]` —
   `hello_world_test` has no arena anywhere near that size, which is why this
   didn't surface earlier). Fixed by marking `.bss (NOLOAD)` too, matching
   `.heap`/`.stack`, and dropping its now-inapplicable `:ram` PHDR
   assignment.

With both fixes, `tflm_benchmark` builds and runs cleanly:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal \
  TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  BUILD_TYPE=default run_tflm_benchmark \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=153600
```

All 30 ops executed, arena usage `89,248 B` total (`61.96%` non-persistent /
`38.04%` persistent) — identical numbers to the `riscv64_generic` SE-mode
run of the same model documented above, a good sanity check that the
computation itself is unaffected by execution mode. gem5 exited cleanly at
`tick 422108725000` (~422 ms simulated) — notably faster simulated time
than the SE-mode run's `6.179 s`; no investigation was done into why (could
be FS mode's simpler system/cache config, could be something else), so no
conclusions should be drawn from that difference yet. Per-op timings are all
`0 ticks (0 ms)`, the same pre-existing profiling-clock gap noted for
`riscv{32,64}_generic` above.

## Whisper support on `riscv64_baremetal`: a second, faster simulator

gem5 FS mode (above) is cycle-accurate but slow. Since `riscv64_baremetal`
binaries do their I/O via RISC-V semihosting (`start_semi.S`'s
`SYS_WRITE0`/`SYS_EXIT`), and [whisper](https://github.com/chipsalliance/whisper)
— a functional-only RISC-V ISS, no timing model, already used by the
sibling `gemm` project — supports semihosting directly via `--semihosting`,
the exact same `riscv64_baremetal` binaries run under it unmodified. Wired
in as a second `SIMULATOR` choice, same pattern as `qemu`/`gem5` on
`riscv{32,64}_generic`:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal SIMULATOR=whisper \
  TARGET_TOOLCHAIN_ROOT=$HOME/work/1_toolchain/xpack/xpack-riscv-none-elf-gcc-13.2.0-2/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
  test_hello_world_test
```

`~~~ALL TESTS PASSED~~~`, `Pass`, identical arena stats to the gem5 run.
449,784 instructions executed in `0.04s` wall-clock (~11.8M inst/s) —
whisper has no cycle-accurate timing model, so this isn't directly
comparable to gem5's simulated-tick numbers above, just fast to iterate
with.

New files: `tensorflow/lite/micro/testing/test_with_whisper.sh` (mirrors
`test_with_gem5_fs.sh`'s interface — same positional args, same pass-string
grep logic — just invokes `whisper --configfile <config> --semihosting
--counters --target <binary>` instead of `gem5.opt`) and
`sim_config/whisper_rv64gcv_config.json` (outside the repo, alongside the
gem5 configs, same reasoning). `riscv64_baremetal_makefile.inc` gained a
`SIMULATOR ?= gem5` switch (whisper is opt-in; default behavior/existing
verified gem5 runs are unaffected).

### Why the whisper config declares vector/float support the binary doesn't use

The straightforward move would've been a trimmed `{"isa": "rv64imc", ...}`
config matching `riscv64_baremetal`'s actual `-march=rv64imc_zicsr` build —
and that does work, but produces a warning on every run:
`Bit 21 (v) is set in the MISA register but the d/f extensions are not
enabled -- ignored`. Cause: `start_semi.S` (inherited from `gemm`'s crt0,
written for its vector-capable `rv64gcv` target) unconditionally executes
`csrs mstatus, 0x2200` at boot to enable `mstatus.FS`/`mstatus.VS`,
regardless of what the target ISA actually supports. gem5 never surfaced
this because its FS config always sets `RiscvISA(vlen=512, elen=64)`
independent of the compiled binary's extensions; whisper's `rv64imc` config
correctly has no vector support at all, so it notices the mismatch.

Per instruction, `sim_config/whisper_rv64gcv_config.json` is instead an
exact copy of gemm's own config — full `rv64imafdcv_zfh_zvfh_...` ISA,
VLEN=512/ELEN=64, HPM counters for `Fp`/`FpDouble`/`Vector`/`VectorLoad`/
`VectorStore` — even though the current `riscv64_baremetal` target doesn't
compile with `-march=rv64gcv` and so never exercises any of that. Confirmed
this superset config runs the existing `rv64imc_zicsr` binary cleanly (no
warning, since the ISA now actually has the bits `start_semi.S` is trying
to enable) and, more importantly, is ready as-is for whenever a vectorized
TFLM kernel build exists to point at it — no config rework needed later,
just a target/`-march=` change on the TFLM build side.

## The `.init_array` bug: silent vacuous test passes on `riscv64_baremetal`

Picked `dtln_noise_suppression` (a real, larger model — `M=1, K=128, N=257`
`FULLY_CONNECTED` layer, 364 KB, LSTM + FC) as a matrix-optimization
benchmark candidate and built its dedicated `dtln_test` target
(`tensorflow/lite/micro/examples/dtln/dtln_test.cc`). First run: built and
executed cleanly under gem5 FS mode, printed `~~~ALL TESTS PASSED~~~`,
exit code 0 — but the actual test log read:

```
[==========] Running tests.
[==========] 0 tests ran.
[  PASSED  ] 0 tests.
```

Zero tests ran, yet it "passed" — a vacuous pass (0 failures out of 0
tests is trivially true), not a real one. `dtln_test.cc` uses
`testing/micro_test_v2.h`'s GTest-style `TEST(suite, name)` macro, unlike
`hello_world_test`'s older `TF_LITE_MICRO_TEST` macro. Checking
`micro_test_v2.h` confirmed why: `TEST(...)` expands to a
`static micro_test::internal::TestRegistrar suite_name_Reg(&info);` at
namespace scope — a static C++ object whose **non-trivial constructor**
(`TestRegistrar(TestInfo*) { TestRunner::Get().RegisterTest(info); }`)
registers the test into a global list. Static objects with non-trivial
constructors are initialized by the C++ runtime *before* `main()` runs, via
the `.init_array`/`.preinit_array` mechanism — and `start_semi.S` never
called it. It went straight from the `.data` copy loop to `call main`.
`linker_semi.ld` already defined the `__{preinit,init}_array_{start,end}`
symbols (needed regardless, so `.init_array`/`.preinit_array` content
isn't silently dropped by the linker) — nothing was ever walking them at
runtime.

Net effect: every `TEST()`-based test on `riscv64_baremetal` was silently
running zero of its tests and reporting success the entire time this
target has existed, including `test_micro_utils_test`, which had been used
earlier as the "second test, not just `hello_world_test`" sanity check
when this target was first wired into the Makefile (see the correction
inline in the "Verified working commands" section above) — that check
never actually verified anything beyond "the binary doesn't crash."
`hello_world_test` itself was never affected, since its `TF_LITE_MICRO_TEST`
macro (the older style) doesn't rely on static-constructor registration.

Fix: `start_semi.S` now walks `__preinit_array_start`..`__preinit_array_end`
and `__init_array_start`..`__init_array_end`, calling each function pointer
before `call main` — standard freestanding-crt0 boilerplate that should
have been there from the start (copied from `gemm`'s crt0, which likely
never hit this because gemm's own kernels don't use static-constructor-based
registration anywhere).

Re-verified after the fix, no regressions:

| Test | Simulator | Result |
|---|---|---|
| `dtln_test` | gem5 | Now genuinely runs: `[ RUN ] DtlnTest.TestInvoke` → all `EXPECT_EQ` assertions (input/output shape, `Invoke()` status, full 257-element golden-reference comparison) pass → `[ OK ]` → `1 tests ran` / `[PASSED] 1 tests`. `tick 5996072000` (~6.0 ms simulated) — the biggest/slowest `riscv64_baremetal` test run so far, consistent with it being the largest model. |
| `dtln_test` | whisper | Same genuine pass (`[ RUN ]` → `Ran successfully` → `[ OK ]` → `1 tests ran` / `PASSED`). `4,711,964` instructions in `0.38s` wall-clock (~12.5M inst/s) — no cycle-accurate timing model, so not directly comparable to gem5's tick count, but the fastest way to iterate on this test. |
| `micro_utils_test` | gem5 | Also `TEST()`-based, also silently vacuous before this fix. Now genuinely runs `8 tests ran` / `[PASSED] 8 tests`. |
| `hello_world_test` | gem5 + whisper | Unaffected either way (older macro style) — re-confirmed identical output/arena stats after the fix. |
| `tflm_benchmark` (`person_detect.tflite`) | gem5 | Unaffected — identical `89,248 B` arena, same 30 ops, `tick 429485090000` (~429 ms, matches the ~422 ms from the pre-fix run within normal variance). |

## gem5 SE mode disabled

With `riscv64_baremetal` + `SIMULATOR=whisper` now verified (`hello_world_test`
and `dtln_test`, both real passes — see above), the "fast, no-timing-model
functional simulator" role is covered by whisper, and the "cycle-accurate"
role is covered by gem5 FS mode (`riscv64_baremetal`'s default) — leaving
gem5 SE mode on `riscv{32,64}_generic` without a distinct purpose it alone
serves. Disabled rather than deleted: `riscv{32,64}_generic_makefile.inc`'s
`SIMULATOR=gem5` branch now trips a `$(error ...)` with a message pointing
at the replacement, instead of silently falling back to `qemu` (which could
mask a stale `SIMULATOR=gem5` left in a command line/script) or being
removed outright (which would lose the working, previously-debugged
implementation — the two non-obvious gem5 API fixes documented above,
`RiscvISA` word-size matching and the explicit `SEWorkload`, took real
effort to find).

```
$ make ... TARGET=riscv64_generic SIMULATOR=gem5 test_hello_world_test
tools/make/targets/riscv64_generic_makefile.inc:70: *** SIMULATOR=gem5 (SE mode) is
disabled for riscv64_generic — use TARGET=riscv64_baremetal with SIMULATOR=whisper
(fast, functional) or the default SIMULATOR=gem5 there (cycle-accurate FS mode)
instead.  Stop.
```

`SIMULATOR=qemu` (the default, unchanged) still works exactly as before —
confirmed via `test_hello_world_test` on `riscv64_generic` with no
`SIMULATOR` override. (Separately, `qemu-riscv64` itself isn't currently
installed/on `PATH` on this host — a pre-existing environment gap, not
something this change touched or caused; `qemu-riscv32` was the only QEMU
path previously confirmed actually running end-to-end in this environment,
per "Verified working commands" above.)

Nothing was deleted: `tensorflow/lite/micro/testing/test_with_gem5.sh` and
`sim_config/gem5_riscv_se.py` are both still present, just unreferenced by
any Makefile target now. `riscv64_baremetal` (gem5 FS mode + whisper) is
untouched by this change.

## Per-op cycle counts on `riscv64_baremetal`: the two gaps that made every op read "0 ticks"

Wanted real per-layer cycle counts for `dtln_noise_suppression.tflite`.
Two independent gaps, both needed fixing:

1. **`dtln_test.cc` never wires a profiler.** `MicroInterpreter`'s
   constructor takes an optional `MicroProfilerInterface* profiler =
   nullptr`; `dtln_test.cc` doesn't pass one, so every op's
   `ScopedMicroProfiler` (in `micro_interpreter_graph.cc`, wrapping every
   `registration->invoke()` call) is constructed with a null profiler
   pointer and no-ops entirely — zero profiling events, regardless of
   simulator. Worked around without touching `dtln_test.cc` at all: ran the
   model through `run_tflm_benchmark` instead
   (`GENERIC_BENCHMARK_MODEL_PATH=.../dtln_noise_suppression.tflite`) — that
   harness already constructs a real `MicroProfiler` and passes it in (the
   same mechanism that produced the `person_detect` per-op table earlier).
   `run_tflm_benchmark`'s wall-clock cost tracks the model's actual compute,
   not the harness itself — `dtln`'s single FC + 2×LSTM calls finished in
   single-digit seconds under gem5, nothing like `person_detect`'s 7–9
   minutes.
2. **The clock source was hardcoded to 0.** `tensorflow/lite/micro/micro_time.cc`'s
   reference implementation (used by every target without its own override)
   returns `0` from both `ticks_per_second()` and `GetCurrentTimeTicks()` —
   exactly why `person_detect`'s per-op table earlier always printed
   `0 ticks (0 ms)` even with a profiler wired. Fixed by adding
   `tensorflow/lite/micro/riscv64_baremetal/micro_time.cc`, reading the
   `mcycle` CSR (M-mode cycle counter, 0xB00) via inline `csrr`.

`mcycle`, not `cycle`/`rdcycle`: first attempt used the `rdcycle`
pseudo-instruction (targets the *unprivileged* shadow CSR `cycle`, 0xC00).
Worked under gem5, but whisper trapped it as illegal
(`mcause=0x2`, `mtval=0xc0002573` — decodes to exactly this `csrrs`
encoding) unless the ISA string explicitly declares `Zicntr`, which
`sim_config/whisper_rv64gcv_config.json` doesn't. Rather than patch that
shared config (used identically by the sibling `gemm` project, per
instruction to keep them matching), switched to reading `mcycle` directly —
this crt0 runs entirely in M-mode, where `mcycle` is always accessible
regardless of declared ISA extensions, and it's exactly what `gemm`'s own
kernels (`src/gemm.c`, `src/dgemm.c`, etc.) already do for the same reason.

Result, `dtln_noise_suppression.tflite` via `run_tflm_benchmark`
(`GENERIC_BENCHMARK_ARENA_SIZE=16384`, matching `dtln_test.cc`'s own arena
size):

| Op | gem5 cycles | whisper cycles |
|---|---|---|
| `UNIDIRECTIONAL_SEQUENCE_LSTM` (1st call) | 2,685,618 | 2,479,287 |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` (2nd call) | 1,845,791 | 1,688,145 |
| `FULLY_CONNECTED` | 378,379 | 311,697 |
| `LOGISTIC` | 89,537 | 88,405 |
| **Total (profiled ops only)** | **4,999,325** | **4,567,534** |

Output CRC32 (`0x7E578D1C`) identical between simulators — same computation,
just different cycle counts, as expected: gem5's `RiscvMinorCPU` models
real pipeline stalls/hazards; whisper is a functional simulator with no
timing model of its own, so its per-instruction cycle attribution is closer
to an idealized IPC assumption. Don't treat whisper's numbers as
cycle-accurate — gem5's are the trustworthy ones for actual performance
comparisons; whisper's are useful for fast *relative* op-to-op comparison
and quick iteration, not absolute cycle counts.

The LSTM dominates by a wide margin (4.17–4.53M of ~4.57–5.0M total profiled
cycles, i.e. ~91% either way) — the `FULLY_CONNECTED` layer this benchmark
target was originally picked for is comparatively cheap. Worth keeping in
mind if/when comparing a vectorized `FULLY_CONNECTED` kernel against this
baseline: the LSTM, not the FC layer, is this model's actual bottleneck.

## A vectorized `FULLY_CONNECTED` kernel: `riscv64_baremetal_vector`

Built a real RVV-vectorized replacement for the int8-quantized
`FullyConnected()` reduction and compared it against the scalar baseline
above (378,379 gem5 cycles / 311,697 whisper cycles).

### The math, and why it needed a genuinely new target

`dtln`'s FC layer's quantization params (pulled from the model's flatbuffer):
input zero-point `-4` (so `input_offset = +4`, **not** zero — asymmetric
activation quantization), filter zero-point `0` with a single scale
(per-tensor, so `is_per_channel` is false — the plain `FullyConnected()`
template runs, not `FullyConnectedPerChannel()`). The reference scalar loop
computes, per output channel:

```
acc = Σ_d (filter[d] + filter_offset) * (input[d] + input_offset)
```

Since `input_offset != 0`, this isn't a pure int8×int8 dot product as-is.
Rather than the usual gemmlowp-style trick (precompute a per-row filter
sum once, correct for the offset afterward), widening *both* operands to
int16 first — via a single `vwadd.vx` each, folding `filter_offset`/
`input_offset` into the widen itself — keeps the two approaches
term-by-term identical to the reference formula (integer add/multiply is
exactly associative, so there's no reordering-sensitive precision loss
the way float accumulation would have). Then `int16×int16→int32` widening
multiply (`vwmul.vv`) and a widening-free `int32` reduction
(`vredsum.vs`). Verified the toolchain (xpack GCC 13.2.0) accepts this
against `-march=rv64imc_zicsr_zve64x` (`Zve64x`: integer-only embedded
vector profile, `ELEN=64` — matches `elen=64` in both
`sim_config/gem5_riscv_baremetal_fs.py` and
`sim_config/whisper_rv64gcv_config.json`, so **no sim-config changes were
needed** — they were already vector-capable, per the earlier section on
why the whisper config declares extensions the (then-)current binary
didn't use) via a standalone probe file before touching the real kernel.

New target: `riscv64_baremetal_vector` (own `TARGET_ARCH`/`makefile.inc`/
`riscv64_baremetal_vector/{start_semi.S,linker_semi.ld,debug_log.cc,
micro_time.cc}` — copies of the plain `riscv64_baremetal` ones, unchanged).
Deliberately a **separate target**, not a `RISCV_ARCH` override on the
existing one: TFLM's `GENDIR` path
(`gen/<TARGET>_<host>_<build_type>_<toolchain>/`) is keyed on `TARGET`
name, not on `RISCV_ARCH` — overriding the arch on the same target would
risk silently reusing stale, wrong-ISA `.o` files from a previous build
without forcing a rebuild.

### The fast path itself

Added `Int8DotProductRvv()` and a `#if defined(__riscv_vector)`-gated call
site inside `tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h`'s
plain `FullyConnected()` template — inert (compiles to nothing extra) on
every target that doesn't define `__riscv_vector`, active only on
`riscv64_baremetal_vector`. Modifying this shared reference header (rather
than inventing new build-system plumbing to swap in a per-target kernel
file, the way `xtensa`/`cmsis_nn`/etc. do) was the simplest available
mechanism: there's no existing "optimized kernel" override slot for
`FullyConnected` the way there is for `debug_log.cc`/`micro_time.cc`, and
building one would have been a much bigger, unrelated undertaking.

### Bug found and fixed: the `if constexpr` guard didn't check `OutputType`

First build compiled clean and ran — `FULLY_CONNECTED` dropped from
378,379 to 79,742 gem5 cycles, and (unexpectedly) `UNIDIRECTIONAL_SEQUENCE_LSTM`
*also* sped up ~4.9×. Investigating that: `kernels/lstm_eval.cc` calls this
exact same `reference_integer_ops::FullyConnected()` template internally
for its gate matmuls (`int8` input/filter, `int32` bias — same as the
top-level `FullyConnected` op's dtln usage — but `int16_t` **output**,
since gate pre-activations need more precision than int8 before going
through the sigmoid/tanh nonlinearities). The `if constexpr` guard checked
`InputType`/`WeightType`/`BiasType` but not `OutputType`, so the fast path
silently applied there too. And **Output CRC32 changed**
(`0x50433D2B` vs. the correct `0x7E578D1C`) — a real correctness
regression, not just a missed-optimization gap.

Restricting the guard to also require `OutputType == int8_t` initially
appeared to have *no effect at all* — identical cycle counts, identical
wrong CRC32 — because **this build has zero `.d` dependency files
anywhere** (confirmed via `find ... -iname "*.d"`, zero results): the
Makefile has no automatic header-dependency generation
(no `-MMD`/`-MD`), so `make` has no way to know any `.o` file depends on a
transitively-`#include`d header, and never recompiles on a header-only
change — only when the directly-listed `.cc` source itself changes. Had to
manually `rm` the specific stale objects
(`fully_connected.o`, `fully_connected_common.o`, `lstm_eval.o`,
`lstm_eval_common.o`) plus the archived `.a` and the linked binary before
the guard fix actually took effect. **This is a real, generally-applicable
gap in this build system worth remembering**: editing a header used by
multiple `.cc` files needs a manual forced rebuild (delete the affected
`.o`s, or `make clean`) — `make` alone will silently keep linking stale
object code.

With the `OutputType == int8_t` restriction actually compiled in: Output
CRC32 back to `0x7E578D1C` (matches baseline exactly — correctness
confirmed), LSTM back to near-baseline (no longer vectorized, as intended —
its `int16_t`-output overload wasn't validated safe and wasn't the target
of this exercise anyway), `FULLY_CONNECTED` still fast.

### Results

| | gem5 (cycle-accurate `RiscvMinorCPU`) | whisper (functional, no timing model) |
|---|---|---|
| Baseline `FULLY_CONNECTED` | 378,379 cycles | 311,697 cycles |
| Vectorized `FULLY_CONNECTED` | 79,786 cycles | 25,477 cycles |
| **Speedup** | **4.74×** | **12.2×** |

Output CRC32 (`0x7E578D1C`) identical to the scalar baseline in both cases —
same computation, verified correct. Whisper's speedup number is
substantially larger than gem5's — expected, given whisper has no
cycle-accurate memory/pipeline model, so it can't capture the real cost of
the vector loads/stores the way `RiscvMinorCPU` does; **gem5's 4.74× is
the trustworthy figure for actual hardware-relevant comparison**, whisper's
12.2× shouldn't be read as a real-world expectation.

Because the LSTM dominates total model cycles (~91%, see the per-op cycle
count section above) and wasn't accelerated here, the *whole-model*
speedup is much smaller than the FC-layer speedup alone: gem5 total ticks
across profiled ops go from 4,999,325 to 4,369,141 (~12.6% faster overall),
whisper from 4,567,534 to 4,281,314 (~6.3% faster overall) — worth keeping
in mind when characterizing "the win" from this work: it's real and large
for the layer it targets, modest for this particular model end-to-end.

### Known limitations of this specific kernel

- Only handles the plain (non-per-channel) int8 `FullyConnected()`
  overload with `int8_t` output — deliberately, per the bug above. The
  per-channel variant (`FullyConnectedPerChannel`), the int16-activation
  variant, and float32 are all untouched, still scalar.
- `filter_offset`/row-sum-equivalent correction is folded into the
  `vwadd.vx` widen and recomputed fresh on every `Invoke()` call — a
  production kernel would instead precompute anything filter-derived once
  at `Prepare()` time (the filter doesn't change between invocations) and
  cache it. Not done here, since it wasn't necessary to get a valid,
  correct comparison — just something a production version would want.
- Only exercised against `dtln_noise_suppression.tflite`'s specific FC
  shape (`M=1, K=128, N=257`). Not verified against other FC shapes/models
  (e.g. `micro_speech`'s `K=4000, N=4`).

### LSTM vectorization: a real bug found and fixed in `Int8DotProductRvv`, unblocked by upgrading GCC

Followed up on whether `UNIDIRECTIONAL_SEQUENCE_LSTM`'s gate matmuls
(excluded above by the `OutputType == int8_t` guard) could also take the
RVV fast path — they're the same `[gate_size,K] x [K]` int8 GEMV shape as
FC, just `int16_t` output (higher precision needed before the
sigmoid/tanh gates). Relaxing the guard to drop the `OutputType` check
reproduced the exact known-bad CRC32 (`0x50433D2B`) documented above —
confirming this was never just an overly conservative type check. There's
a real numeric bug.

**Root cause, precisely identified.** Added a temporary diagnostic (both
scalar and vector accumulation computed on every call, comparing them,
using the scalar result so model output stays correct while logging the
first divergence) and rebuilt `test_dtln_test` under whisper:

```
RVV_DOT_MISMATCH accum_depth=257 input_offset=128 filter_offset=0 scalar_acc=-3023 vec_acc=144177
```

`input_offset=128` is exactly one past `int8_t`'s range (`-128..127`).
`Int8DotProductRvv` folds `input_offset`/`filter_offset` into a widening-add
(`vwadd.vx`) whose scalar operand is SEW(=8)-bit-wide — `128` silently
wraps to `-128` there, corrupting the whole accumulation. `128` is a
legitimate value (`input_offset = -zero_point`, and `zero_point=-128` is a
valid `int8_t` code) — FC's own zero-point (`+4`) never comes close to this
edge, so the kernel has looked correct for FC this whole time. This is a
**latent bug in the original vectorized FC kernel itself**, not an
LSTM-specific issue — it would break FC too, for any model whose FC layer
happened to have `zero_point=-128`.

**The fix, algebraically:** expand
`(filter[d]+filter_offset)*(input[d]+input_offset)` into
`Sigma(filter*input) + input_offset*Sigma(filter) + filter_offset*Sigma(input)
+ K*filter_offset*input_offset` (the standard gemmlowp-style trick the
original kernel comment explicitly said it was avoiding for simplicity —
turns out to be the textbook-correct approach for exactly this reason).
Offsets are then only ever applied via plain `int32_t` scalar arithmetic
*after* vector reduction, never passed to a narrow vector intrinsic — so no
truncation is possible regardless of offset value. Both widens
(`vwadd.vx(va, 0, vl)`) use offset `0`, which can never overflow `int8_t`.

**Validated correct at `-O0`, but initially blocked by a GCC 13.2 codegen
bug at `-O2`.** Built a standalone probe (linked against
`riscv64_baremetal_vector`'s own `start_semi.S`/`linker_semi.ld`, run under
whisper for fast iteration) comparing this expansion against the scalar
reference across `accum_depth` 1–513 (including the exact `K=257` shape)
and offset pairs including the failing `±128` case: **114/114 correct at
`-O0`.** But applying the identical fix to the real kernel — which compiles
at `-O2` (`KERNEL_OPTIMIZATION_LEVEL := -O2`) — failed two different ways
depending on simulator, under the project's toolchain at the time (xPack
GCC 13.2.0):
- **whisper**: illegal-instruction trap (`mcause=2`), the exact same trap
  the standalone probe hit at `-O1`/`-O2` (only `-O0` was clean there too).
- **gem5**: no trap, but a *third*, different wrong CRC32 — the classic
  signature of miscompiled code producing plausible-looking garbage rather
  than crashing outright.

Tried scoping just `Int8DotProductRvv` to `__attribute__((optimize("O0")))`
so the rest of the translation unit keeps `-O2`. This fixed whisper but
**broke gem5 a third way**: `src/mem/xbar.cc:368: fatal: Unable to find
destination for [...] on system.membus` — a fatal invalid memory access.
Likely an ABI/register-save mismatch at the `-O0`/`-O2` optimization
boundary, rather than anything about the dot-product logic itself, which
was already proven correct in isolation.

**Resolved: upgrading the toolchain fixed the codegen bug.** Checked xPack's
`riscv-none-elf-gcc` releases — the project had been on `v13.2.0-2`
(2023-09), and multiple newer releases existed, including `v13.4.0-1`
(2025-10, same GCC 13.x major branch, patch-level fixes only — lowest risk
of unrelated regressions elsewhere in the project). Downloaded it
(`xpack-riscv-none-elf-gcc-13.4.0-1-linux-arm64.tar.gz`, same parent dir as
the existing toolchain: `1_toolchain/xpack/`) and re-ran the standalone
probe at `-O2`: **114/114 correct, no trap.** Rebuilt the real kernel with
both fixes (the algebraic expansion, plus dropping `OutputType` from the
`if constexpr` guard so LSTM's `int16_t`-output gate matmuls take the fast
path) against GCC 13.4.0-1 — clean on both simulators, no traps, no fatal
errors, `Output CRC32: 0x7E578D1C` (correct) on both.

**Cross-shape validation** (scalar-target CRC32 vs. vector-target-with-fix
CRC32, must match exactly): `dtln_noise_suppression` (FC `K=128,N=257`;
LSTM `K=257/128→128`, single timestep/call), `mnist_lstm`'s
`trained_lstm_int8.tflite` (LSTM `K=28/20→20`, genuinely sequential
**28-timestep** processing per call — a different model entirely, not just
a different shape), `micro_speech_quantized` (FC `K=4000,N=4`, extreme
deep-K/narrow-N), `hello_world_int8` (FC `K=16,N=16`, tiny). All four
matched exactly, on whisper; `dtln_noise_suppression` and `mnist_lstm` also
re-verified on gem5. No mismatches, no crashes, anywhere.

**Results** (`dtln_noise_suppression`, `riscv64_baremetal_vector`, GCC
13.4.0-1):

| | gem5 (cycle-accurate) | whisper (functional) |
|---|---|---|
| `UNIDIRECTIONAL_SEQUENCE_LSTM` (1st) | 2,685,618 → **621,580** (4.32×) | 2,479,287 → **221,910** (11.17×) |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` (2nd) | 1,845,791 → **435,055** (4.24×) | 1,688,145 → **183,513** (9.20×) |
| `FULLY_CONNECTED` | 378,379 → **87,788** (4.31×) | 311,697 → **30,873** (10.09×) |
| **Total (profiled ops)** | 4,999,325 → **1,233,600** | 4,567,534 → **524,701** |
| **Whole-model speedup** | **~4.05×** | **~8.71×** |

Unlike the FC-only fix (which only bought ~12.6% whole-model speedup, since
the LSTM — ~91% of total cycles — was untouched), vectorizing the LSTM
gates too gets the whole-model win the roofline analysis predicted was
available: gem5's ~4.05× is the trustworthy figure (whisper's larger
speedup is inflated the same way the FC-only whisper number was — no
cycle-accurate memory/pipeline model).

**Project toolchain default switched to GCC 13.4.0-1** (`script/
0_env_var_setup.sh`, `script/1_run_pattern.sh`, `script/2_run_benchmark.sh`,
and `performance.md`/`dtln/performance_dtln.md`'s `Reproducing` commands — the
`TARGET_TOOLCHAIN_ROOT=...13.2.0-2...` examples earlier in *this* doc are
untouched, since they're historical record of what was actually run for
those earlier, unrelated fixes, and remain reproducible either way since
both toolchain versions stay installed). The old 13.2.0-2 install and its
tarball are left in place alongside the new one (`1_toolchain/xpack/`, both
versions present) rather than removed, in case anything else in the
project turns out to depend on the exact older version.

### `Int8DotProductRvv`: widened from base `LMUL=1` to `LMUL=2`

Follow-up after the correctness fix above: with `Int8DotProductRvv`
correct, checked whether wider `LMUL` closes any of the gap to the
roofline's peak-compute ceiling (`dtln/performance_dtln.md`'s "Achieved
performance" table still showed only ~2.3-2.9% efficiency post-fix).
`e8m2 -> e16m4 -> e32m8` is the widest feasible base LMUL for this
double-widening chain — a third step (base `m4`) would need `m16` for the
second widen, which doesn't exist (RVV's max `LMUL` is 8).

Measured on gem5 across every `accum_depth` this project's models actually
use: **25-34% fewer cycles for `accum_depth >= 128`** (dtln's FC/LSTM
shapes: 128, 257; `micro_speech`'s FC: 4000) since a wider load/widen/
multiply processes more elements per instruction, directly cutting loop
iteration count; **~4% *more* cycles for `accum_depth <= 28`**
(`mnist_lstm`'s LSTM gates: 20, 28; `hello_world`'s FC: 16) — those already
complete in a single vector op at `LMUL=1` (`VLMAX=64` for `e8` at
`VLEN=512`), so `LMUL=2` there only adds fixed per-instruction overhead
with no iteration-count reduction to offset it. Applied unconditionally
(no runtime branch on `accum_depth`) since dtln — the standing benchmark —
is entirely `accum_depth in {128, 257}`, both net wins; the small
regression on the other models' already-cheap small-K gates was accepted
rather than adding branch complexity to the hot path. Re-validated against
all 4 models from the correctness-fix validation above (dtln, `mnist_lstm`,
`micro_speech_quantized`, `hello_world_int8`) — all CRC32s still match.
See `dtln/performance_dtln.md` for the resulting cycle counts.

Also tried accumulating the widening-multiply's result in a *vector*
register across loop iterations (instead of a full horizontal `vredsum`
every iteration) and reducing only once at the end — the more
theoretically appealing fix for the "redundant reduces" pattern. Measured
**worse** for every shape `<=257` (32-84% *more* cycles) and only better
for `accum_depth=4000` (17% fewer) — the accumulator version pays fixed
setup cost (an extra `vsetvl` plus three `vmv.v.x` zero-inits) on every
call regardless of iteration count, and none of this project's actual
shapes have enough iterations to amortize that against the saved reduces.
Not applied.

### Why efficiency stays low even after vectorization: `MinorCPU`'s single shared `FloatSimd` functional unit

Investigated why the roofline's "achieved performance" numbers stay in the
low single-digit percent of the peak-compute ceiling even with a correct,
`LMUL`-widened vectorized kernel. Root cause is in gem5's default
`MinorCPU` functional-unit pool
(`src/cpu/minor/BaseMinorCPU.py:172-294`, unmodified by
`sim_config/gem5_riscv_baremetal_fs.py` — it just instantiates
`RiscvMinorCPU()` with no FU pool override):

- `MinorDefaultFloatSimdFU` covers essentially every SIMD/vector op class,
  including `SimdAdd`/`SimdMult` (our `vwadd`/`vwmul`) and `SimdReduceAdd`
  (our `vredsum`/`vwredsum`) — and there is exactly **one instance** of it
  in `MinorDefaultFUPool`, vs. two `MinorDefaultIntFU`s for plain scalar
  ops. Every vector instruction in the kernel's inner loop competes for
  this single unit.
- It has `opLat = 6` (6-cycle result latency) but the default
  `issueLat = 1` (can *accept* a new instruction every cycle) — so it's
  not simply serialized at 1-per-6-cycles; it's pipelined for throughput.
- The actual bottleneck is that `MinorCPU` is a **simple in-order
  pipeline** with no out-of-order execution to hide dependency latency.
  The kernel's loop body is a strict chain — `load -> widen ->
  widen-multiply -> reduce` — 3-4 deep, and each step must wait out the
  full 6-cycle latency of the one before it before it can even issue. That
  compounds to ~18-24+ cycles of pure latency per iteration, even though
  the FU itself could theoretically accept unrelated work every cycle.
  `vredsum`/`vwredsum` aren't intrinsically slower than any other vector
  op here — they just sit at the end of the chain, so their latency is
  what the scalar accumulation (`dot +=` etc.) visibly waits on.
- Confirmed `RiscvMinorCPU()`'s defaults are **dual-issue**
  (`decodeInputWidth`/`executeInputWidth`/`executeIssueLimit`/
  `executeCommitLimit` all default to `2`,
  `src/cpu/minor/BaseMinorCPU.py:359-386`) — but that dual-issue front end
  is mostly wasted on vector-heavy code like this kernel, since at most 1
  of the 2 instructions issued per cycle can be *any* SIMD/vector op
  (loads/stores go through the separate `MemFU`), with only one FU to
  issue it into.

**Experiment: does a second `FloatSimd` FU help?** The sibling `softmax`
project already has exactly this diagnostic knob built (unused there too —
checked its `run_log.txt`/`doc/softmax_analysis.md`, every recorded run
used the default 1 FU) in
[`sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py`](../sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py)
(copied here from `/home/ajno5/work/2_pattern/softmax_cpp/sim_config/` —
otherwise identical board to `gem5_riscv_baremetal_fs.py`): a
`CustomMinorFUPool` reading `MINOR_FLOAT_FU_COUNT` from the environment
(default `1`) to instantiate that many `MinorDefaultFloatSimdFU` copies.
Ran the `LMUL` probe from above against it with `MINOR_FLOAT_FU_COUNT=2`:

| `accum_depth` | 1 FU, `LMUL=1` cycles | 1 FU, `LMUL=2` cycles | 2 FU, `LMUL=1` cycles | 2 FU, `LMUL=2` cycles |
|---|---|---|---|---|
| 16 | 56 | 58 | 56 (0%) | 58 (0%) |
| 20 | 58 | 60 | 56 (3.4%) | 58 (3.3%) |
| 28 | 58 | 60 | 56 (3.4%) | 58 (3.3%) |
| 128 | 101 | 80 | 87 (**13.9%**) | 67 (**16.3%**) |
| 257 | 273 | 217 | 244 (**10.6%**) | 187 (**13.8%**) |
| 4000 | 2436 | 1812 | 1997 (**18.0%**) | 1439 (**20.6%**) |

A second FU gives another real, meaningful speedup that **compounds with**
`LMUL=2` rather than substituting for it — same shape-dependent pattern
(tiny `K` has too little independent work to spread across 2 FUs; larger
`K` benefits more as there's more to overlap). Correctness re-verified
(`m1_match`/`m2_match` both `1` for every depth, both FU counts).

**Not applied to the project's default board.** Unlike `LMUL`, this is a
simulated-hardware-model decision, not a kernel optimization — it changes
what CPU we're claiming to benchmark against (2x the vector execution
resources), not how well our code uses the CPU we already modeled.
`sim_config/gem5_riscv_baremetal_fs.py` (the default board every other
number in this project is measured against) is untouched;
`gem5_riscv_demo_riscv_baremetal_semihost_minor.py` stays available as an
opt-in diagnostic config, kept purely as a documented experiment. No new
`TARGET` is needed to use it — `TARGET` governs the compiled binary
(`-march=`/linker script/`GENDIR`), which this experiment doesn't touch at
all; only `test_with_gem5_fs.sh`'s `GEM5_FS_CONFIG` env var (independent
of `TARGET`, defaults to `gem5_riscv_baremetal_fs.py`) needs overriding to
point at this file, against the exact same `riscv64_baremetal_vector`
binaries already built:

```bash
GEM5_FS_CONFIG=/home/ajno5/work/2_pattern/tflm/sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py \
MINOR_FLOAT_FU_COUNT=2 \
tensorflow/lite/micro/testing/test_with_gem5_fs.sh riscv64 minor \
  gen/riscv64_baremetal_vector_aarch64_default_gcc/bin/tflm_benchmark \
  non_test_binary riscv64_baremetal_vector
```

### Measuring the real ceiling: `microbenchmark/int8dot_ceiling.c`

The `25.6 GFLOP/s` memory-bound figure the roofline analysis originally
used as "the ceiling" was itself wrong — not imprecise, the wrong *kind*
of bound for this CPU. `AI × peak_BW` only holds as an upper bound if the
core can actually saturate DRAM; `MinorCPU`'s single-`FloatSimd`-FU
bottleneck above means it structurally can't, for this instruction mix.
Rather than derive a corrected ceiling analytically, measured it directly
— same "measure, don't assume" methodology the sibling `gemm` project
uses for its own `fmacc.c` compute roof. (Build settings, test config, and
current results for both probes below are collected in
[`microbenchmark/README.md`](microbenchmark/README.md) — this section is
the "how we got here" story, that one's the reusable reference.)

**[`microbenchmark/int8dot_ceiling.c`](../patterns/microbenchmark/int8dot_ceiling.c)**
replicates `Int8DotProductRvv`'s actual instruction sequence (`vle8` →
`vwadd` ×2 → `vwmul` → `vredsum`, `LMUL=2`) across `N` independent chains,
hand-interleaved (all loads, then all widens, then all multiplies, then
all reduces) so the in-order pipeline has genuinely independent work to
issue while any one chain's 6-cycle latency is in flight.

**First attempt (8 chains, mirroring `gemm`'s `fmacc.c` 8-way unroll)
measured a physically impossible >800 GFLOP/s** — the compiler had
recognized every iteration reads identical unchanging buffers and
hoisted/CSE'd the entire computation out of the loop. Fixed by perturbing
one byte per chain each iteration with an optimizer-opaque value (a
lighter-weight fix than a blanket memory-clobber barrier, which was tried
first and found to add its own spill overhead).

**With that fixed, 8 chains still measured far worse than expected
(~1360 cycles/iteration) — register pressure, not FU latency, turned out
to be the actual bottleneck in that version.** At this kernel's `LMUL=2`
base, the widest intermediate type is `vint32m8_t` (8 physical vector
registers). With 8 chains needing this simultaneously live at the
multiply stage, that's `8×8=64` registers required against only 32 that
exist — confirmed via disassembly (20 whole-register spill/reload
instructions). **`NCHAINS=3`** keeps peak simultaneous use at `3×8=24`
(multiply stage), comfortably under 32 with no spilling — the largest
chain count that fits cleanly at this `LMUL`.

**Result — GCC 13.4.0-1, gem5 (cycle-accurate): 206 cycles/iteration,
3.723 GFLOP/s.** This is the authoritative figure used in
`dtln/performance_dtln.md`'s roofline efficiency table, since the project's
actual kernel is GCC-built. Cross-checked against clang-18 for the same
source (see "Investigating clang as an alternative toolchain" below):
278 cycles/iteration, 2.760 GFLOP/s — same order of magnitude (confirms
this reflects a genuine hardware-bound ceiling, not a compiler artifact
of either toolchain), GCC ~35% faster for this exact code, consistent
with GCC being the toolchain this project's real kernel already uses.
Both produce identical `sink` values (`-361804616`), confirming
correctness on both.

Against this real, measured ceiling instead of the wrong `25.6 GFLOP/s`
figure, the vectorized kernel's efficiency comes out to ~18-21% (was
~2.6-3.1% against the wrong ceiling) — meaning it's already within ~5× of
what this CPU can genuinely sustain for this op mix, not 30-40× short of
it. See `dtln/performance_dtln.md`'s "Achieved performance vs. the roofline"
section for the full corrected table and comparison against the sibling
`gemm` project's own efficiency findings.

### Correction: the 2-FU experiment helps the real kernel much less than the synthetic probe suggested

The `MINOR_FLOAT_FU_COUNT=2` experiment above was only ever run against
the synthetic `LMUL` comparison probe (a standalone loop doing nothing but
the dot-product chain repeatedly), which showed 14-21% fewer cycles at the
shapes `dtln` actually uses. Two things needed checking before trusting
that number for the real kernel: (1) does the *ceiling* itself also move
with 2 FUs (yes — see above, `int8dot_ceiling.c` goes from 3.723 to
**4.510 GFLOP/s**, +21.2%, so any 2-FU real-kernel comparison needs to be
against this ceiling, not the 1-FU one), and (2) does the real kernel —
which does much more per call than the isolated dot-product loop — see
the same proportional benefit at all.

Re-ran the actual `dtln_noise_suppression` benchmark (GCC-built, same
binary, only `GEM5_FS_CONFIG`/`MINOR_FLOAT_FU_COUNT` changed — no
rebuild) under both configs:

| | 1 FU cycles | 2 FU cycles | Cycle reduction | Efficiency @ 1 FU ceiling (3.723 GFLOP/s) | Efficiency @ 2 FU ceiling (4.510 GFLOP/s) |
|---|---|---|---|---|---|
| `FULLY_CONNECTED` | 84,311 | 80,654 | 4.34% | 20.96% | **18.09%** |
| LSTM 1st call | 573,952 | 542,501 | 5.48% | 18.45% | **16.11%** |
| LSTM 2nd call | 394,391 | 377,801 | 4.21% | 17.85% | **15.39%** |
| **Whole-model** | 1,141,740 | 1,089,804 | **4.55%** | — | — |

> Re-measured 2026-08-18 (this row corrects an earlier version of this table
> that used a stale baseline — `70,821`/`578,992`/`394,771` cycles — from
> a build that predated a later fix without a full clean rebuild; the
> `.d`-dependency-tracking gap noted elsewhere in this doc let a stale
> `gen/` object slip through undetected across several re-confirmations
> in the same session, since re-measuring the *same* stale binary just
> reproduces the same wrong number. Caught by an unrelated later
> measurement disagreeing with this table by an amount too large to be
> gem5's normal ~0.5% run-to-run noise; re-verified via a full `rm -rf
> gen/` from scratch, reproduced identically 4 times via 3 independent
> methods (direct default-board run, the demo-minor config at
> `MINOR_FLOAT_FU_COUNT=1`, and a repeat), before trusting it. The
> qualitative conclusion below is unchanged — efficiency still goes
> *down* with 2 FUs, not up — only the absolute cycle counts moved.

**The real kernel gets roughly a third of what the synthetic probe
predicted (4.55% whole-model vs. 14-21% for the probe at the same
shapes), and relative efficiency against the correctly-scaled ceiling
actually goes *down*, not up.** Part of this is genuinely Amdahl's Law:
the real kernel's cycle count includes real scalar work that never
touches the `FloatSimd` FU at all — bias addition, quantization-multiplier
rescaling, activation min/max clamping, LSTM's gate-combination/cell-state
logic — none of which a 2nd vector FU can help. But direct measurement
(see "Realistic FULLY_CONNECTED bottleneck decomposition" below) found
this scalar share is smaller than it sounds — vector dot-product work is
actually ~74% of a real `FULLY_CONNECTED` call's cycles, not a minority
diluted by scalar overhead. The bigger factor is that `MinorCPU`'s 2-wide
issue front end caps how much *any* FU count can help even within that
dominant vector portion (see "The ceiling saturates at 2 FUs" below) —
`int8dot_ceiling.c` gets closer to the full available benefit because it's
hand-interleaved to exploit that issue width; the real kernel's
un-interleaved `out_c` loop can't reach the same ceiling as cleanly. Since
the ceiling itself grew *more* (21.2%) than the real kernel's achieved
performance did (~4-5%), efficiency-against-ceiling nets out lower with 2
FUs, not higher.

**Conclusion: the 2-FU hardware-model change is a weaker lever for this
project's actual workload than the synthetic probe alone suggested.**
Doesn't change the "not applied to the default board" decision above —
if anything, this is a further reason not to, since the real payoff is
smaller than it first looked.

### The ceiling saturates at 2 FUs: `MinorCPU`'s 2-wide issue front end, not FU count, is the real limiter

Follow-up on the 2-FU result above: re-ran `int8dot_ceiling.c` at 3 and 4
`FloatSimd` FUs, to check whether the pattern was "FU count still helps,
just less each time" (diminishing returns) or something sharper.

| FUs | cycles/iter | GFLOP/s | Δ vs. previous |
|---|---|---|---|
| 1 | 206 | 3.723 | — |
| 2 | 170 | 4.510 | +21.2% |
| 3 | 170 | 4.511 | **+0.0%** |
| 4 | 170 | 4.511 | **+0.0%** |

Not diminishing returns — a flat wall. 3 and 4 FUs give *exactly* zero
further improvement over 2, down to the cycle. Notably this saturates at
2 FUs even though the probe has 3 independent chains — if FU
*availability* were still the constraint, a 3rd FU should let all three
chains' compute ops run fully in parallel. It doesn't, because
`RiscvMinorCPU`'s dual-issue front end (`executeIssueLimit`/
`executeInputWidth`/`decodeInputWidth`/`executeCommitLimit` all `=2`,
`src/cpu/minor/BaseMinorCPU.py:359-386`) caps total instructions issued
per cycle at 2, across every instruction class combined (vector compute,
vector loads via the separate `MemFU`, and this project's own scalar loop
overhead) — a fixed property of the modeled CPU that no number of extra
`FloatSimd` FU instances can work around. Correctness held (`sink`
identical across all four runs).

**Why the isolated probe reaches a GFLOP/s figure the real kernel can't
get anywhere close to, even though it's measuring the exact same
instruction sequence: the probe is deliberately too idealized to be a
usage scenario, by design.** A ceiling probe's job is to answer "what's
the best this hardware could possibly do for this instruction mix," not
"what does the real workload achieve" — the gap between the two is the
point, not a flaw. Two separate ways the probe is unrealistic relative to
`dtln`:

1. **Scope.** It's *only* the vector chain (`load -> widen -> widen ->
   multiply -> reduce`) — none of the real kernel's surrounding cost (bias
   add, quantization-multiplier rescale, activation clamp,
   per-output-channel loop/call overhead, LSTM gate-combination logic).
   None of that touches the `FloatSimd` FU, so more FUs can't help it, but
   it still costs real cycles in the actual kernel.
2. **Code shape.** Even restricted to just the vector-compute part, the
   probe is *hand-interleaved* across 3 independent chains specifically to
   extract whatever overlap the 2-wide issue front end allows. The real
   kernel doesn't do that — each output channel's dot product runs as its
   own separate, non-interleaved call.

Tried porting the same chain-interleaving idea directly into the real
kernel's own `out_c` loop next (below) — it did not reproduce the probe's
gain, for exactly this reason. `3.723 GFLOP/s` isn't "what this kernel
should hit"; it's a hard upper bound for one instruction sequence in
isolation, and the real kernel's 18-21% against it is the honest measure
of how far a much messier, real workload sits from that idealized best
case. This section's finding sharpens that ceiling further: even in the
*best possible* case, more `FloatSimd` FUs stops helping past 2, so this
lever was never going to be a big win for the real kernel either.

### Realistic FULLY_CONNECTED bottleneck decomposition: vector work dominates, not scalar overhead

Every paragraph above (the 2-FU correction, the ceiling saturation) has
been explaining the real kernel's weak FU-count/interleaving payoff by
pointing at scalar post-processing — bias add, requantization, activation
clamping — as "a lot that never touches the `FloatSimd` FU." That claim
was never actually *measured*, just asserted from first principles. Built
[`../patterns/microbenchmark/fc_bottleneck.c`](../patterns/microbenchmark/fc_bottleneck.c)
to check it directly, at `dtln`'s real FC shape (`M=1,K=128,N=257`) and
real quantization params (pulled from the flatbuffer, not guessed:
`input_offset=4`, `filter_offset=0`, `output_offset=-2`,
`output_multiplier=1820954201`, `output_shift=-7`) — full per-channel
pipeline (dot product -> bias -> requantize -> offset -> clamp -> store),
with each stage individually removable via a compile-time `FC_VARIANT`
flag (a *compile-time* flag, not a runtime branch -- see below for why
that distinction mattered again here).

**First attempt: a standalone replica, and a real methodology trap.**
The `FC_VARIANT=0` (full pipeline) build measured only 19,510 cycles for
one 257-channel pass — nowhere near the real kernel's 84,311. Disassembly
found why: `filter_offset` had been passed in as a plain `const int32_t
filter_offset = 0` local. Since that's a compile-time-visible literal
zero, GCC proved `filter_offset * input_sum` in `Int8DotProductRvv` is
always exactly 0 and deleted the entire `input_sum` reduction at compile
time — the standalone benchmark was silently measuring a *cheaper,
different* computation than the real kernel, which sees `filter_offset`
as a genuine runtime value (`params.weights_offset`, populated from the
model at `Prepare()` time) that GCC has no way to fold away, even though
it happens to equal 0 for this model. Marking the quantization params
`volatile` (forcing genuine runtime reads GCC can't constant-propagate)
restored the correct 3-reduction instruction sequence (confirmed via
disassembly) and closed part of the gap (19,510 -> 25,492 cycles), but a
real gap to the true kernel remained even so.

**A follow-up register-spill theory was raised and then disproven by a
fresh disassembly recheck (2026-08-19).** The working theory at the time
was that the real kernel's per-channel loop spills scalar constants
(`output_multiplier`, `output_shift`, `activation_min`/`max`, `zero32`)
to the stack and reloads them every iteration, based on a diff against
`fc_bottleneck.c`'s `FC_VARIANT=0` disassembly. That theory does not
hold up: a clean rebuild of both binaries (forced recompile, same GCC
13.4.0-1, same `-O2`, verified via object timestamps) followed by a
fresh disassembly of the real `FullyConnected<int8,int8,int8,int32>`
instantiation shows **zero** `csrr vlenb` instructions anywhere in the
object file, and the per-channel loop body holds `output_multiplier`,
the precomputed `round` constant, `output_shift`, and
`activation_max` in registers (`s7`, `s5`, `s4`, `s1`) across the entire
loop — hoisted once, never reloaded. `zero32` is materialized once
(`vmv.v.i v8,0`) outside every loop. The whole per-channel body contains
exactly one conditional stack load, on the cold clamp-to-min path.

The reload pattern that prompted the original theory turned out to
belong to `fc_bottleneck.c` itself, not the real kernel: every
quantization scalar in that microbenchmark (`output_multiplier`,
`output_shift`, `activation_min`/`max`, `output_offset`,
`input_offset`, `filter_offset`) is deliberately declared `volatile`
(see its header comment above) specifically to stop GCC from
constant-folding them away — a legitimate technique for keeping the dot
product honest, but one that also forces the requantize stage to reload
from memory every channel as a side effect. That's what the earlier
disassembly diff was actually seeing. The comparison that produced the
original claim mixed up which binary the reload pattern came from.

The inner *vector* instruction sequence genuinely is the same between
the two (same opcodes, same shape) — that part of the original
observation was correct, and it's still true the vector work itself
isn't the gap. But there is no register-pressure/spill bug left to fix
in the real kernel's `FullyConnected()`: after the
`MultiplyByQuantizedMultiplier` inlining fix (see "Applied: inlining
`MultiplyByQuantizedMultiplier`" below), its scalar quantization
constants already have clean register allocation. `fc_bottleneck.c`'s
standalone `FULL` pipeline still measures faster (2.581 GFLOP/s vs. the
real kernel's 1.203 GFLOP/s at this shape) — that gap is real and still
unexplained, but it is not caused by register spilling in the real
kernel, and no further lever has been identified here.

**Follow-up (2026-08-19, same day): the gap is fully explained — it's a
cache-warmth artifact in `fc_bottleneck.c`'s own benchmark methodology,
not anything about the real kernel.** Isolated the dot-product stage
specifically with a minimal 2-point `mcycle` probe (immediately before
and after the `Int8DotProductRvv` call, avoiding the boundary-count
inflation the 4-stage probe above carries) and got **204 cycles/channel**
for the real kernel — consistent with the 4-stage probe's 203, so not a
probe-placement artifact. `fc_bottleneck.c`'s own `FC_VARIANT=4`
(`DOT_ONLY`) measures only **74 cycles/channel** in its normal
configuration (`ITERS=20`, same `filter[257][128]` array reused every
pass) for the *identical* instruction sequence — a 2.76x gap that static
disassembly can't explain, since the code is the same.

The difference is cache state, not code. `fc_bottleneck.c`'s own init
loops write `filter`/`input`/`bias` immediately before the timed region,
and `ITERS=20` reuses that same ~33 KB array on every pass, so after the
first touch it's essentially permanently resident in the 64 KB L1 —
even the `ITERS=1` single-pass case inherits this, since the init loops
still just wrote it. The real kernel's `FullyConnected()` sees none of
that: its filter tensor was last touched at model-load time, and the
LSTM ops that run immediately before it in the same `dtln` inference
have a much larger combined working set that evicts it from L1 well
before `FullyConnected()` gets to read it — a genuinely cold cache on
every single invocation, never a warm, self-reused one.

Confirmed directly: patched a scratch copy of `fc_bottleneck.c`
(`ITERS=1`, `FC_VARIANT=4`) to touch a 256 KiB `volatile` distractor
buffer (4x the L1) immediately before the timed region, reproducing a
genuinely cold cache the same way the preceding LSTM ops do in the real
model. Result: **200 cycles/channel** — within 2% of the real kernel's
204. The entire 2.76x "gap" was this benchmark measuring an artificially
cache-hot scenario the real kernel never gets to be in; there is no
inefficiency in `FullyConnected()` to explain or fix here.

This also retroactively explains the smaller, already-known
`ITERS=1`-vs-`ITERS=20` gap in `FC_VARIANT=0` (29,732 vs. 25,492
cycles/pass, ~17%): that's the same effect at much smaller magnitude,
because the untouched `ITERS=1` case still benefits from the init loops'
implicit warm-up — it only pays the cost of one compulsory miss per
cache line, not a genuinely cold cache. The 256 KiB eviction step above
is what actually reproduces the real kernel's memory environment;
`ITERS=1` alone does not.

**A bigger L1 does not help — confirmed empirically, not assumed.**
Before formalizing the cold-cache mode below, tested whether the miss is
compulsory (unavoidable, cache-size-independent) or a capacity/conflict
miss a bigger cache could prevent. `dtln_test.cc` calls `Invoke()`
exactly once (not a streaming loop over frames), and the model's weights
are baked into the binary image (`dtln_noise_suppression_model_data.cc`)
and loaded before execution starts — so `FullyConnected()`'s filter
tensor is read exactly once, ever, in the whole program's lifetime, by
construction. Bumped the L1 16x (`64kB` -> `1MB` in a scratch copy of
`sim_config/gem5_riscv_baremetal_fs.py`) with a proportionally larger
2 MiB eviction buffer (this target's bare-metal RAM is only 4 MB, so 4x
the *real* L1 was reused as the standard eviction size rather than scaled
with the experiment) and reran `FC_VARIANT=4` cold: **206
cycles/channel**, statistically identical to the 64 kB result. A bigger
cache only pays off when evicted data gets *re-referenced* later in the
same run; this access pattern has zero reuse within a single inference,
so no plausible L1 size changes anything here.

**Formalized as a permanent `FC_CACHE_MODE` build flag** (`0`=warm,
`1`=cold — see `fc_bottleneck.c`'s header comment, or
[`microbenchmark/README.md`](microbenchmark/README.md)'s "Test config" for
the short version) instead of leaving this as a one-off scratch-copy
experiment, so both ceilings are reproducible directly: `make
run-fc-bottleneck` / `make run-fc-bottleneck-cold`.

**Follow-up (2026-08-19, same day): unified `FC_ITERS=1` for both cache
modes.** The table below originally came from a WARM build that still
ran `ITERS=20` (reusing the same resident array every pass) against a
COLD build forced to a single pass — two variables differing at once
(cache state *and* iteration count), not just the one the mode flag is
named for. Pinned WARM to the same single timed pass as COLD (see
`fc_bottleneck.c`'s `FC_ITERS` definition); the array is already
L1-resident from the init-loop touch by the first pass either way, so
this mostly just removes the confound rather than changing what WARM
represents. Refreshed all 10 numbers below accordingly — `DOT_ONLY`
moved from 74 to 76 cyc/ch (3.450 → 3.365 GFLOP/s), a ~2.7% shift; COLD
numbers are unchanged since COLD was already single-pass.

**Follow-up (2026-08-20): fixed the same requantize rounding bug here as
in the real kernel** (see the correctness-bug entry in "Known
limitations" below — `fc_bottleneck.c`'s own `MultiplyByQuantizedMultiplier`
copy had the identical hardcoded-single-rounding bug, independently, and
was still out of sync with the just-fixed real kernel until this).
`DOT_ONLY` is unaffected (it skips requantize entirely — `FC_VARIANT=4`
never calls this function) and so is `NO_REQUANT` (skips it by
definition) — both columns' `DOT_ONLY`/`NO_REQUANT` rows below are
unchanged from the table above. `FULL`/`NO_BIAS`/`NO_CLAMP` all got
slower (the corrected double-rounding path has one more conditional):

| Variant | Warm cyc/ch | Warm GFLOP/s | Cold cyc/ch | Cold GFLOP/s | Cold/warm |
|---|---|---|---|---|---|
| `FULL` (0) | 131 | 1.946 | 256 | 0.999 | 1.95x |
| `NO_BIAS` (1) | 123 | 2.068 | 254 | 1.005 | 2.07x |
| `NO_REQUANT` (2) | 92 | 2.767 | 224 | 1.138 | 2.43x |
| `NO_CLAMP` (3) | 115 | 2.208 | 243 | 1.050 | 2.11x |
| `DOT_ONLY` (4) | 76 | 3.365 | 200 | 1.278 | 2.63x |

Every variant still slows ~2-2.6x cold vs. warm, same conclusion as
before (dot product, memory-bound either way, dominates all of them).
The cold `FULL` ceiling moved to 256 cyc/ch (0.999 GFLOP/s) — compare
against the real kernel's current directly-instrumented total, which
itself moved from 213.5 to ~177.7 cyc/ch (`45,658/257`, 1.44097 GFLOP/s,
post rounding-fix — see "Achieved performance vs. the roofline" in
`dtln/performance_dtln.md`) after the same fix made the real kernel
*faster* on this metric even though it now does the algorithmically
*correct* thing — counterintuitive, but consistent with the real kernel
also gaining the extra conditional and the ⚠ >100%-cold-ceiling anomaly
documented in "Known limitations" below (still open, still unexplained,
and this is more evidence for it, not against it). The qualitative
"real kernel beats or nearly matches the cold `FULL` ceiling, because
`fc_bottleneck.c`'s `volatile`-forced requantize recomputes `round` from
scratch every channel (see the fidelity-gap note above) while the real
kernel hoists it once" conclusion still holds — the real kernel beats
this new cold `FULL` ceiling by an even wider margin than before.

`DOT_ONLY`'s warm figure (3.365 GFLOP/s) is also tracked as the roofline
report/SVG's third ceiling line (`script/4_roofline_report.py
--measure-fc-warm-ceiling`) — same op, shape, and single-pass count as
the cold-cache ceiling, differing only in cache state, so it isolates
exactly the cache-eviction cost against the cold line rather than mixing
in the int8dot ceiling's different harness.

`fc_bottleneck.c` itself is kept in the microbenchmark directory as a
real, working tool, useful for stage-relative ablation (bias/requant/
clamp cost deltas, which aren't memory-bound the way the dot product is)
and much faster to iterate on than rebuilding the whole benchmark. For
absolute, real-kernel-comparable numbers, build with `FC_CACHE_MODE=1`
(cold) — the default (`0`, warm) reflects a best-case that the real
kernel's actual memory environment never gets close to.

**What actually resolved it: instrumenting the real kernel directly**,
the same technique used to root-cause the original offset-truncation bug
earlier in this doc. Added temporary `mcycle` probes bracketing each
stage inside `FullyConnected()`'s `out_c` loop (guarded on
`output_depth==257` so LSTM's gate matmuls, same template function,
`output_depth==128`, don't pollute the aggregate), rebuilt, ran the real
`dtln_noise_suppression` benchmark, reverted immediately after. Two
independent passes (a 2-way dot-vs-post split, then a 4-way split) agreed
to within 0.3%:

| Stage | Cycles | % of clean 84,311-cycle baseline |
|---|---|---|
| Vector dot product (`Int8DotProductRvv`, 257 calls) | ~62,500 | **~74%** |
| Requantize (`MultiplyByQuantizedMultiplier`) | ~19,600 | ~23% |
| Clamp + store | ~10,900 | ~13% |
| Bias add | ~8,500 | ~10% |

(Percentages sum past 100% — the probes themselves add real overhead to
every bucket, and stage boundaries aren't perfectly exclusive. Relative
ordering and rough magnitude are the trustworthy part.)

**Correction: the vector dot-product work is the majority of the real
cost (~74%), not the scalar post-processing.** The scalar work is real
and non-trivial — requantize alone (~23%) outweighs bias and clamp
combined — but "a lot that never touches the FloatSimd FU" overstated its
share; roughly three-quarters of this op's cycles are spent in exactly
the code the FU-count and interleaving experiments were trying to
speed up.

**This does *not* mean the 2-FU/interleaving experiments' weak results
were mismeasured — it changes *why* they underperformed.** Given vector
work is the majority, a 2nd FU should help substantially if FU
availability were the bottleneck. It measurably doesn't (`FULLY_CONNECTED`
only 4.34% faster with 2 FUs, "Correction" above), which lines up with
"The ceiling saturates at 2 FUs" above: `MinorCPU`'s 2-wide issue front
end caps how much *any* number of `FloatSimd` FUs can help, and that cap
applies inside this dominant vector portion too, not just to the isolated
ceiling probe. The probe gets closer to the full FU-count benefit *within
that cap* because it's hand-interleaved to maximize issue-slot overlap;
the real kernel's `out_c` loop isn't (see "Tried: interleaving" below —
porting that interleaving into the real loop didn't reproduce the probe's
gain either). So: vector work dominates the real kernel's cycles, but
the reason more FUs / interleaving don't help much isn't "it's mostly
scalar" — it's that the 2-wide issue front end caps the achievable
benefit inside the vector work itself, and the real kernel's un-interleaved
code shape can't reach even that capped benefit as cleanly as the
isolated probe does.

### Applied: inlining `MultiplyByQuantizedMultiplier` — the single biggest win found in this project

Direct follow-up to the requantize stage's ~23% share above: disassembled
the actual call site and found `MultiplyByQuantizedMultiplier(int32_t,
int32_t, int)` compiles to a genuine out-of-line `jalr` — confirmed via
the relocation table (`R_RISCV_CALL_PLT
_ZN6tflite29MultiplyByQuantizedMultiplierEiii`) — because the shared
version in `tensorflow/lite/kernels/internal/common.cc` is declared
`TFLITE_NOINLINE`. That's deliberate upstream code-size discipline (the
function is called from many kernels across the codebase; inlining it
everywhere would bloat every call site), but on this specific hot path
it's called once per output channel — 257 times for `dtln`'s FC alone,
plus every LSTM gate matmul (8 gate weight matrices × 2 LSTM calls) — so
the call/return overhead (caller-saved register spills around the call,
the jump itself, the callee's own prologue/epilogue) pays out repeatedly
in a way it wouldn't at a single call site elsewhere.

Added a byte-for-byte copy of the same algorithm as a local, force-inlined
`MultiplyByQuantizedMultiplierInlined()` in `fully_connected.h` (the
shared `TFLITE_NOINLINE` version is untouched everywhere else in the
codebase — this doesn't change upstream behavior or code size anywhere
except this one file). Purely a call-overhead fix, no numerical change:
rebuilt clean, re-verified correctness against all 4 cross-shape models
(`dtln`, `mnist_lstm`, `micro_speech_quantized`, `hello_world_int8`) —
every CRC32 matched exactly, both before and after.

**First version applied it unconditionally (both scalar and RVV builds,
since the algorithm doesn't depend on `__riscv_vector`) — and that broke
the scalar baseline.** A clean A/B rebuild (`rm -rf gen`, not a
stale-build artifact) showed dtln's *scalar*-target LSTM cycles jumped
from 2,498,098 to 4,651,956 (**+86%**) with nothing else changed. Most
likely cause: this inline copy gets duplicated into the scalar path's
much larger per-element loop body (a genuine `K`-iteration scalar loop,
unlike the RVV path's tight ~8-instruction vector chain), bloating that
function's code size enough to cause real I-cache pressure on gem5's
64 kB 4-way L1 — not confirmed further, since the fix (below) resolved
it without needing to. This mattered a lot: every "vectorized speedup vs.
scalar baseline" number in `dtln/performance_dtln.md` depends on the scalar
baseline staying exactly as it was, so an inflated scalar denominator
would have silently made every speedup number in this project look
better than it really is. Fixed by scoping the inlined version to
`#if defined(__riscv_vector)` only — the scalar path falls back to the
original shared, `TFLITE_NOINLINE` function, matching how every other
optimization in this file is already gated. Verified: scalar baseline
restored bit-for-bit (351,280/2,498,098/1,704,241 cycles, identical to
before either change), vector-target win fully intact (unchanged from the
table below).

**Result: a 29.34% whole-model cycle reduction** — bigger than every
other optimization in this project combined (`LMUL` widening: 25-34% on
the vector portion alone; 2 FUs: 4-6%; interleaving: mixed, -3% to +10%).

| | Before | After | Change |
|---|---|---|---|
| `FULLY_CONNECTED` | 84,311 | 54,858 | **-34.93%** |
| LSTM 1st call | 573,952 | 395,600 | **-31.07%** |
| LSTM 2nd call | 394,391 | 270,328 | **-31.46%** |
| `LOGISTIC` (untouched control — doesn't call `FullyConnected()`) | 89,086 | 86,014 | -3.45% (noise floor) |
| **Whole-model** | 1,141,740 | 806,800 | **-29.34%** |

Efficiency against the measured ceiling (3.723 GFLOP/s) moved
correspondingly: FC 20.96% -> **32.21%**, LSTM-1 18.45% -> **26.77%**,
LSTM-2 17.85% -> **26.05%**. `LOGISTIC`'s -3.45% move (an op this change
doesn't touch at all) is the noise floor for this kind of comparison —
confirms the effect is isolated to what actually changed, not measurement
drift.

**Applied and kept** — unlike every other experiment in this section
(2-FU, interleaving, the `filter_offset==0` skip), this one is now part of
the project's standing kernel, not a reverted diagnostic. The mechanism
(removing call/return overhead) is well-understood and predictable in
direction, unlike the earlier `filter_offset==0` attempt which looked
equally "obviously free" but measurably backfired via a GCC
scheduling artifact (see "Tried: interleaving" below for another instance
of that same lesson) — this one doesn't touch the vector code's
scheduling at all, only removes a scalar call, and the result matches
what the mechanism predicts.

This also means every cycle count and percentage in the sections above
this one (the 2-FU correction, the ceiling-saturation experiment, and the
bottleneck-decomposition percentages themselves) was measured **before**
this fix, against what's now a superseded baseline. Their qualitative
conclusions (2-wide issue width caps FU-count benefit; vector work is the
majority of real cost; the isolated ceiling probe overstates real-kernel
gains) are unaffected, but the exact cycle counts and percentages in
those sections reflect the pre-inlining kernel and haven't been
re-measured against the new baseline.

### Tried: interleaving independent output-channel dot-product chains — mixed result, not applied

Software-only follow-up to the FU analysis above: `int8dot_ceiling.c`
proved that hand-interleaving independent dot-product chains (all loads,
then all widens, then all multiplies, then all reduces, instead of running
each chain's full load->widen->multiply->reduce sequence to completion
before starting the next) hides much of the shared `FloatSimd` FU's
6-cycle latency. `FullyConnected()`'s `out_c` loop is exactly this shape —
every output channel's dot product against the same input row is
independent of every other one, computed by calling `Int8DotProductRvv()`
once per channel, back-to-back. Tried applying the same interleaving
technique there: a new `Int8DotProductRvvNx` helper computing N channels'
dot products at once with their chains interleaved (sharing one input
load/widen/`input_sum` reduction across all N, since the input row is the
same for every channel), wired into the `out_c` loop in chunks of N with a
scalar tail for the remainder. Correctness held throughout (all 4
cross-shape models' CRC32s matched the scalar-target reference at every N
tried).

**N=3** (matching the ceiling probe's own chain count): regresses
`FULLY_CONNECTED` by itself (84,311 -> 87,902 cycles, **+4.26%**) —
disassembly showed why: this helper carries more simultaneously-live
vector state per chain than the probe's chains do (each channel also needs
its own `filter_sum` reduction, on top of the probe's plain
load/widen/multiply/reduce), so 3-way spills here (6 whole-register
`vs*r.v`/`vl*re*.v` spill/reload instructions per loop iteration) where
the probe's pure 3-way didn't. But LSTM's gate computations — which route
through this same `FullyConnected()` template — got *faster* despite the
spilling (LSTM 1st call 573,952 -> 536,376, **-6.55%**; 2nd call 394,391
-> 398,647, **+1.08%**; combined 968,343 -> 935,023, **-3.44%**), and
since LSTM is ~91% of the model's cycles, whole-model total came out
**2.79% better** (1,141,740 -> 1,109,852) even though the op it was
modeled on (FC) got worse.

> Re-measured 2026-08-18 against the corrected baseline above (was
> compared against the same stale `70,821`/`578,992`/`394,771` figures —
> the interleaved variant's own absolute cycle count barely moved on
> re-measurement, 87,138 -> 87,902, well within normal noise, so only the
> *baseline* it was being compared against was ever wrong here). The
> qualitative conclusion is unchanged.

**N=2** (tried specifically to eliminate the spilling): confirmed via
disassembly that this fits within RVV's 32 registers with no spills at
all — and made things worse across the board anyway: `FULLY_CONNECTED`
84,311 -> 93,128 (**+10.46%**, worse than the spilling N=3 version), LSTM
combined 968,343 -> 1,007,631 (**+4.06%**, a regression, unlike N=3's
improvement), whole-model total 1,141,740 -> 1,189,164 (**+4.15% worse**).
Eliminating the register spills did not recover the loss, or reverse its
direction — pointing at something less mechanical than register pressure
alone (the most likely candidate: GCC's own instruction scheduler is
already doing a reasonable job overlapping the simple, un-interleaved
per-channel chains in the baseline code, and restructuring into a larger,
more complex helper function interferes with that more than it adds new
overlap of its own — not confirmed further, since both variants already
pointed the same direction: not worth pursuing).

**Not applied.** The technique that closed a clean, reproducible gap in
the isolated ceiling probe does not transfer cleanly to the real kernel:
results are shape-dependent in a way that's hard to predict (net win for
LSTM's gate shapes, net loss for FC's own shape, at the *same* interleave
depth, through the *same* code path), and the one variant with no
mechanical downside (N=2, no spilling) was the worst of the three
configurations measured, not the best. Reverted to the original
one-channel-at-a-time `Int8DotProductRvv()`; `fully_connected.h` is
unchanged from the version documented above. Kept here as a genuine,
measured negative result — like the vector-accumulate-then-reduce-once
attempt above, worth recording precisely so it isn't tried again from a
plausible-sounding first-principles argument without re-deriving why it
didn't work.

### Investigating clang as an alternative toolchain: builds clean, but crashes gem5 on the full `dtln` model

While cross-validating the roofline ceiling probe against both compilers
(see the ceiling-microbenchmark work above), the question came up of
whether TFLM's actual `riscv64_baremetal_vector` build — not just small
standalone probes — could also be compiled with clang instead of GCC
13.4.0-1.

**Flag compatibility: only one GCC-specific flag.** Tested every flag in
`riscv64_baremetal_vector_makefile.inc`'s `PLATFORM_FLAGS` plus the
project's base `CXXFLAGS`/`CCFLAGS` individually against `clang-18`.
Every single one compiled clean except `-mexplicit-relocs`
(`clang-18: error: unknown argument`) — a GCC-specific flag clang doesn't
recognize and doesn't need.

**Built wrapper scripts** (`script/clang_wrapper/riscv-none-elf-{gcc,g++}`)
standing in for the real toolchain, so `TARGET_TOOLCHAIN_ROOT` can point at
them with zero changes to TFLM's actual `Makefile`/`Makefile.inc` files.
Three fixes were needed beyond dropping `-mexplicit-relocs`:
- **Explicit libstdc++ `-I` paths.** clang's `--gcc-toolchain=` GCC
  auto-detection keys off `--target=` matching the on-disk toolchain
  triple name; `riscv64-unknown-elf` doesn't match this toolchain's actual
  `riscv-none-elf` naming, so the scan silently finds no C++ headers at
  all (`fatal error: 'utility' file not found` was the first symptom) --
  same underlying gotcha the sibling `softmax_cpp` project's `Makefile`
  already documents for its own clang usage.
- **Explicit `-L` paths at link time**, so the linker finds
  `-lgcc`/`-lc`/`-lm` — clang doesn't know this specific GCC toolchain's
  library layout the way real `gcc` does internally
  (`ld.lld-18: error: unable to find library -lgcc` otherwise).
- **Symlinked the real binutils** (`ar`, `objcopy`, `objdump`, `size`,
  `nm`, `ranlib`, `strip`) into the wrapper directory — these are
  toolchain-agnostic GNU tools, no reason to reimplement them; TFLM's
  `Makefile` resolves them via the same `TARGET_TOOLCHAIN_ROOT` +
  `TARGET_TOOLCHAIN_PREFIX` mechanism as `CC`/`CXX`.
- Both multilib/target-specific `-B`/`-L`/`-I` paths are pinned to GCC
  13.4.0-1's install layout and the `rv64im/lp64` multilib
  `-march=rv64imc_zicsr_zve64x` resolves to (matching what
  `-print-multi-directory` reports) — see the wrapper scripts' own header
  comments for the full explanation and the regenerate-if-toolchain-
  changes caveat.

**Result: the entire TFLM kernel library compiles clean under clang —
zero source-level changes needed anywhere.** Hundreds of `.cc`/`.c` files
across the whole `libtensorflow-microlite.a` build, only harmless warnings
(`-Wdouble-promotion`, an unrecognized `#pragma GCC diagnostic` warning
group in a vendored dependency). `test_hello_world_test` built, linked,
and ran correctly end-to-end under gem5: `~~~ALL TESTS PASSED~~~`, `Pass`.

**But the full `dtln_noise_suppression` benchmark crashes gem5 itself —
a segfault, not a wrong-answer or slow-performance issue.** Rebuilding
`run_tflm_benchmark` with the same clang wrapper and running it under
gem5 produces no output at all:

```
gem5 has encountered a segmentation fault!
...
gem5.opt(...RiscvISAInst::Vlse32_vMicro::initiateAcc...)
```

`Vlse32_v` is a **strided** vector load — notably, `Int8DotProductRvv`
(the kernel this whole investigation is centered on) never uses strided
loads, only unit-stride `vle8.v`. This means clang's auto-vectorizer chose
a strided-load codegen pattern *somewhere else* in the much larger
`dtln`/LSTM code surface (`lstm_eval.cc`'s gate/state-buffer handling,
quantization code, activation functions — none of which
`hello_world_test`'s single tiny `FULLY_CONNECTED` op ever touches) that
triggers what looks like a real bug/gap in gem5's `MinorCPU` RVV decode —
GCC's codegen for the same C++ source apparently never emits this
instruction in the same spot, so this particular crash is specific to
clang's optimizer choices, not a TFLM source-level issue.

Checked for prior art before concluding this was worth documenting rather
than debugging further: TFLM's `Makefile` does have existing clang usage
for other targets (Xtensa uses `CC_TOOL := clang`), so clang support isn't
foreign to the project. A web search surfaced what looked like a
directly-relevant precedent (a claimed CI report of clang producing
crashing Cortex-M firmware specifically in `FullyConnected` evaluation),
but the specific issue URLs the search cited (`tensorflow/tensorflow#35939`,
`tensorflow/tflite-micro#390`) both returned `404` when checked directly —
they don't exist, so that specific claim couldn't be verified and isn't
being relied on here. This finding stands on its own local reproduction,
not on unverified external precedent.

**Not pursued further, reverted to GCC.** Root-causing exactly which
line/function clang vectorizes into the offending `vlse32.v` (and whether
it's fixable via a targeted `-fno-vectorize`-style flag on just that file,
or is a genuine gem5 bug that would need a gem5-side fix) was judged not
worth the effort relative to what it would unblock — GCC 13.4.0-1 already
produces correct, validated results for every number in this project.
`gen/riscv64_baremetal_vector_aarch64_default_gcc/` was rebuilt clean with
GCC after this investigation (bit-for-bit matching the documented
baseline: `Output CRC32: 0x7E578D1C`, gem5 total 1,141,740 ticks — see the
note on "Correction: the 2-FU experiment..." above about this figure
being corrected from an earlier, stale `1,133,683` — whisper 455,304). The wrapper scripts remain saved in `script/clang_wrapper/` as
a working, documented reference — useful for narrower experiments (like
the ceiling-microbenchmark cross-validation that prompted this
investigation, or `hello_world_test`-scale sanity checks) — but not
validated for building the full `dtln` benchmark, which is confirmed
broken.

## Known limitations / follow-ups not yet done

- **Root-caused and fixed (2026-08-20): `MultiplyByQuantizedMultiplierInlined`
  (the RVV-only requantize copy in `fully_connected.h`) used the wrong
  rounding algorithm for this build, corrupting vectorized `FULLY_CONNECTED`/
  LSTM output by ±1 on a data-dependent fraction of values.** Found via
  `anomaly_detection_int8.tflite` (MLPerf Tiny `ad01`), whose output CRC32
  didn't match scalar (`0xC6F70B6E` vs `0xFA7AD6B9`, reproduced twice) —
  but the bug itself was model-independent, latent in every RVV
  `FULLY_CONNECTED`/LSTM call in this project, including `dtln`'s (just
  never observed to flip a final CRC32 there — `dtln`'s specific values
  happened not to land on this bug's rounding-boundary cases, so it was
  "accidentally" correct before this fix, not correct by construction).
  `common.cc` has two `MultiplyByQuantizedMultiplier` implementations
  gated by `#if TFLITE_SINGLE_ROUNDING`: single-rounding (shift-based)
  when defined truthy, double-rounding (`gemmlowp::SaturatingRoundingDoublingHighMul`
  + `RoundingDivideByPOT`) otherwise -- the default, and confirmed
  (grepped the whole `tools/make/` tree) this project's build never
  defines `TFLITE_SINGLE_ROUNDING`. `MultiplyByQuantizedMultiplierInlined`
  hardcoded single-rounding *unconditionally*, contradicting its own
  header comment's claim of being "byte-for-byte unchanged from the
  shared version." Confirmed and quantified with a standalone probe
  (`microbenchmark/requant_correctness_probe.c`) reimplementing both
  algorithms and sweeping `anomaly_detection_int8.tflite`'s own 10 real
  `(multiplier, shift)` pairs against 2000 accumulator values each:
  455/20,000 mismatches, every one off by exactly `±1` (the textbook
  single-vs-double-rounding divergence, not a magnitude bug), 0.1%-12.1%
  per layer depending on its own multiplier/shift. Over 10 sequential
  layers this compounded into a guaranteed-different final result,
  matching the observed CRC32 mismatch. `Int8DotProductRvv` itself was
  separately verified correct at every shape/offset this model uses
  (`microbenchmark/int8dot_correctness_probe.c`, 10/10 passed) — ruling
  out an earlier `K=8`/`K=640` shape hypothesis; the bug was entirely in
  the requantize step. **Fix**: `MultiplyByQuantizedMultiplierInlined`
  now mirrors `common.cc`'s own `#if`/`#else` exactly, calling the real
  `gemmlowp` functions directly (`fully_connected.h` already transitively
  includes `fixedpoint.h` via `common.h`) instead of a from-scratch
  reimplementation, for both branches. Verified: `anomaly_detection`'s
  vectorized `Output CRC32` now matches scalar exactly (`0xC6F70B6E`);
  `dtln`'s vectorized `Output CRC32` is unchanged (`0x7E578D1C`). Cycle
  counts shifted slightly for both models (the corrected double-rounding
  path has one more conditional than the previously-hardcoded
  single-rounding path) — `dtln/performance_dtln.md` and
  `anomaly_detection/performance_anomaly_detection.md` were both
  refreshed with post-fix numbers. See "Correctness" in
  [`anomaly_detection/performance_anomaly_detection.md`](anomaly_detection/performance_anomaly_detection.md)
  for the full writeup, including why this never explained the separate
  >100%-cold-ceiling-efficiency anomaly below (a rounding-value bug
  changes results, not cycle counts — confirmed by that anomaly
  persisting, essentially unchanged, after this fix).
- **RESOLVED (2026-08-20): the >100%-cold-ceiling-efficiency anomaly was
  never a stable property of the kernel — it's build-to-build
  machine-code non-determinism from byte-identical source.** This
  explains the whole trail of drifting numbers this entry used to track
  (54,695 → 41,373 → 45,658 → 58,958, all claimed at various points to be
  "the" cycle count for the exact same `dtln` `FULLY_CONNECTED`
  vectorized op) — every one of those was a real, reproducible
  measurement *of a specific build*, not evidence of a regression or a
  stable ceiling-beating fact. Confirmed with three independent checks:
  1. **`git diff` on `tflite-micro` shows zero source changes** between
     the 45,658-cycle build and the 58,958-cycle build (the latter
     obtained by adding temporary per-channel `mcycle` instrumentation to
     `fully_connected.h`, measuring, then `git checkout --`-reverting it
     — see below) — so the two numbers are for byte-identical source.
  2. **The same already-built binary reruns in gem5 with bit-identical
     results** (`md5sum` unchanged, all 4 ops' tick counts unchanged
     across independent `gem5.opt` invocations) — ruling out simulation
     randomness. The variance is entirely in what machine code gets
     *produced*, not how it's *simulated*.
  3. **Multiple independent `rm -rf <gen tree> && rebuild` cycles from
     the same source converge on the same value and stay there** (58,958
     reproduced 3x in a row) — so it's not per-build-invocation random
     either; it's *sticky*, changing only when something about the build
     inputs (not necessarily this file's own content) shifts, most likely
     object-file link order or enumeration affecting final code/data
     layout in memory (untested exactly which factor, but the pattern —
     identical source, differing final binary, stable once produced — is
     consistent with that class of cause, not a simulation or measurement
     artifact).

  **Follow-up (2026-08-20, same day): far more robust than "any rebuild
  might drift" — and the specific trigger theorized below was directly
  tested and disproven.** Ran 3 more independent `rm -rf <gen tree> &&
  rebuild` cycles from the *current, untouched, git-committed* source
  tree (no edits in between) — **all 3 came back bit-identical**:
  58,958/437,137/313,777/86,405 ticks, every time, zero variance. So
  plain repeated rebuilding of a stable, already-checked-out tree is
  fully deterministic.

  Initial theory: the drift was triggered by `git checkout --`-based file
  recreation (new inode/mtime, same bytes) between the 45,658 and 58,958
  builds, feeding into object-file enumeration/link order. **Tested this
  directly and it does not hold**: edited `fully_connected.h` with a
  trivial, content-neutral change via a normal edit (confirmed to
  recreate the file's inode, same as `git checkout` would), then edited
  it back (`git diff` confirms byte-identical to before, inode changed
  *twice*) — a full clean rebuild afterward still gave **58,958**, no
  drift. Also ruled out: stale build state living outside the
  vector-target gen directory (a full `rm -rf gen` wiping *both*
  scalar and vector build trees entirely, not just the one this
  investigation had been narrowly cleaning, still gave 58,958), and
  `ccache` (not installed, no `ccache` env vars, `riscv-none-elf-gcc` is
  a real binary not a cache wrapper — so nothing persists across a `gen/`
  wipe from outside the project tree either).

  **Where this leaves it**: 58,958 is now confirmed reproducible across 5
  independent tests, including the two most likely disruption vectors
  (file recreation, stale caches) — call this the value for the current
  source tree with real confidence. The original 45,658 measurement's
  exact trigger remains **not identified** — every specific mechanism
  tested for reproducing it failed to, and the intermediate build states
  that led to it (several rebuilds with actually-different, larger
  `fully_connected.h` content during the instrumentation experiments
  described below) are not preserved anywhere to re-test directly. Not
  chased further; flagged in the TODO below for anyone who wants to
  bisect it properly (e.g. by re-deriving the exact original edit
  sequence step by step, which this investigation didn't keep a
  byte-for-byte record of).

  **Confirmed to actually resolve the anomaly, not just explain drift**:
  regenerated the full roofline report on the 58,958-cycle build —
  `FULLY_CONNECTED` efficiency vs. the cold ceiling is now **87.32%**,
  below 100%, the sane result (see `dtln/performance_dtln.md`). Doing the
  same for `anomaly_detection_int8.tflite` on a matching rebuild shows the
  *same* shuffling at the per-layer level: calls that previously exceeded
  100% (1, 4, 8) no longer all do (4 and 8 now sit at 85-86%; call 1 now
  reads even higher, 142%, and call 9 — previously nowhere close — now
  reads 111%) — confirming this isn't about specific shapes/layers being
  special, it's which layers happen to land on favorable memory layout in
  a given build, different every time.

  **What was checked and ruled out before finding this** (kept for the
  record, since these were real, correct eliminations, just not the
  actual explanation):
    - `fc_bottleneck.c`'s `volatile` offsets forcing extra per-channel
      stack reloads (confirmed real via disassembly, but only accounted
      for ~3 of the then-22-cycle/channel gap when removed in a scratch
      build).
    - A hardware prefetcher masking cold misses differently for the two
      access patterns (ruled out entirely — no prefetcher is configured
      anywhere in `sim_config/gem5_riscv_baremetal_fs.py`, and gem5's
      `BaseCache.prefetcher` defaults to `NULL`).
    - Per-channel `mcycle` instrumentation of the real kernel's dot
      product, cross-checked against `fc_bottleneck.c`'s own per-channel
      probe — this is what surfaced the build-sensitivity in the first
      place (measuring the real kernel changed its own timing between two
      supposedly-identical-source runs, which is what prompted checking
      binary identity rather than trusting the source diff alone). One
      methodology note from this step, worth keeping: an earlier attempt
      at per-channel instrumentation printed each channel's result with
      `printf` *inside* the timed loop — semihosting I/O calls between
      channels turned out to perturb cache state enough to invalidate the
      per-channel shape (channel costs read far higher than the
      un-instrumented aggregate). Buffering into a static array and
      printing everything in one burst *after* the loop fixed this.
  **Practical takeaway for reading any of this project's absolute cycle
  counts or ceiling-relative percentages going forward**: a number
  measured on this source tree and reproduced by rebuilding *is*
  trustworthy — 5/5 identical reruns support that, surviving every
  disruption this investigation specifically tried (file recreation,
  full `gen/` wipe). There is no known, reproducible trigger for
  drift — file-recreation-via-edit was the leading theory and it's now
  disproven by direct test, not just unconfirmed. The actual range
  observed across this investigation (mid-40s to high-50s thousands of
  cycles for `dtln`'s `FULLY_CONNECTED`, i.e. cold-ceiling efficiency
  from the high-80s to over 110%) still stands as what's *possible*
  across different binary layouts — the 45,658-cycle build was real, not
  a measurement error — but with no known way to reproduce or predict
  when it recurs, the practical guidance is simply: trust whatever the
  current build measures, and if a number looks surprising (like exceeding
  100% of an isolated-probe ceiling), that alone is grounds to re-measure
  before concluding it's a stable fact, not a specific pre-condition to
  watch for. The qualitative conclusions (vectorization wins big,
  cold-cache is the ceiling that matters, requantize rounding needed the
  earlier fix) are unaffected either way.
- A previously-suspected register-spill bug in the real `FullyConnected()`'s
  per-channel loop was investigated and ruled out, and the 2.581 vs. 1.203
  GFLOP/s gap against `microbenchmark/fc_bottleneck.c` that motivated it
  was fully explained the same day (2026-08-19) — see "Realistic
  FULLY_CONNECTED bottleneck decomposition" above. Short version: the gap
  is a cache-warmth artifact in the microbenchmark, not a real-kernel
  inefficiency — `fc_bottleneck.c` keeps its own filter array resident in
  L1 across repeated iterations in a way the real kernel's single,
  post-LSTM-eviction invocation never gets to be. Confirmed the miss is
  compulsory, not capacity-bound (a 16x larger L1 changed nothing), then
  formalized a cold-cache mode (`FC_CACHE_MODE=1`, `make
  run-fc-bottleneck-cold`) that reproduces the real kernel's actual
  memory environment and closes the gap to within 2%. No fix is needed in
  `FullyConnected()` itself, and no register-pressure or memory-locality
  lever remains open here — the dot product is genuinely latency-bound on
  a single, unavoidable cold DRAM access per channel, not inefficiently
  coded. The remaining lever, if anyone wants it, is architectural
  (software prefetch one channel ahead to overlap the miss latency, not
  a cache-size or code-shape change) — not attempted here.
- `keyword_benchmark` and `person_detection_benchmark` (the two dedicated
  benchmark binaries under `tensorflow/lite/micro/benchmarks/`, as opposed
  to the generic `tflm_benchmark`) haven't been tried on `riscv{32,64}_generic`
  under `qemu` (the only supported simulator there now — see "gem5 SE mode
  disabled" above).
- Per-op timing breakdowns are now real on `riscv64_baremetal` (see the
  `micro_time.cc`/`mcycle` section above) — for `riscv32_generic`/
  `riscv64_generic`, still unusable; those targets have no `micro_time.cc`
  override of their own, so `GetCurrentTimeTicks()` still always returns 0
  there. Same fix (an `mcycle`-reading `micro_time.cc`) would apply if
  needed, just not done for those targets yet.
- `person_detect.tflite` hasn't been re-run through `run_tflm_benchmark` to
  get its own per-op cycle breakdown yet (deliberately skipped for now,
  given its ~7–9 minute gem5 wall-clock cost) — only `dtln_noise_suppression`
  has real per-op numbers so far.
- `TARGET_TOOLCHAIN_ROOT`/`TARGET_TOOLCHAIN_PREFIX` must be overridden by
  hand on every invocation, since the upstream default toolchain doesn't
  run on this (aarch64) host at all. Could be made the target's own default
  if this host will be used long-term, but that's a bigger, more
  opinionated change than what was asked for here.
- `--cpu=minor` is gem5's default in `test_with_gem5.sh`/`gem5_riscv_se.py`
  for parity with the `gemm` project's convention, not because it was
  compared against `atomic`/`timing` for this workload — no performance
  claims should be read into that choice yet.
- No cycle-count/roofline-style analysis has been attempted here (unlike
  the `gemm` project) — this work only establishes that the RISC-V
  binaries execute correctly under gem5, not what their performance looks
  like.
- FS-mode bare-metal (`riscv64_baremetal`) is now a proper Makefile target,
  verified for `test_hello_world_test`, `test_micro_utils_test`, and
  `run_tflm_benchmark` (`person_detect.tflite`, 30 ops, 89,248 B arena —
  see above). `linker_semi.ld`'s memory budget is now `FLASH` 4 MB /
  `RAM` 4 MB (bumped from the original 128 KB / 2 MB once `tflm_benchmark`
  overflowed FLASH), which comfortably covers everything tried so far, but
  the full `test` suite still hasn't been run — some individual kernel
  tests with unusually large fixture tensors, or benchmarks with bigger
  models/arenas, could still exceed 4 MB and haven't been checked.
- There is no RV32 bare-metal FS-mode target (`riscv32_baremetal`) —
  `riscv64_baremetal` is RV64-only, mirroring how the FS-mode prototyping
  work happened to start on RV64. Adding an RV32 counterpart would follow
  the same pattern (new `rv32imc_zicsr`/`ilp32` target file + linker
  script), just not done yet.
- `keyword_benchmark`/`person_detection_benchmark` (the two dedicated
  benchmark binaries, as opposed to the generic `tflm_benchmark` verified
  above) are unverified under FS mode.
- `dtln_test` (see the `.init_array` bug section above) is the current pick
  for matrix-optimization benchmarking going forward — its one
  `FULLY_CONNECTED` layer (`M=1, K=128, N=257`) is the largest/most
  "square" GEMM shape among the example models checked
  (`hello_world`: `1×16×16` at most; `micro_speech`: `1×4000×4`, deep-K but
  only 4 outputs). Verified passing under both `SIMULATOR=gem5` and
  `SIMULATOR=whisper` now; no vectorized kernel exists yet to actually
  exercise the vector-capable whisper config against it.
- `whisper_rv64gcv_config.json` declares vector/float support that no
  current TFLM build actually uses — `riscv64_baremetal` still compiles
  `rv64imc_zicsr` (no `v`/`f`/`d`). It's a placeholder for future
  vectorized-kernel work (see the whisper section above), not something
  currently exercised; whisper's `Fp`/`Vector`/etc. HPM counters will read
  0 for every binary run against this target today.
- `tflm_benchmark` under `SIMULATOR=whisper` hasn't been tried yet — only
  `test_hello_world_test` has been verified with whisper so far.
