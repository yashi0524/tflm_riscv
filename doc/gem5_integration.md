# TFLite Micro: gem5 Support for riscv32_generic / riscv64_generic

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

## Known limitations / follow-ups not yet done

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
