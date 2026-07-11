"""gem5 Full-System (FS) mode config for bare-metal RISC-V M-mode binaries
with semihosting I/O — e.g. TFLM's riscv64_baremetal target
(tensorflow/lite/micro/riscv64_baremetal/{start_semi.S,linker_semi.ld}).

Unlike gem5_riscv_se.py (syscall-emulation mode, for the normal
riscv{32,64}_generic targets whose binaries are newlib+Linux-ABI-ecall
executables run the same way QEMU's linux-user mode runs them), this
config boots the ELF directly at its linked entry point with no host OS
underneath at all — no kernel, no libc syscall emulation. The binary must
provide its own crt0 (stack/bss/data setup) and its own I/O (here, via
RISC-V semihosting SYS_WRITE0/SYS_EXIT calls implemented in start_semi.S,
which back newlib's _write/_exit).

This is the exact same System/CPU/cache setup used by the sibling `gemm`
project's bare-metal sim_configs (see
/home/ajno5/work/2_pattern/gemm/sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py),
reused here since it's dtype/workload-agnostic — only the workload ELF
differs.

Usage:
  gem5.opt -d <m5out_dir> gem5_riscv_baremetal_fs.py <elf_binary>
"""
import sys

import m5
from m5.objects import *

# --- System ---
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]
system.m5ops_base = 0x10010000  # enables m5ops pseudo-inst decoding

# --- CPU ---
system.cpu = RiscvMinorCPU()
system.cpu.isa = RiscvISA(vlen=512, elen=64)

# --- Memory bus ---
system.membus = SystemXBar()

# --- L1 caches (64 kB each, 4-way) ---
system.cpu.icache = Cache(
    size="64kB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)
system.cpu.dcache = Cache(
    size="64kB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.dcache.mem_side = system.membus.cpu_side_ports

system.cpu.createInterruptController()

# --- Memory controller ---
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

# --- Bare-metal workload (M-mode, no BBL/Linux) ---
system.workload = RiscvBareMetal()
system.workload.bootloader = sys.argv[1]
system.workload.wait_for_remote_gdb = False

# RISC-V semihosting — output goes directly to gem5's stdout
system.workload.semihosting = RiscvSemihosting()

system.cpu.createThreads()

root = Root(full_system=True, system=system)
m5.instantiate()

print("Starting bare-metal RISC-V M-mode simulation (MinorCPU, FS mode)...")
exit_event = m5.simulate()
print(f"Exit @ tick {m5.curTick()}: {exit_event.getCause()}")
