"""gem5 syscall-emulation (SE) mode config for TFLM's riscv32_generic /
riscv64_generic targets.

Runs a single statically-linked RISC-V ELF binary directly (RV32 or RV64,
auto-detected from the ELF header — see detect_riscv_type() below) — this
is gem5's built-in equivalent of QEMU's linux-user mode (`qemu-riscv32`/
`qemu-riscv64`), which is what tensorflow/lite/micro/testing/test_with_qemu.sh
normally uses for these targets: no kernel, bootloader, or UART model
needed, since gem5 services the binary's syscalls (write/exit/brk/...)
directly against the host, same as qemu-user does.

Usage:
  gem5.opt -d <m5out_dir> gem5_riscv_se.py --cpu=minor <binary> [args...]

--cpu selects the CPU model: atomic (fastest, functional-only timing),
timing (RiscvTimingSimpleCPU), or minor (RiscvMinorCPU, in-order
pipelined — the default, matching this project's gemm sim_config
convention).

Both riscv32_generic and riscv64_generic build for *mc (imc/imac, see
tensorflow/lite/micro/tools/make/targets/riscv{32,64}_generic_makefile.inc)
— no vector extension — so this config always sets enable_rvv=False; the
rest of gem5's default extension set (imafdc) is a superset of what these
binaries actually use, which is harmless (unused capability, not a
correctness issue).
"""
import argparse

import m5
from m5.objects import (
    AddrRange,
    AtomicSimpleCPU,
    DDR3_1600_8x8,
    MemCtrl,
    MinorCPU,
    Process,
    Root,
    RiscvISA,
    SEWorkload,
    SrcClockDomain,
    System,
    SystemXBar,
    TimingSimpleCPU,
    VoltageDomain,
)

CPU_TYPES = {
    "atomic": AtomicSimpleCPU,
    "timing": TimingSimpleCPU,
    "minor": MinorCPU,
}


def detect_riscv_type(binary_path):
    """Reads the ELF header's EI_CLASS byte (offset 4: 1=32-bit, 2=64-bit)
    to pick "RV32"/"RV64" — matches what gem5's own SEWorkload.init_compatible
    (used below) infers to build the RiscvProcess32/64 subclass, so the
    CPU's ISA object stays consistent with it automatically instead of
    needing a manual --xlen flag that could be set wrong."""
    with open(binary_path, "rb") as f:
        header = f.read(5)
    if header[:4] != b"\x7fELF":
        raise ValueError(f"{binary_path} is not an ELF file")
    ei_class = header[4]
    if ei_class == 1:
        return "RV32"
    elif ei_class == 2:
        return "RV64"
    raise ValueError(f"{binary_path}: unrecognized ELF class byte {ei_class}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary", help="path to the RISC-V ELF binary to run")
parser.add_argument(
    "bin_args", nargs=argparse.REMAINDER, help="arguments passed to the binary"
)
parser.add_argument(
    "--cpu", choices=CPU_TYPES.keys(), default="minor", help="CPU model (default: minor)"
)
args = parser.parse_args()

riscv_type = detect_riscv_type(args.binary)

system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

system.cpu = CPU_TYPES[args.cpu]()
system.cpu.isa = [RiscvISA(riscv_type=riscv_type, enable_rvv=False)]

system.membus = SystemXBar()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.createInterruptController()

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

process = Process()
process.cmd = [args.binary] + args.bin_args
system.cpu.workload = process
system.workload = SEWorkload.init_compatible(args.binary)
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Beginning simulation! ({riscv_type}, cpu={args.cpu})")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
