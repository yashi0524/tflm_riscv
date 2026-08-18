#!/usr/bin/env python3
"""Collect everything doc/performance_dtln.md's "Roofline analysis" table
needs and print it, instead of assembling it by hand.

As documented in doc/performance_dtln.md ("Achieved performance vs. the
roofline") and gem5_integration.md, the table's inputs come from three
independent sources, only one of which is this benchmark run itself:

  1. Per-op CYCLE COUNTS (T = cycles / 1e9 s at the board's 1 GHz clock):
     from actually running run_tflm_benchmark under gem5 -- this script
     does that part, via `make ... run_tflm_benchmark` (same invocation
     as script/2_run_benchmark.sh), for TARGET=riscv64_baremetal (scalar)
     and TARGET=riscv64_baremetal_vector (vectorized).
  2. FLOPs and weight-bytes per op: NOT measured at runtime -- pulled
     straight from the .tflite flatbuffer's static tensor shapes (same
     technique as script/3_extract_lstm_shapes.py), since these are all
     batch-1 int8 GEMVs with a closed-form FLOP count. This script does
     that part too, generically for FULLY_CONNECTED and
     UNIDIRECTIONAL_SEQUENCE_LSTM (the only two op types the roofline
     table covers).
  3. The empirical compute-roof CEILING (default 3.723 GFLOP/s): from a
     *separate* microbenchmark, ../microbenchmark/int8dot_ceiling.c, not
     from this benchmark run at all -- it measures what MinorCPU's single
     shared FloatSimd FU can sustain for this kernel's instruction mix in
     isolation. This script does NOT re-measure it by default (pass
     --measure-ceiling to actually build+run that microbenchmark and
     parse its result, or --ceiling to supply an already-known value --
     both override the hardcoded default, which goes stale if the
     toolchain or kernel's instruction mix changes).

Known limitation: assumes 1 timestep per UNIDIRECTIONAL_SEQUENCE_LSTM
node call. True for dtln_noise_suppression's two LSTM layers (see
doc/gem5_integration.md's "LSTM vectorization" section -- "single
timestep/call") but NOT for genuinely multi-timestep models like
mnist_lstm's 28-timestep LSTM; FLOPs would need scaling by time_steps
there, not implemented.

Usage (must run with cwd == tflite-micro/, after sourcing
script/0_env_var_setup.sh in the same shell -- same requirement as
script/2_run_benchmark.sh and script/3_extract_lstm_shapes.py):

  python3 /home/ajno5/work/2_pattern/tflm/script/4_roofline_report.py
  python3 /home/ajno5/work/2_pattern/tflm/script/4_roofline_report.py --measure-ceiling
  python3 /home/ajno5/work/2_pattern/tflm/script/4_roofline_report.py \\
      --scalar-log /path/to/captured_scalar.log \\
      --vector-log /path/to/captured_vector.log   # skip re-running gem5

Report is printed to stdout AND written to analysis/roofline_log.txt
(project root, alongside doc/microbenchmark/script/sim_config -- created
if it doesn't exist yet), so each run leaves a plain-text record behind
without needing a shell redirect. Override with --output, or pass
--output - to only print to stdout.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from collections import defaultdict, deque

TFLM_HOME = "/home/ajno5/work/2_pattern/tflm/tflite-micro"
PROJECT_ROOT = os.path.dirname(TFLM_HOME)
sys.path.insert(
    0,
    os.path.join(
        TFLM_HOME, "tensorflow/lite/micro/tools/make/downloads/flatbuffers/python"
    ),
)
sys.path.insert(0, TFLM_HOME)

from tensorflow.lite.python import schema_py_generated as schema_fb  # noqa: E402

# Relative to TFLM_HOME, matching script/2_run_benchmark.sh's own
# GENERIC_BENCHMARK_MODEL_PATH convention -- the Makefile derives generated
# object-file paths from this string, so it must stay relative (an absolute
# path gets baked verbatim into a bogus gen/.../obj/... path and fails).
DEFAULT_MODEL = "tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite"
DEFAULT_ARENA = 16384
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "analysis", "roofline_log.txt")

# sim_config/gem5_riscv_baremetal_fs.py: RiscvMinorCPU, 1 GHz, VLEN=512/ELEN=64,
# DDR3_1600_8x8, no L2/L3 -- see doc/performance_dtln.md's "Machine parameters".
CLOCK_HZ = 1e9
PEAK_BW_GBPS = 12.8  # 1600 MT/s x 8 B
PEAK_COMPUTE_GFLOPS = 128.0  # idealized 1 vector-MAC-instr/cycle ceiling
RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BW_GBPS

# microbenchmark/int8dot_ceiling.c, GCC 13.4.0-1 -- re-measure (--measure-ceiling)
# if the toolchain or Int8DotProductRvv's instruction mix changes.
DEFAULT_CEILING_GFLOPS = 3.723

TICK_RE = re.compile(r"^(\S+) took (\d+) ticks")
CRC32_RE = re.compile(r"^(Input|Output) CRC32: (0x[0-9A-Fa-f]+)")
CEILING_GFLOPS_RE = re.compile(r"GFLOP/s=(\d+)\.(\d+)")

DTYPE_BYTES = {
    "INT8": 1,
    "UINT8": 1,
    "INT16": 2,
    "FLOAT16": 2,
    "BFLOAT16": 2,
    "INT32": 4,
    "FLOAT32": 4,
    "INT64": 8,
    "FLOAT64": 8,
}

BUILTIN_OP_NAMES = {
    v: k for k, v in vars(schema_fb.BuiltinOperator).items() if not k.startswith("_")
}
TENSOR_TYPE_NAMES = {
    v: k for k, v in vars(schema_fb.TensorType).items() if not k.startswith("_")
}


def require_cwd_tflm_home():
    if os.path.realpath(os.getcwd()) != os.path.realpath(TFLM_HOME):
        sys.exit(
            f"error: must run with cwd == {TFLM_HOME} "
            "(same requirement as script/2_run_benchmark.sh)"
        )


def require_toolchain_env():
    if not os.environ.get("TOOLCHAIN"):
        sys.exit(
            "error: $TOOLCHAIN is not set -- source script/0_env_var_setup.sh "
            "first, in this same shell (env vars don't persist across separate "
            "shell invocations)"
        )


# ---------------------------------------------------------------------------
# Flatbuffer shape extraction: FLOPs / weight-bytes per op, in execution order.
# ---------------------------------------------------------------------------


def tensor_shape_and_dtype(subgraph, idx):
    if idx < 0:
        return None, None
    t = subgraph.Tensors(idx)
    shape = t.ShapeAsNumpy().tolist() if not t.ShapeIsNone() else []
    dtype = TENSOR_TYPE_NAMES.get(t.Type(), str(t.Type()))
    return shape, dtype


def op_stats_fully_connected(subgraph, op):
    input_shape, _ = tensor_shape_and_dtype(subgraph, op.Inputs(0))
    filter_shape, filter_dtype = tensor_shape_and_dtype(subgraph, op.Inputs(1))
    if not input_shape or not filter_shape:
        return None, None
    n, k = filter_shape[-2], filter_shape[-1]
    m = 1
    for d in input_shape[:-1]:
        m *= d
    flops = 2 * m * k * n
    weight_bytes = n * k * DTYPE_BYTES.get(filter_dtype, 1)
    return flops, weight_bytes


# UNIDIRECTIONAL_SEQUENCE_LSTM's fixed input operand order (schema.fbs) --
# only need slots 1-8 (the 4 input_to_* + 4 recurrent_to_* gate weight
# matrices) for FLOPs/bytes; see script/3_extract_lstm_shapes.py for the
# full 24-slot layout (peephole/projection/layer-norm, unused here).
LSTM_GATE_WEIGHT_SLOTS = range(1, 9)


def op_stats_lstm(subgraph, op):
    flops = 0
    weight_bytes = 0
    any_found = False
    for k in LSTM_GATE_WEIGHT_SLOTS:
        if k >= op.InputsLength():
            continue
        shape, dtype = tensor_shape_and_dtype(subgraph, op.Inputs(k))
        if not shape:
            continue  # unused optional slot (index -1), e.g. no peephole
        any_found = True
        hidden, in_dim = shape[-2], shape[-1]
        flops += 2 * hidden * in_dim
        weight_bytes += hidden * in_dim * DTYPE_BYTES.get(dtype, 1)
    return (flops, weight_bytes) if any_found else (None, None)


OP_STAT_HANDLERS = {
    schema_fb.BuiltinOperator.FULLY_CONNECTED: op_stats_fully_connected,
    schema_fb.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: op_stats_lstm,
}


def extract_op_sequence(model_path):
    """[(op_name, flops_or_None, weight_bytes_or_None), ...] in execution
    order -- matches the order run_tflm_benchmark's MicroProfiler prints
    "<op> took N ticks" lines in, for a single-subgraph model with no
    control flow (true for dtln_noise_suppression)."""
    with open(model_path, "rb") as f:
        buf = bytearray(f.read())
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    seq = []
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        builtin = opcode.BuiltinCode()
        name = BUILTIN_OP_NAMES.get(builtin, f"OP_{builtin}")
        handler = OP_STAT_HANDLERS.get(builtin)
        flops, weight_bytes = handler(subgraph, op) if handler else (None, None)
        seq.append((name, flops, weight_bytes))
    return seq


# ---------------------------------------------------------------------------
# Running (or reading a captured log of) run_tflm_benchmark.
# ---------------------------------------------------------------------------


def run_benchmark(target, model_path, arena_size, simulator=None):
    toolchain_args = [
        f"TARGET_TOOLCHAIN_ROOT={os.environ['TOOLCHAIN']}/bin/",
        "TARGET_TOOLCHAIN_PREFIX=riscv-none-elf-",
    ]
    cmd = [
        "make",
        "-f",
        "tensorflow/lite/micro/tools/make/Makefile",
        f"TARGET={target}",
        *toolchain_args,
        "BUILD_TYPE=default",
    ]
    if simulator:
        cmd.append(f"SIMULATOR={simulator}")
    cmd += [
        "run_tflm_benchmark",
        f"GENERIC_BENCHMARK_MODEL_PATH={model_path}",
        f"GENERIC_BENCHMARK_ARENA_SIZE={arena_size}",
    ]
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        sys.exit(
            f"error: benchmark run failed (target={target}, exit "
            f"{result.returncode}):\n--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    return result.stdout


def parse_ticks(log_text):
    """[(op_name, cycles), ...] in the order printed."""
    ticks = []
    for line in log_text.splitlines():
        m = TICK_RE.match(line.strip())
        if m:
            ticks.append((m.group(1), int(m.group(2))))
    return ticks


def parse_crc32(log_text):
    crc = {}
    for line in log_text.splitlines():
        m = CRC32_RE.match(line.strip())
        if m:
            crc[m.group(1)] = m.group(2)
    return crc


def align(ticks, op_sequence):
    """Zips parsed (name, cycles) ticks with the flatbuffer's (name, flops,
    weight_bytes) sequence, both in execution order, matched by name so
    repeated op names (e.g. dtln's two separate LSTM nodes) line up with
    their own distinct shapes in order. Returns
    [(name, call_index, cycles, flops, weight_bytes), ...]."""
    by_name = defaultdict(deque)
    for name, flops, weight_bytes in op_sequence:
        by_name[name].append((flops, weight_bytes))

    counts = defaultdict(int)
    rows = []
    for name, cycles in ticks:
        counts[name] += 1
        flops, weight_bytes = (
            by_name[name].popleft() if by_name[name] else (None, None)
        )
        rows.append((name, counts[name], cycles, flops, weight_bytes))
    return rows


# ---------------------------------------------------------------------------
# Ceiling.
# ---------------------------------------------------------------------------


def measure_ceiling():
    microbench_dir = os.path.join(PROJECT_ROOT, "microbenchmark")
    cmd = ["make", "run"]
    print(f"+ (cd {microbench_dir} && {' '.join(cmd)})", file=sys.stderr)
    result = subprocess.run(
        cmd, cwd=microbench_dir, capture_output=True, text=True, env=os.environ
    )
    if result.returncode != 0:
        sys.exit(
            "error: ceiling microbenchmark failed:\n"
            f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    m = CEILING_GFLOPS_RE.search(result.stdout)
    if not m:
        sys.exit(
            "error: could not parse 'GFLOP/s=' from ceiling microbenchmark "
            f"output:\n{result.stdout[-2000:]}"
        )
    return float(f"{m.group(1)}.{m.group(2)}")


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def row_key(row):
    return (row[0], row[1])  # (name, call_index)


def build_report(model_path, op_sequence, scalar_rows, vector_rows, ceiling_gflops,
                  crc_scalar, crc_vector):
    lines = []

    def emit(s=""):
        lines.append(s)

    emit(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    emit(f"Model: {model_path}")
    emit(
        f"Machine: RiscvMinorCPU, {CLOCK_HZ/1e9:.0f} GHz, peak BW "
        f"{PEAK_BW_GBPS} GB/s, idealized peak compute {PEAK_COMPUTE_GFLOPS} "
        f"GFLOP/s, ridge point {RIDGE_POINT:.1f} FLOP/byte"
    )
    emit(f"Empirical compute-roof ceiling: {ceiling_gflops} GFLOP/s "
         "(microbenchmark/int8dot_ceiling.c)")
    emit(f"Input CRC32: scalar={crc_scalar.get('Input')} vector={crc_vector.get('Input')}")
    emit(f"Output CRC32: scalar={crc_scalar.get('Output')} vector={crc_vector.get('Output')}")
    if crc_scalar.get("Output") != crc_vector.get("Output"):
        emit(
            "WARNING: Output CRC32 mismatch -- should be identical; a "
            "vectorized-kernel correctness regression, or a stale/mismatched "
            "build, would look like this."
        )
    emit()

    emit("### Arithmetic intensity\n")
    emit("| op | call | MACs | FLOPs | weight bytes | AI (FLOP/byte) |")
    emit("|---|---|---|---|---|---|")
    seen_ai = set()
    for name, call_idx, _cycles, flops, weight_bytes in scalar_rows:
        if flops is None or (name, call_idx) in seen_ai:
            continue
        seen_ai.add((name, call_idx))
        macs = flops // 2
        ai = flops / weight_bytes if weight_bytes else float("nan")
        bound = "memory-bound" if ai < RIDGE_POINT else "compute-bound"
        emit(
            f"| `{name}` | {call_idx} | {macs:,} | {flops:,} | {weight_bytes:,} "
            f"| {ai:.2f} ({bound}) |"
        )
    emit()

    emit("### Achieved performance vs. the roofline (gem5, cycle-accurate)\n")
    emit(
        "| op | call | variant | cycles | T (µs) | P (MFLOP/s) | efficiency vs. "
        f"{ceiling_gflops} GFLOP/s | cycles/weight-byte |"
    )
    emit("|---|---|---|---|---|---|---|---|")

    scalar_by_key = {row_key(r): r for r in scalar_rows}
    vector_by_key = {row_key(r): r for r in vector_rows}
    key_order = list(scalar_by_key) + [k for k in vector_by_key if k not in scalar_by_key]
    all_keys = sorted(set(scalar_by_key) | set(vector_by_key), key=key_order.index)

    for key in all_keys:
        for variant, table in (("scalar", scalar_by_key), ("vectorized", vector_by_key)):
            row = table.get(key)
            if row is None:
                continue
            name, call_idx, cycles, flops, weight_bytes = row
            t_us = cycles / CLOCK_HZ * 1e6
            if flops is not None:
                p_mflops = flops / (cycles / CLOCK_HZ) / 1e6
                efficiency = (p_mflops / 1000) / ceiling_gflops * 100
                cyc_per_byte = cycles / weight_bytes if weight_bytes else float("nan")
                emit(
                    f"| `{name}` | {call_idx} | {variant} | {cycles:,} | "
                    f"{t_us:.2f} | {p_mflops:.2f} | {efficiency:.2f}% | "
                    f"{cyc_per_byte:.2f} |"
                )
            else:
                emit(
                    f"| `{name}` | {call_idx} | {variant} | {cycles:,} | "
                    f"{t_us:.2f} | n/a (no FLOP model for this op) | -- | -- |"
                )

    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arena-size", type=int, default=DEFAULT_ARENA)
    parser.add_argument("--simulator", default=None,
                         help="gem5 (default, cycle-accurate) or whisper (functional-only, not valid for this table)")
    parser.add_argument("--scalar-log", help="skip running the scalar-target benchmark; read its captured stdout from this file instead")
    parser.add_argument("--vector-log", help="skip running the vector-target benchmark; read its captured stdout from this file instead")
    parser.add_argument("--ceiling", type=float, default=None,
                         help=f"empirical compute-roof ceiling in GFLOP/s (default: hardcoded {DEFAULT_CEILING_GFLOPS}, from int8dot_ceiling.c's GCC 13.4.0-1 result)")
    parser.add_argument("--measure-ceiling", action="store_true",
                         help="build and run microbenchmark/int8dot_ceiling.c to get a fresh ceiling instead of using the hardcoded default")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"also write the report here (default: {DEFAULT_OUTPUT}); pass '-' to only print to stdout")
    args = parser.parse_args()

    require_cwd_tflm_home()

    op_sequence = extract_op_sequence(args.model)

    if args.scalar_log:
        with open(args.scalar_log) as f:
            scalar_log = f.read()
    else:
        require_toolchain_env()
        scalar_log = run_benchmark("riscv64_baremetal", args.model, args.arena_size, args.simulator)

    if args.vector_log:
        with open(args.vector_log) as f:
            vector_log = f.read()
    else:
        require_toolchain_env()
        vector_log = run_benchmark("riscv64_baremetal_vector", args.model, args.arena_size, args.simulator)

    scalar_rows = align(parse_ticks(scalar_log), op_sequence)
    vector_rows = align(parse_ticks(vector_log), op_sequence)
    crc_scalar = parse_crc32(scalar_log)
    crc_vector = parse_crc32(vector_log)

    if args.measure_ceiling:
        ceiling_gflops = measure_ceiling()
    elif args.ceiling is not None:
        ceiling_gflops = args.ceiling
    else:
        ceiling_gflops = DEFAULT_CEILING_GFLOPS

    lines = build_report(args.model, op_sequence, scalar_rows, vector_rows,
                          ceiling_gflops, crc_scalar, crc_vector)
    report_text = "\n".join(lines) + "\n"

    print(report_text, end="")

    if args.output != "-":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"(report also written to {args.output})", file=sys.stderr)


if __name__ == "__main__":
    main()
