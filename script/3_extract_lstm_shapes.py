#!/usr/bin/env python3
"""Pull UNIDIRECTIONAL_SEQUENCE_LSTM gate tensor shapes straight out of a
.tflite flatbuffer, without needing a pip/full TensorFlow install.

A .tflite file is a flatbuffer serialized against
tensorflow/lite/schema/schema.fbs. Reading it only needs two things, both
already vendored in this checkout:
  - the flatbuffers Python runtime:
      tensorflow/lite/micro/tools/make/downloads/flatbuffers/python/
  - the pre-generated schema bindings:
      tensorflow/lite/python/schema_py_generated.py

Same technique used for the FC/Conv layer shapes in doc/performance.md's
"Benchmark candidate comparison" table, extended here to
UNIDIRECTIONAL_SEQUENCE_LSTM, which needs its own decoder since it has 24
fixed input operand slots (gate weights/biases/state/peephole/projection/
layer-norm) instead of FULLY_CONNECTED's plain [M,K]x[K,N].

Usage (must be run with cwd == tflite-micro/, same reason as
1_run_pattern.sh / 2_run_benchmark.sh — paths below are repo-relative):
  python3 /home/ajno5/work/2_pattern/tflm/script/3_extract_lstm_shapes.py \
    [path/to/model.tflite]

Defaults to dtln_noise_suppression.tflite (the standing benchmark target,
see doc/performance.md) if no path is given.
"""
import os
import sys

TFLM_HOME = "/home/ajno5/work/2_pattern/tflm/tflite-micro"
sys.path.insert(0, os.path.join(TFLM_HOME, "tensorflow/lite/micro/tools/make/downloads/flatbuffers/python"))
sys.path.insert(0, TFLM_HOME)

from tensorflow.lite.python import schema_py_generated as schema_fb  # noqa: E402

DEFAULT_MODEL = os.path.join(
    TFLM_HOME, "tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite"
)

TENSOR_TYPE_NAMES = {v: k for k, v in vars(schema_fb.TensorType).items() if not k.startswith("_")}

# UNIDIRECTIONAL_SEQUENCE_LSTM's fixed input operand order (schema.fbs).
# Unused optional slots (peephole/projection/layer-norm on a plain LSTM)
# come through as tensor index -1.
LSTM_INPUT_NAMES = [
    "input",
    "input_to_input_weights", "input_to_forget_weights", "input_to_cell_weights", "input_to_output_weights",
    "recurrent_to_input_weights", "recurrent_to_forget_weights", "recurrent_to_cell_weights", "recurrent_to_output_weights",
    "cell_to_input_weights", "cell_to_forget_weights", "cell_to_output_weights",
    "input_gate_bias", "forget_gate_bias", "cell_gate_bias", "output_gate_bias",
    "projection_weights", "projection_bias",
    "input_activation_state", "input_cell_state",
    "input_layer_norm_coefficients", "forget_layer_norm_coefficients",
    "cell_layer_norm_coefficients", "output_layer_norm_coefficients",
]


def tensor_info(subgraph, idx):
    if idx < 0:
        return None
    t = subgraph.Tensors(idx)
    shape = t.ShapeAsNumpy().tolist() if not t.ShapeIsNone() else []
    return {
        "name": t.Name().decode() if t.Name() else f"tensor_{idx}",
        "shape": shape,
        "type": TENSOR_TYPE_NAMES.get(t.Type(), t.Type()),
    }


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    with open(model_path, "rb") as f:
        buf = bytearray(f.read())

    model = schema_fb.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    print(f"Model: {model_path}")
    print(f"Subgraph operator count: {subgraph.OperatorsLength()}\n")

    found = 0
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        if opcode.BuiltinCode() != schema_fb.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM:
            continue
        found += 1

        print(f"=== Operator[{i}]: UNIDIRECTIONAL_SEQUENCE_LSTM ===")
        for k in range(op.InputsLength()):
            tidx = op.Inputs(k)
            info = tensor_info(subgraph, tidx)
            label = LSTM_INPUT_NAMES[k] if k < len(LSTM_INPUT_NAMES) else f"input[{k}]"
            if info is None:
                print(f"  [{k:2}] {label:32} -- (unused, index -1)")
            else:
                print(f"  [{k:2}] {label:32} shape={str(info['shape']):20} dtype={info['type']} name={info['name']}")

        print("  outputs:")
        for k in range(op.OutputsLength()):
            info = tensor_info(subgraph, op.Outputs(k))
            print(f"    [{k}] shape={info['shape']} dtype={info['type']} name={info['name']}")
        print()

    if found == 0:
        print("No UNIDIRECTIONAL_SEQUENCE_LSTM operator found in this model.")


if __name__ == "__main__":
    main()
