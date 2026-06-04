"""Verifier for encoded jeff modules.

The verifier intentionally works on the Cap'n Proto representation instead of
the higher-level readers because its job is to detect malformed programs rather
than assume that they can be read safely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from jeff.capnp import load_schema


SCHEMA_VERSION = (0, 2, 0)
INT_BINARY_OPS = {
    "add",
    "sub",
    "mul",
    "divS",
    "divU",
    "pow",
    "and",
    "or",
    "xor",
    "minS",
    "minU",
    "maxS",
    "maxU",
    "remS",
    "remU",
    "shl",
    "shr",
}
INT_COMPARE_OPS = {"eq", "ltS", "lteS", "ltU", "lteU"}
INT_UNARY_OPS = {"not", "abs"}
FLOAT_BINARY_OPS = {"add", "sub", "mul", "pow", "atan2", "max", "min"}
FLOAT_COMPARE_OPS = {"eq", "lt", "lte"}
FLOAT_UNARY_OPS = {
    "sqrt",
    "abs",
    "ceil",
    "floor",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
}
FLOAT_PREDICATE_OPS = {"isNan", "isInf"}


@dataclass(frozen=True)
class Diagnostic:
    """A verifier diagnostic."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True)
class ValueType:
    """Comparable representation of jeff value types."""

    kind: str
    width: int | None = None
    length: int | None = None

    @property
    def is_linear(self) -> bool:
        return self.kind in {"qubit", "qureg"}

    def compatible_with(self, expected: ValueType) -> bool:
        if expected.kind == "any":
            return True
        if self.kind != expected.kind:
            return False
        if expected.width is not None and self.width != expected.width:
            return False
        if expected.length is not None and self.length not in {None, expected.length}:
            return False
        return True

    def __str__(self) -> str:
        if self.kind in {"int", "float", "intArray", "floatArray"}:
            suffix = "" if self.width is None else str(self.width)
            if self.kind.endswith("Array"):
                length = "?" if self.length is None else str(self.length)
                return f"{self.kind}{suffix}[{length}]"
            return f"{self.kind}{suffix}"
        return self.kind


ANY = ValueType("any")
QUBIT = ValueType("qubit")
QUREG = ValueType("qureg")
INT1 = ValueType("int", 1)
INT32 = ValueType("int", 32)
FLOAT = ValueType("float")
VAR_QUBITS = ValueType("varQubits")


def validate_file(path: str | Path) -> list[Diagnostic]:
    """Validate a jeff file."""

    schema = load_schema()
    try:
        with Path(path).open("rb") as file:
            module = schema.Module.read(file)
    except Exception as exc:  # noqa: BLE001 - verifier must surface parser errors
        return [Diagnostic(str(path), f"failed to read jeff module: {exc}")]
    return validate_module(module, source=str(path))


def validate_module(module: Any, *, source: str = "<module>") -> list[Diagnostic]:
    """Validate a loaded Cap'n Proto jeff module."""

    diagnostics: list[Diagnostic] = []
    functions = list(module.functions)
    strings = list(module.strings)
    version = (module.version, module.versionMinor, module.versionPatch)

    if version != SCHEMA_VERSION:
        diagnostics.append(
            Diagnostic(
                source,
                f"unsupported schema version {version[0]}.{version[1]}.{version[2]} "
                f"(expected {SCHEMA_VERSION[0]}.{SCHEMA_VERSION[1]}.{SCHEMA_VERSION[2]})",
            )
        )
    if not functions:
        diagnostics.append(
            Diagnostic(source, "module must define at least one function")
        )
        return diagnostics
    if module.entrypoint >= len(functions):
        diagnostics.append(
            Diagnostic(
                source,
                f"entrypoint index {module.entrypoint} is outside function table of size {len(functions)}",
            )
        )

    seen_names: set[str] = set()
    for index, function in enumerate(functions):
        path = f"{source}.functions[{index}]"
        name = _string_at(strings, function.name, f"{path}.name", diagnostics)
        if name in seen_names:
            diagnostics.append(Diagnostic(path, f"duplicate function name {name!r}"))
        seen_names.add(name)

        if function.which() == "definition":
            definition = function.definition
            values = [
                _read_type(value.type, f"{path}.values[{i}]", diagnostics)
                for i, value in enumerate(definition.values)
            ]
            _validate_region(
                definition.body,
                values,
                functions,
                strings,
                f"{path}.body",
                diagnostics,
            )
        else:
            _function_signature(function, path, diagnostics)

    return diagnostics


def _validate_region(
    region: Any,
    values: Sequence[ValueType],
    functions: Sequence[Any],
    strings: Sequence[str],
    path: str,
    diagnostics: list[Diagnostic],
) -> tuple[list[ValueType], list[ValueType]]:
    sources = _index_list(region.sources)
    targets = _index_list(region.targets)
    available = set(sources)
    produced: dict[int, int] = {}
    consumed: dict[int, int] = {}

    for value_id in sources:
        if _check_value_index(value_id, values, f"{path}.sources", diagnostics):
            produced[value_id] = produced.get(value_id, 0) + 1

    for op_index, operation in enumerate(region.operations):
        op_path = f"{path}.operations[{op_index}]"
        inputs = _index_list(operation.inputs)
        outputs = _index_list(operation.outputs)

        for value_id in inputs:
            if _check_value_index(value_id, values, f"{op_path}.inputs", diagnostics):
                consumed[value_id] = consumed.get(value_id, 0) + 1
                if value_id not in available:
                    diagnostics.append(
                        Diagnostic(
                            op_path, f"value {value_id} is used before it is defined"
                        )
                    )

        for value_id in outputs:
            if _check_value_index(value_id, values, f"{op_path}.outputs", diagnostics):
                if value_id in available:
                    diagnostics.append(
                        Diagnostic(
                            op_path, f"value {value_id} is produced more than once"
                        )
                    )
                produced[value_id] = produced.get(value_id, 0) + 1

        _validate_operation(operation, values, functions, strings, op_path, diagnostics)
        available.update(outputs)

    for value_id in targets:
        if _check_value_index(value_id, values, f"{path}.targets", diagnostics):
            consumed[value_id] = consumed.get(value_id, 0) + 1
            if value_id not in available:
                diagnostics.append(
                    Diagnostic(
                        path,
                        f"target value {value_id} is returned before it is defined",
                    )
                )

    _validate_linear_values(values, produced, consumed, path, diagnostics)
    return _types_for(sources, values), _types_for(targets, values)


def _validate_operation(
    operation: Any,
    values: Sequence[ValueType],
    functions: Sequence[Any],
    strings: Sequence[str],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    instruction = operation.instruction
    family = instruction.which()
    inputs = _types_for(operation.inputs, values)
    outputs = _types_for(operation.outputs, values)

    if family == "qubit":
        expected_inputs, expected_outputs = _qubit_signature(
            instruction.qubit, path, diagnostics
        )
        _check_signature(
            inputs, outputs, expected_inputs, expected_outputs, path, diagnostics
        )
    elif family == "qureg":
        expected_inputs, expected_outputs = _qureg_signature(instruction.qureg)
        _check_signature(
            inputs, outputs, expected_inputs, expected_outputs, path, diagnostics
        )
    elif family == "int":
        _validate_int_op(instruction.int, inputs, outputs, path, diagnostics)
    elif family == "intArray":
        _validate_int_array_op(instruction.intArray, inputs, outputs, path, diagnostics)
    elif family == "float":
        _validate_float_op(instruction.float, inputs, outputs, path, diagnostics)
    elif family == "floatArray":
        _validate_float_array_op(
            instruction.floatArray, inputs, outputs, path, diagnostics
        )
    elif family == "func":
        _validate_func_op(
            instruction.func, inputs, outputs, functions, path, diagnostics
        )
    elif family == "scf":
        _validate_scf_op(
            instruction.scf,
            inputs,
            outputs,
            values,
            functions,
            strings,
            path,
            diagnostics,
        )


def _qubit_signature(
    qubit_op: Any, path: str, diagnostics: list[Diagnostic]
) -> tuple[list[ValueType], list[ValueType]]:
    op = qubit_op.which()
    if op == "alloc":
        return [], [QUBIT]
    if op in {"free", "freeZero"}:
        return [QUBIT], []
    if op == "measure":
        return [QUBIT], [INT1]
    if op == "measureNd":
        return [QUBIT], [QUBIT, INT1]
    if op == "reset":
        return [QUBIT], [QUBIT]
    if op != "gate":
        diagnostics.append(Diagnostic(path, f"unsupported qubit operation {op!r}"))
        return [], []

    gate = qubit_op.gate
    kind = gate.which()
    controls = int(gate.controlQubits)
    if kind == "wellKnown":
        gate_inputs, float_inputs = _well_known_gate_shape(
            str(gate.wellKnown), path, diagnostics
        )
    elif kind == "ppr":
        gate_inputs, float_inputs = len(gate.ppr.pauliString), 1
    else:
        gate_inputs, float_inputs = (
            int(gate.custom.numQubits),
            int(gate.custom.numParams),
        )
    return [QUBIT] * (gate_inputs + controls) + [FLOAT] * float_inputs, [QUBIT] * (
        gate_inputs + controls
    )


def _well_known_gate_shape(
    name: str, path: str, diagnostics: list[Diagnostic]
) -> tuple[int, int]:
    if name == "gphase":
        return 0, 1
    if name in {"i", "x", "y", "z", "s", "t", "h"}:
        return 1, 0
    if name in {"r1", "rx", "ry", "rz"}:
        return 1, 1
    if name == "u":
        return 1, 3
    if name == "swap":
        return 2, 0
    diagnostics.append(Diagnostic(path, f"unknown well-known gate {name!r}"))
    return 0, 0


def _qureg_signature(qureg_op: Any) -> tuple[list[ValueType], list[ValueType]]:
    op = qureg_op.which()
    if op == "alloc":
        return [INT32], [QUREG]
    if op in {"free", "freeZero"}:
        return [QUREG], []
    if op == "extractIndex":
        return [QUREG, INT32], [QUREG, QUBIT]
    if op == "insertIndex":
        return [QUREG, INT32, QUBIT], [QUREG]
    if op == "extractSlice":
        return [QUREG, INT32, INT32], [QUREG, QUREG]
    if op == "insertSlice":
        return [QUREG, INT32, QUREG], [QUREG]
    if op == "length":
        return [QUREG], [QUREG, INT32]
    if op == "split":
        return [QUREG, INT32], [QUREG, QUREG]
    if op == "join":
        return [QUREG, QUREG], [QUREG]
    if op == "create":
        return [VAR_QUBITS], [QUREG]
    return [], []


def _validate_int_op(
    int_op: Any,
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    op = int_op.which()
    const_widths = {
        "const1": 1,
        "const8": 8,
        "const16": 16,
        "const32": 32,
        "const64": 64,
    }
    if op in const_widths:
        _check_signature(
            inputs, outputs, [], [ValueType("int", const_widths[op])], path, diagnostics
        )
    elif op in INT_BINARY_OPS:
        _check_same_numeric(inputs, outputs, "int", 2, path, diagnostics)
    elif op in INT_COMPARE_OPS:
        _check_same_numeric(inputs, outputs, "int", 2, path, diagnostics, result=INT1)
    elif op in INT_UNARY_OPS:
        _check_same_numeric(inputs, outputs, "int", 1, path, diagnostics)


def _validate_float_op(
    float_op: Any,
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    op = float_op.which()
    if op == "const32":
        _check_signature(
            inputs, outputs, [], [ValueType("float", 32)], path, diagnostics
        )
    elif op == "const64":
        _check_signature(
            inputs, outputs, [], [ValueType("float", 64)], path, diagnostics
        )
    elif op in FLOAT_BINARY_OPS:
        _check_same_numeric(inputs, outputs, "float", 2, path, diagnostics)
    elif op in FLOAT_COMPARE_OPS:
        _check_same_numeric(inputs, outputs, "float", 2, path, diagnostics, result=INT1)
    elif op in FLOAT_UNARY_OPS:
        _check_same_numeric(inputs, outputs, "float", 1, path, diagnostics)
    elif op in FLOAT_PREDICATE_OPS:
        _check_same_numeric(inputs, outputs, "float", 1, path, diagnostics, result=INT1)


def _validate_int_array_op(
    array_op: Any,
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    op = array_op.which()
    const_widths = {
        "const1": 1,
        "const8": 8,
        "const16": 16,
        "const32": 32,
        "const64": 64,
    }
    if op in const_widths:
        _check_signature(
            inputs,
            outputs,
            [],
            [ValueType("intArray", const_widths[op])],
            path,
            diagnostics,
        )
    elif op == "zero":
        _check_signature(
            inputs,
            outputs,
            [INT32],
            [ValueType("intArray", int(array_op.zero))],
            path,
            diagnostics,
        )
    elif op == "getIndex" and len(inputs) == 2 and inputs[0].kind == "intArray":
        _check_signature(
            inputs,
            outputs,
            [ValueType("intArray", inputs[0].width), INT32],
            [ValueType("int", inputs[0].width)],
            path,
            diagnostics,
        )
    elif op == "setIndex" and len(inputs) == 3 and inputs[0].kind == "intArray":
        _check_signature(
            inputs,
            outputs,
            [ValueType("intArray", inputs[0].width), INT32, ValueType("int")],
            [inputs[0]],
            path,
            diagnostics,
        )
    elif op == "length":
        _check_signature(
            inputs, outputs, [ValueType("intArray")], [INT32], path, diagnostics
        )
    elif op == "create":
        _check_create_array(inputs, outputs, "int", "intArray", path, diagnostics)


def _validate_float_array_op(
    array_op: Any,
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    op = array_op.which()
    if op == "const32":
        _check_signature(
            inputs, outputs, [], [ValueType("floatArray", 32)], path, diagnostics
        )
    elif op == "const64":
        _check_signature(
            inputs, outputs, [], [ValueType("floatArray", 64)], path, diagnostics
        )
    elif op == "zero":
        _check_signature(
            inputs,
            outputs,
            [INT32],
            [ValueType("floatArray", _precision_width(array_op.zero))],
            path,
            diagnostics,
        )
    elif op == "getIndex" and len(inputs) == 2 and inputs[0].kind == "floatArray":
        _check_signature(
            inputs,
            outputs,
            [ValueType("floatArray", inputs[0].width), INT32],
            [ValueType("float", inputs[0].width)],
            path,
            diagnostics,
        )
    elif op == "setIndex" and len(inputs) == 3 and inputs[0].kind == "floatArray":
        _check_signature(
            inputs,
            outputs,
            [
                ValueType("floatArray", inputs[0].width),
                INT32,
                ValueType("float", inputs[0].width),
            ],
            [inputs[0]],
            path,
            diagnostics,
        )
    elif op == "length":
        _check_signature(
            inputs, outputs, [ValueType("floatArray")], [INT32], path, diagnostics
        )
    elif op == "create":
        _check_create_array(inputs, outputs, "float", "floatArray", path, diagnostics)


def _validate_func_op(
    func_op: Any,
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    functions: Sequence[Any],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    func_index = int(func_op.funcCall)
    if func_index >= len(functions):
        diagnostics.append(
            Diagnostic(
                path, f"function call index {func_index} is outside function table"
            )
        )
        return
    expected_inputs, expected_outputs = _function_signature(
        functions[func_index], f"{path}.funcCall", diagnostics
    )
    _check_signature(
        inputs, outputs, expected_inputs, expected_outputs, path, diagnostics
    )


def _validate_scf_op(
    scf_op: Any,
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    values: Sequence[ValueType],
    functions: Sequence[Any],
    strings: Sequence[str],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    op = scf_op.which()
    if op == "for":
        if len(inputs) < 3:
            diagnostics.append(
                Diagnostic(path, "for loop requires start, stop, and step inputs")
            )
            return
        _check_signature(
            inputs[:3], [], [inputs[0], inputs[0], inputs[0]], [], path, diagnostics
        )
        state = list(inputs[3:])
        _check_signature(
            inputs, outputs, list(inputs[:3]) + state, state, path, diagnostics
        )
        region_inputs, region_outputs = _validate_region(
            getattr(scf_op, "for"),
            values,
            functions,
            strings,
            f"{path}.for",
            diagnostics,
        )
        _check_signature(
            region_inputs,
            region_outputs,
            [inputs[0], *state],
            state,
            f"{path}.for",
            diagnostics,
        )
    elif op == "while":
        state = list(inputs)
        _check_signature(inputs, outputs, state, state, path, diagnostics)
        while_op = getattr(scf_op, "while")
        condition_inputs, condition_outputs = _validate_region(
            while_op.condition,
            values,
            functions,
            strings,
            f"{path}.while.condition",
            diagnostics,
        )
        body_inputs, body_outputs = _validate_region(
            while_op.body,
            values,
            functions,
            strings,
            f"{path}.while.body",
            diagnostics,
        )
        _check_signature(
            condition_inputs,
            condition_outputs,
            state,
            [INT1],
            f"{path}.while.condition",
            diagnostics,
        )
        _check_signature(
            body_inputs, body_outputs, state, state, f"{path}.while.body", diagnostics
        )
    elif op == "doWhile":
        state = list(inputs)
        _check_signature(inputs, outputs, state, state, path, diagnostics)
        body_inputs, body_outputs = _validate_region(
            scf_op.doWhile.body,
            values,
            functions,
            strings,
            f"{path}.doWhile.body",
            diagnostics,
        )
        condition_inputs, condition_outputs = _validate_region(
            scf_op.doWhile.condition,
            values,
            functions,
            strings,
            f"{path}.doWhile.condition",
            diagnostics,
        )
        _check_signature(
            body_inputs, body_outputs, state, state, f"{path}.doWhile.body", diagnostics
        )
        _check_signature(
            condition_inputs,
            condition_outputs,
            state,
            [INT1],
            f"{path}.doWhile.condition",
            diagnostics,
        )
    elif op == "switch":
        if not inputs or inputs[0].kind != "int":
            diagnostics.append(
                Diagnostic(path, "switch first input must be an integer selector")
            )
        state = list(inputs[1:])
        for index, branch in enumerate(scf_op.switch.branches):
            branch_inputs, branch_outputs = _validate_region(
                branch,
                values,
                functions,
                strings,
                f"{path}.switch.branches[{index}]",
                diagnostics,
            )
            _check_signature(
                branch_inputs,
                branch_outputs,
                state,
                list(outputs),
                f"{path}.switch.branches[{index}]",
                diagnostics,
            )


def _function_signature(
    function: Any, path: str, diagnostics: list[Diagnostic]
) -> tuple[list[ValueType], list[ValueType]]:
    if function.which() == "definition":
        values = [
            _read_type(value.type, f"{path}.values[{i}]", diagnostics)
            for i, value in enumerate(function.definition.values)
        ]
        return _types_for(function.definition.body.sources, values), _types_for(
            function.definition.body.targets, values
        )
    return (
        [
            _read_type(value.type, f"{path}.inputs[{i}]", diagnostics)
            for i, value in enumerate(function.declaration.inputs)
        ],
        [
            _read_type(value.type, f"{path}.outputs[{i}]", diagnostics)
            for i, value in enumerate(function.declaration.outputs)
        ],
    )


def _check_same_numeric(
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    kind: str,
    arity: int,
    path: str,
    diagnostics: list[Diagnostic],
    *,
    result: ValueType | None = None,
) -> None:
    if len(inputs) != arity or len(outputs) != 1:
        diagnostics.append(
            Diagnostic(
                path,
                f"expected {arity} input(s) and 1 output, got {len(inputs)} and {len(outputs)}",
            )
        )
        return
    if any(value.kind != kind for value in inputs):
        diagnostics.append(Diagnostic(path, f"expected all inputs to be {kind} values"))
        return
    width = inputs[0].width
    if any(value.width != width for value in inputs):
        diagnostics.append(
            Diagnostic(path, f"{kind} operands must have the same precision/bitwidth")
        )
    expected_result = result if result is not None else ValueType(kind, width)
    if not outputs[0].compatible_with(expected_result):
        diagnostics.append(
            Diagnostic(path, f"expected output {expected_result}, got {outputs[0]}")
        )


def _check_create_array(
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    scalar_kind: str,
    array_kind: str,
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    if len(outputs) != 1:
        diagnostics.append(Diagnostic(path, f"{array_kind}.create expects one output"))
        return
    if not inputs:
        diagnostics.append(
            Diagnostic(path, f"{array_kind}.create expects at least one input")
        )
        return
    width = inputs[0].width
    if any(value.kind != scalar_kind or value.width != width for value in inputs):
        diagnostics.append(
            Diagnostic(
                path,
                f"{array_kind}.create inputs must be same-width {scalar_kind} values",
            )
        )
    if not outputs[0].compatible_with(ValueType(array_kind, width)):
        diagnostics.append(
            Diagnostic(
                path,
                f"expected output {ValueType(array_kind, width)}, got {outputs[0]}",
            )
        )


def _check_signature(
    inputs: Sequence[ValueType],
    outputs: Sequence[ValueType],
    expected_inputs: Sequence[ValueType],
    expected_outputs: Sequence[ValueType],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    if len(expected_inputs) == 1 and expected_inputs[0] == VAR_QUBITS:
        if any(value.kind != "qubit" for value in inputs):
            diagnostics.append(Diagnostic(path, "expected all inputs to be qubits"))
    elif len(inputs) != len(expected_inputs):
        diagnostics.append(
            Diagnostic(
                path, f"expected {len(expected_inputs)} input(s), got {len(inputs)}"
            )
        )
    else:
        for index, (actual, expected) in enumerate(
            zip(inputs, expected_inputs, strict=True)
        ):
            if not actual.compatible_with(expected):
                diagnostics.append(
                    Diagnostic(path, f"input {index} expected {expected}, got {actual}")
                )

    if len(outputs) != len(expected_outputs):
        diagnostics.append(
            Diagnostic(
                path, f"expected {len(expected_outputs)} output(s), got {len(outputs)}"
            )
        )
    else:
        for index, (actual, expected) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            if not actual.compatible_with(expected):
                diagnostics.append(
                    Diagnostic(
                        path, f"output {index} expected {expected}, got {actual}"
                    )
                )


def _validate_linear_values(
    values: Sequence[ValueType],
    produced: dict[int, int],
    consumed: dict[int, int],
    path: str,
    diagnostics: list[Diagnostic],
) -> None:
    linear_ids = set(produced) | set(consumed)
    for value_id in sorted(linear_ids):
        if value_id >= len(values) or not values[value_id].is_linear:
            continue
        produced_count = produced.get(value_id, 0)
        consumed_count = consumed.get(value_id, 0)
        if produced_count != 1:
            diagnostics.append(
                Diagnostic(
                    path,
                    f"linear value {value_id} must be produced exactly once, got {produced_count}",
                )
            )
        if consumed_count != 1:
            diagnostics.append(
                Diagnostic(
                    path,
                    f"linear value {value_id} must be consumed exactly once, got {consumed_count}",
                )
            )


def _read_type(type_reader: Any, path: str, diagnostics: list[Diagnostic]) -> ValueType:
    kind = type_reader.which()
    if kind == "qubit":
        return QUBIT
    if kind == "qureg":
        qureg = type_reader.qureg
        return ValueType(
            "qureg", length=int(qureg.static) if qureg.which() == "static" else None
        )
    if kind == "int":
        return ValueType("int", int(type_reader.int))
    if kind == "intArray":
        array = type_reader.intArray
        return ValueType("intArray", int(array.bitwidth), _array_length(array.length))
    if kind == "float":
        return ValueType("float", _precision_width(type_reader.float))
    if kind == "floatArray":
        array = type_reader.floatArray
        return ValueType(
            "floatArray", _precision_width(array.precision), _array_length(array.length)
        )
    diagnostics.append(Diagnostic(path, f"unknown value type {kind!r}"))
    return ANY


def _precision_width(precision: Any) -> int:
    return 32 if str(precision) == "float32" else 64


def _array_length(length: Any) -> int | None:
    return int(length.static) if length.which() == "static" else None


def _types_for(indices: Iterable[int], values: Sequence[ValueType]) -> list[ValueType]:
    return [values[index] for index in indices if 0 <= index < len(values)]


def _index_list(values: Iterable[int]) -> list[int]:
    return [int(value) for value in values]


def _check_value_index(
    value_id: int, values: Sequence[ValueType], path: str, diagnostics: list[Diagnostic]
) -> bool:
    if value_id >= len(values):
        diagnostics.append(
            Diagnostic(
                path,
                f"value index {value_id} is outside values table of size {len(values)}",
            )
        )
        return False
    return True


def _string_at(
    strings: Sequence[str], index: int, path: str, diagnostics: list[Diagnostic]
) -> str:
    if index >= len(strings):
        diagnostics.append(
            Diagnostic(
                path,
                f"string index {index} is outside string table of size {len(strings)}",
            )
        )
        return f"<invalid:{index}>"
    return strings[index]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate encoded jeff modules.")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="One or more .jeff files to validate."
    )
    args = parser.parse_args(argv)

    diagnostics: list[Diagnostic] = []
    for path in args.paths:
        diagnostics.extend(validate_file(path))

    for diagnostic in diagnostics:
        print(diagnostic)
    return 1 if diagnostics else 0
