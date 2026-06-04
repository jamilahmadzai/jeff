from __future__ import annotations

from pathlib import Path

from jeff.capnp import load_schema

from tools.verifier import validate_module
from tools.verifier.verifier import main


def _schema():
    return load_schema()


def _module(
    type_specs: list[tuple[str, int | None]],
    *,
    sources: list[int],
    targets: list[int],
    op_count: int,
):
    schema = _schema()
    module = schema.Module.new_message()
    module.version = 0
    module.versionMinor = 2
    module.versionPatch = 0
    module.entrypoint = 0

    strings = module.init("strings", 1)
    strings[0] = "main"

    function = module.init("functions", 1)[0]
    function.name = 0
    definition = function.init("definition")
    values = definition.init("values", len(type_specs))
    for index, (kind, width) in enumerate(type_specs):
        _set_type(values[index].type, kind, width)

    body = definition.init("body")
    body_sources = body.init("sources", len(sources))
    for index, value_id in enumerate(sources):
        body_sources[index] = value_id
    body_targets = body.init("targets", len(targets))
    for index, value_id in enumerate(targets):
        body_targets[index] = value_id
    operations = body.init("operations", op_count)
    return module, operations


def _set_type(type_builder, kind: str, width: int | None = None) -> None:
    if kind == "qubit":
        type_builder.qubit = None
    elif kind == "qureg":
        type_builder.init("qureg").dynamic = None
    elif kind == "int":
        type_builder.int = width or 32
    elif kind == "float":
        type_builder.float = "float32" if width == 32 else "float64"
    else:
        raise AssertionError(f"unsupported test type {kind}")


def _set_values(builder, field: str, values: list[int]) -> None:
    value_list = builder.init(field, len(values))
    for index, value_id in enumerate(values):
        value_list[index] = value_id


def _int_const(op_builder, output: int, *, width: int = 32) -> None:
    _set_values(op_builder, "outputs", [output])
    int_op = op_builder.instruction.init("int")
    setattr(int_op, f"const{width}", 1)


def _float_const(op_builder, output: int, *, width: int = 64) -> None:
    _set_values(op_builder, "outputs", [output])
    float_op = op_builder.instruction.init("float")
    setattr(float_op, f"const{width}", 1.0)


def _int_add(op_builder, inputs: list[int], output: int) -> None:
    _set_values(op_builder, "inputs", inputs)
    _set_values(op_builder, "outputs", [output])
    op_builder.instruction.init("int").add = None


def _float_add(op_builder, inputs: list[int], output: int) -> None:
    _set_values(op_builder, "inputs", inputs)
    _set_values(op_builder, "outputs", [output])
    op_builder.instruction.init("float").add = None


def test_valid_existing_examples_pass() -> None:
    assert validate_module(_read_example("examples/qubits/qubits.jeff")) == []
    assert (
        validate_module(_read_example("examples/entangled_qs/entangled_qs.jeff")) == []
    )


def test_module_entrypoint_is_checked() -> None:
    module, _ = _module([("int", 32)], sources=[0], targets=[0], op_count=0)
    module.entrypoint = 1

    diagnostics = validate_module(module)

    assert any("entrypoint index 1" in diagnostic.message for diagnostic in diagnostics)


def test_value_use_before_definition_is_checked() -> None:
    module, operations = _module(
        [("int", 32), ("int", 32), ("int", 32)], sources=[], targets=[2], op_count=1
    )
    _int_add(operations[0], [0, 1], 2)

    diagnostics = validate_module(module)

    assert any(
        "value 0 is used before it is defined" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_value_indexes_must_be_in_function_values_table() -> None:
    module, operations = _module([("int", 32)], sources=[], targets=[0], op_count=1)
    _int_add(operations[0], [0, 2], 0)

    diagnostics = validate_module(module)

    assert any(
        "value index 2 is outside values table" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_operation_input_and_output_types_are_checked() -> None:
    module, operations = _module(
        [("int", 32), ("float", 64), ("int", 32)],
        sources=[],
        targets=[2],
        op_count=3,
    )
    _int_const(operations[0], 0)
    _float_const(operations[1], 1)
    _int_add(operations[2], [0, 1], 2)

    diagnostics = validate_module(module)

    assert any(
        "expected all inputs to be int values" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_integer_operation_operands_must_have_same_bitwidth() -> None:
    module, operations = _module(
        [("int", 32), ("int", 64), ("int", 32)],
        sources=[],
        targets=[2],
        op_count=3,
    )
    _int_const(operations[0], 0, width=32)
    _int_const(operations[1], 1, width=64)
    _int_add(operations[2], [0, 1], 2)

    diagnostics = validate_module(module)

    assert any(
        "int operands must have the same precision/bitwidth" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_float_operation_operands_must_have_same_precision() -> None:
    module, operations = _module(
        [("float", 32), ("float", 64), ("float", 32)],
        sources=[],
        targets=[2],
        op_count=3,
    )
    _float_const(operations[0], 0, width=32)
    _float_const(operations[1], 1, width=64)
    _float_add(operations[2], [0, 1], 2)

    diagnostics = validate_module(module)

    assert any(
        "float operands must have the same precision/bitwidth" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_linear_values_must_be_consumed_once() -> None:
    module, _ = _module([("qubit", None)], sources=[0], targets=[], op_count=0)

    diagnostics = validate_module(module)

    assert any(
        "linear value 0 must be consumed exactly once" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_linear_passthrough_is_valid() -> None:
    module, _ = _module([("qubit", None)], sources=[0], targets=[0], op_count=0)

    assert validate_module(module) == []


def test_nested_regions_are_isolated_from_outer_values() -> None:
    module, operations = _module(
        [("int", 32), ("int", 32), ("int", 32), ("int", 32)],
        sources=[],
        targets=[2],
        op_count=3,
    )
    _int_const(operations[0], 0)
    _int_const(operations[1], 1)

    switch_op = operations[2]
    _set_values(switch_op, "inputs", [0, 1])
    _set_values(switch_op, "outputs", [2])
    switch = switch_op.instruction.init("scf").init("switch")
    branches = switch.init("branches", 1)
    branch = branches[0]
    _set_values(branch, "sources", [1])
    _set_values(branch, "targets", [3])
    branch_ops = branch.init("operations", 1)
    _int_add(branch_ops[0], [1, 0], 3)

    diagnostics = validate_module(module)

    assert any(
        "value 0 is used before it is defined" in diagnostic.message
        for diagnostic in diagnostics
    )


def test_cli_reports_invalid_file(tmp_path: Path, capsys) -> None:
    module, _ = _module([("int", 32)], sources=[0], targets=[0], op_count=0)
    module.entrypoint = 4
    path = tmp_path / "bad.jeff"
    with path.open("wb") as file:
        module.write(file)

    exit_code = main([str(path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "entrypoint index 4" in captured.out


def _read_example(path: str):
    schema = _schema()
    with Path(path).open("rb") as file:
        return schema.Module.read(file)
