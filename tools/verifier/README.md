# jeff verifier

`tools.verifier` validates encoded `.jeff` modules directly from the Cap'n
Proto representation.

Run it from the repository root:

```bash
uv run python -m tools.verifier examples/qubits/qubits.jeff
```

The verifier checks module-level metadata, function entrypoints, value-table
references, operation signatures, dataflow ordering, linear qubit/qureg usage,
and nested region isolation.
