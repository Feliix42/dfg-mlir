# Design

`dfg-mlir` provides dialects for dataflow graph representation at the Kahn Process Network (KPN) level.

See [test/Dialect/](test/Dialect/) for usage examples.

## Dialects

### `dfg`

The main dialect, representing Data-Flow Graphs at KPN level with deterministic semantics.

#### Types

Two port types encapsulate any element type:

| Type | Semantic |
|------|----------|
| `!dfg.input<ElementType>` | Sending end of a channel (data is pushed to this end) |
| `!dfg.output<ElementType>` | Receiving end of a channel (data is pulled from this end) |

All `dfg` operations use these types.

#### Operations

See [WorkWithDfg.md](WorkWithDfg.md) for operation creation and interoperability.

## Passes

### Conversion Passes

Lower DFG to other dialects or LLVM:

| Pass | Description |
|------|-------------|
| `--insert-olympus-wrappers` | TBD |
| `--convert-dfg-nodes-to-func` | TBD |
| `--convert-dfg-edges-to-llvm` | TBD |
| `--convert-dfg-to-olympus` | TBD |

### Transformation Passes

`dfg` dialect transformations:

#### `--dfg-inline-region`

Inlines `dfg.region` contents at the embedding site. Currently uses strategy **all** for FPGA. Strategy **smart** planned.


#### `--dfg-operator-to-process`

Converts `operator` to `process` with equivalent semantics: single pull/push with input channel monitoring loop.


#### `--dfg-print-graph`

Prints the graph to a DOT file. Use `print-to-pdf=1` for PDF output (requires `dot` and `inkscape` in PATH).


#### `--print-operator-to-yaml`

Generates YAML files for each `operator` for use with [Mocasin](https://github.com/tud-ccc/mocasin).
