# Design of `dfg-mlir` project
Here you will find the design of the contents inside this project.

## Dialects
Here the dialects defined in `dfg-mlir` are listed as following. You can find examples of these dialects [here](../test/Dialect/).

### `dfg`
This is the main dialect in this project, which represents a Data-Flow Graph (DFG) at Kahn Process Network (KPN) level. To ensure the determinism character of KPN we defined the following types and operations.

#### Types
There are two types in `dfg` dialect (see below), they both encapsulate an element type, which can be any other types from upstream or user-defined dialects. These two types are used by all the `dfg` operations, which you'll see later in details.

| Type | Sementic |
| :- | :- |
| !dfg.input<!ElementType> | The input port of an FIFO channel |
| !dfg.output<!ElementType> | The output port of an FIFO channel |

#### Operations
Here you can find all the operations defined in `dfg` dialect. The interoperability, e.g. creation of certain operation please see information in [WorkWithDfg.md](WorkWithDfg.md).


## Passes
Here are all the conversion/lowering passes as well as the transformation passes inside each dialect in `dfg-mlir`.

### Conversion Passes
#### `--insert-olympus-wrappers`
TBD

#### `--convert-dfg-nodes-to-func`
TBD

#### `--convert-dfg-edges-to-llvm`
TBD

#### `--convert-dfg-to-olympus`
TBD


### Transformation Passes

#### `dfg` dialect
##### `--dfg-inline-region`
This transformation will inline the contents of some `dfg.region` into the place where it's embedded. Currently it only inlines all regions for FPGA backend, namely strategy **all**. Later a **smart** will be implemented.

##### `--dfg-operator-to-process`
Before converting dfg to other dialects or translating, `operator` must be converted to the `process` with same semantics, which pulls/pushes only once and loops monitoring the input channels.

##### `--dfg-print-graph`
This pass will print to a dot file (or use the option print-to-pdf=1 to pdf file) of the graph(s) you defined using dfg dialect. For `print-to-pdf` option, make sure you installed `dot` and `inkscape`, and they're in the `PATH`.

##### `--print-operator-to-yaml`
This will generate yaml files for each `operator`, which can be utilized as inputs to [Mocasin](https://github.com/tud-ccc/mocasin) project.

