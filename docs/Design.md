# Design of `dfg-mlir` project
Here you will find the design of the contents inside this project.

## Dialects
Here two dialects defined in `dfg-mlir` are listed as following. You can find examples of these dialects [here](../test/Dialect/).

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

### `vitis`
This dialect works as an intermediate language to connect `dfg` to the famous Xilinx (now AMD) FPGA design toolchains Vivado/Vitis. Everything in this dialect should be a 1:1 mapping to the C++ expressions or classes.

#### Types
| Type | Sementic |
| :- | :- |
| !vitis.ap_axiu<datawidth, keep, user, dest, tlast> | ap_axiu<DATADIWTH, KEEP, USER, DEST, AXI_ENABLE_LAST> |
| !vitis.ap_axis<datawidth, keep, user, dest, tlast> | ap_axis<DATADIWTH, KEEP, USER, DEST, AXI_ENABLE_LAST> |
| !vitis.stream<!stream_type> | hls::stream<DATATYPE> |

#### Operations
| Operation | Sementic |
| :- | :- |
| vitis.constant | see next oepration |
| vitis.variable (constant)? | C/C++ variable definition (w/ initial value) |
| vitis.update | C/C++ variable update |
| vitis.func | function definition |
| vitis.return | function return |
| vitis.while | while-loop monitoring a boolean value |
| vitis.while_true | an infinite while loop |
| vitis.if_break | an if-block monitoring a boolean value and break out of the loop |

## Passes
Here are all the conversion/lowering passes as well as the transformation passes inside each dialect in `dfg-mlir`.

### Conversion Passes
#### `--insert-olympus-wrappers`
TBD

#### `--convert-dfg-to-async`
This pass should convert `dfg` to `async` dialect, however, it's still under construction.

#### `--convert-dfg-nodes-to-func`
TBD

#### `--convert-dfg-edges-to-llvm`
TBD

#### `--convert-dfg-to-olympus`
TBD

#### `--convert-dfg-to-vitis`
This pass will convert `process` to `vitis.functio˙operations.

### Transformation Passes

#### `dfg` dialect
##### `--dfg-inline-region`
This transformation will inline the contents of some `dfg.region` into the place where it's embedded. Currently it only inlines all regions for FPGA backend, namely strategy **all**. Later a **smart** will be implemented.

##### `--dfg-operator-to-process`
Before converting dfg to other dialects or translating, `operator` must be converted to the `process` with same semantics, which pulls/pushes only once and loops monitoring the input channels.

##### `--print-operator-to-yaml`
This will generate yaml files for each `operator`, which can be utilized as inputs to [Mocasin](https://github.com/tud-ccc/mocasin) project.

#### `vitis` dialect
##### None

## Targets
Here are the translations implemented in this repository. Learn more about the usage in [Examples.td](Examples.md)

### `--dfg-to-vivado-tcl`
This translation will generate a tcl script to be used with vivado command line tool. The contents of this script are based on the connection informations in `dfg.region`.

### `--vitis-to-cpp`
This translation will transpile an MLIR code in `vitis` dialect into a C++ file with HLS directives.

### `--vitis-to-tcl`
This translation will generate a tcl script to be used with vitis_run command line tool with `hls` mode. It will automatically synthesize every `vitis` functions and export IPs for later use.
