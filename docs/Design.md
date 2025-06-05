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

### `emitHLS`
This dialect works as an intermediate language to connect `dfg` to the famous Xilinx (now AMD) FPGA design toolchains Vivado/emitHLS. Everything in this dialect should be a 1:1 mapping to the C++ expressions or classes.

#### Types
| Type | Sementic |
| :- | :- |
| !emitHLS.ap_axiu<datawidth, keep, user, dest, tlast> | ap_axiu<DATADIWTH, KEEP, USER, DEST, AXI_ENABLE_LAST> |
| !emitHLS.ap_axis<datawidth, keep, user, dest, tlast> | ap_axis<DATADIWTH, KEEP, USER, DEST, AXI_ENABLE_LAST> |
| !emitHLS.stream<!stream_type> | hls::stream<DATATYPE> |

#### Operations
| Operation | Sementic |
| :- | :- |
| emitHLS.constant | see next oepration |
| emitHLS.variable (constant)? | C/C++ variable definition (w/ initial value) |
| emitHLS.update | C/C++ variable update |
| emitHLS.func | function definition |
| emitHLS.return | function return |
| emitHLS.while | while-loop monitoring a boolean value |
| emitHLS.while_true | an infinite while loop |
| emitHLS.if_break | an if-block monitoring a boolean value and break out of the loop |

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

#### `--convert-dfg-to-emitHLS`
This pass will convert `process` to `emitHLS.functio˙operations.

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

#### `emitHLS` dialect
##### None

## Targets
Here are the translations implemented in this repository. Learn more about the usage in [Examples.td](Examples.md)

### `--emitHLS-generate-project`
This translation will generate the files in the current folder by default that are needed to generate an FPGA design targeting Kria KV260 SOM by default with emitHLS/Vivado.
The contents within the folder are:

1. `main.cpp`: A emitHLS HLS compatible C++ code with pragmas
2. `run_hls.tcl`: A Tcl script to automatically run HLS and package the IP
3. `run_vivado.tcl`: A Tcl script to automatically create a block design with IPs and generate the hardware.
4. `run_design.sh`: A shell script that automatically run the full process to get the `.bit` and `.hwh` files for PYNQ usage.

If one needs to use a custom directory to generate these files and/or uses a different FPGA target, two command line options are provided:

1. `--output-dir`: defines the output directory, please use abosulte path (speicial path symbols such as `~` is not supported)
2. `--target-device`: defines the targeted device name, such as for Kira KV260 SOM it's `xck26-sfvc784-2LV-c`.