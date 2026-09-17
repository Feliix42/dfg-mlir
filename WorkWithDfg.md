# Working with dfg-mlir

This guide covers integrating `dfg-mlir` into your project and creating DFG operations and passes.

## Prerequisites

For TableGen file generation, see [cmake/MLIRUtils.cmake](cmake/MLIRUtils.cmake).

To use the library in your CMake project, link against the appropriate target. For the `dfg` dialect:

```cmake
find_package(dfg-mlir REQUIRED)
target_link_libraries(your_target PRIVATE DFGMLIRDfgDialect)
```

## Creating Operations

The `dfg` dialect is defined in [include/dfg-mlir/Dialect/dfg/IR/Dialect.td](include/dfg-mlir/Dialect/dfg/IR/Dialect.td). Follow the same patterns when defining new operations.

### Example: Defining a Process

```mlir
// A simple process that sums two inputs
dfg.process @sum
    inputs (%a: i32, %b: i32)
    outputs (%result: i32)
{
    dfg.loop inputs(%a: i32, %b: i32) {
        %val_a = dfg.pull %a : i32
        %val_b = dfg.pull %b : i32
        %sum = arith.addi %val_a, %val_b : i32
        dfg.push(%sum) %result : i32
    }
}
```

### Example: Connecting Processes with Channels

```mlir
// Define a region with connected processes
dfg.region @run_pipeline inputs(%in: i32) outputs(%out: i32)
{
    // Create channels
    %in_chan_in, %in_chan_out = dfg.channel() : i32
    %out_chan_in, %out_chan_out = dfg.channel() : i32

    // Connect external I/O to channels
    dfg.connect.input %in, %in_chan_in : i32
    dfg.connect.output %out, %out_chan_out : i32

    // Instantiate and connect processes
    dfg.instantiate @sum inputs(%in_chan_out, %in_chan_out) outputs(%out_chan_in) 
        : (i32, i32) -> (i32)
}
```

## Using Passes

Pass definitions are in [include/dfg-mlir/Conversion/Passes.td](include/dfg-mlir/Conversion/Passes.td).

### Running a Pass

```bash
# Print the graph structure
dfg-opt --dfg-print-graph input.mlir

# Convert operators to processes
dfg-opt --dfg-operator-to-process input.mlir -o output.mlir
```

### Example: Full Lowering Pipeline

```bash
# Convert DFG to LLVM
dfg-opt input.mlir \
    --dfg-operator-to-process \
    --convert-dfg-to-llvm \
    -o output.ll
```

## Key Concepts

- **Channels**: Typed FIFO buffers between nodes. Can be bounded or unbounded.
- **Processes**: Infinite loops that pull from inputs, compute, and push to outputs.
- **Regions**: Containers for connected graphs of processes and channels.
- **Nodes**: Units of parallel execution, scheduled by the OS via OpenMP.

For more examples, see [test/Dialect/dfg/](test/Dialect/dfg/).
