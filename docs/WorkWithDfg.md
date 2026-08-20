# Working with `dfg-mlir`

First and the foremost, check the [cmake functions](../cmake/MLIRUtils.cmake) if you want to generate the correct `.inc` files from TableGen.
Then, if you want to use the `dfg-mlir` library in your project, make sure you use the correct one in your cmake file, such as `DFGMLIRDfgDialect` for the dialect `dfg`.

## Create Operations

Check [dfg dialect definition](../include/dfg-mlir/Dialect/dfg/IR/Dialect.td) to make sure you understand and will use the same guideline to create ops.

## Use Passes

See [conversion passes](../include/dfg-mlir/Conversion/Passes.td) to understand the definition and implementation of passes.
