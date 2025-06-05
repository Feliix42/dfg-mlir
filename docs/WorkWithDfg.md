# Working with `dfg-mlir`

I know, sometimes, you have to work with other people's project.
You may start to think, why on earth I'm gonna integrate you shit into mine. Well, life is hard and grow up. I don't care what is deep in your mind. But if you want to use `dfg-mlir` as source or target of your own project, you'd better know some basics from this doc.

The first and the foremost, check the [cmake functions](../cmake/MLIRUtils.cmake) if you want to generate the correct `.inc` files from TableGen.
Then, if you want to use the `dfg-mlir` library in your project, make sure you use the correct one in your cmake file, such as `DFGMLIRDfgDialect` for the dialect `dfg`.

## Create Operations

Check [dfg dialect definition](../include/dfg-mlir/Dialect/dfg/IR/Dialect.td) and [emitHLS arith operation definition](../include/dfg-mlir/Dialect/emitHLS/IR/ArithOps.td) to make sure you understand and will use the same guideline to create ops.

## Use Passes

See [conversion passes](../include/dfg-mlir/Conversion/Passes.td) and [emitHLS transformation passes](../include/dfg-mlir/Dialect/emitHLS/Transforms/Passes.td) to understand the definition and implementation of passes.
