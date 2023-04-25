//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_OPS
#define DFG_OPS

include "dfg-mlir/Dialect/dfg/IR/Types.td"
include "dfg-mlir/Dialect/dfg/IR/Base.td"


def Dfg_OperatorOp : Dfg_Op<"operator", []> {
    let summary = "Defines a node in the Dataflow Graph";
    let description = [{
        The `operator` operation produces a dataflow node which is linked to other
        operators by hooking up their input and output ports.
        This is done using the `instanciate` function.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    // let arguments = (ins "Type":$value);
    // let results = (outs "Type":$result);
}

#endif
