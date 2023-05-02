//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_OPS
#define DFG_OPS

include "dfg-mlir/Dialect/dfg/IR/Types.td"
include "dfg-mlir/Dialect/dfg/IR/Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// def Dfg_OperatorOp : Dfg_Op<"operator", []> {
//     let summary = "Defines a node in the Dataflow Graph";
//     let description = [{
//         The `operator` operation produces a dataflow node which is linked to other
//         operators by hooking up their input and output ports.
//         This is done using the `instanciate` function.

//         Syntax:

//         ```
//         TODO
//         ```

//         Examples:

//         ```
//         TODO
//         ```
//     }];

//     // let arguments = (ins "Type":$value);
//     // let results = (outs "Type":$result);
// }

def Dfg_ChannelOp : Dfg_Op<
        "channel",
        [
            // THIS ENFORCES OneResult ON THE OP!
            // ConstantLike,
            Pure
            // OpAsmOpInterface
        ]> {
    let summary = "Defines a channel with one input and one output port";
    let description = [{
        The `channel` operation produces a typed channel (i.e., an edge in the
        dataflow graph) that links two operators (i.e., nodes) in the dataflow
        graph.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    // NOTE(feliix42): I want to _materialize_ the type attribute here -> derived attribute?
    //                 Alternatively have it as attribute and derive from custom assembly
    let arguments = (ins TypeAttr:$encapsulatedType);
    let results = (outs
        Dfg_InputType:$in_chan,
        Dfg_OutputType:$out_chan
    );

    let assemblyFormat = [{ attr-dict `:` $encapsulatedType `(` type($in_chan) `->` type($out_chan) `)` }];

    // TODO(feliix42): Ensure type matching on both results and the attribute
}

#endif
