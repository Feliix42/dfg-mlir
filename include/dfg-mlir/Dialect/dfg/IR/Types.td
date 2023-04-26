//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_TYPES
#define DFG_TYPES

include "dfg-mlir/Dialect/dfg/IR/Base.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Dfg_ChannelType : Dfg_Type<"Channel", []> {
    let mnemonic = "channel";
    let summary = "Channel type";
    let description = [{
        The `channel` type represents the link between two nodes in a dataflow graph.

        TODO(feliix42): Expand documentation (Syntax, Examples, details, ...)
    }];

    let parameters = (ins 
        "Type":$elementType);

    let assemblyFormat = "`<` $elementType `>`";

    // TODO(feliix42): Optionally add a verifier
    // let genVerifyDecl = 1;
}

#endif
