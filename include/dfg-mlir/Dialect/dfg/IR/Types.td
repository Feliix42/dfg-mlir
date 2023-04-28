//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_TYPES
#define DFG_TYPES

include "dfg-mlir/Dialect/dfg/IR/Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Dfg_InputType : Dfg_Type<"Input", []> {
    let mnemonic = "input";
    let summary = "Sending channel end type";
    let description = [{
        The `input` type represents the sending end of a dataflow edge in a dataflow graph.

        TODO(feliix42): Expand documentation (Syntax, Examples, details, ...)
    }];

    // FIXME(feliix42): Why does `AnyType` not work??
    let parameters = (ins "Type":$elementType);

    let assemblyFormat = "`<` $elementType `>`";

    // TODO(feliix42): Optionally add a verifier
    // let genVerifyDecl = 1;
}

def Dfg_OutputType : Dfg_Type<"Output", []> {
    let mnemonic = "output";
    let summary = "Receiving channel end type";
    let description = [{
        the `output` type represents the receiving end of a dataflow edge in a dataflow graph.

        TODO(feliix42): Expand documentation (Syntax, Examples, details, ...)
    }];

    let parameters = (ins "Type":$elementType);

    let assemblyFormat = "`<` $elementType `>`";

    // TODO(feliix42): Optionally add a verifier
    // let genVerifyDecl = 1;
}

#endif
