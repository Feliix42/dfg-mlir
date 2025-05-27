//===- EdgeOp.td - DFG edge interface ---------------*- tablegen -*-===//
//
// This is the definitions file for the DFG edge interface.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_EDGEOPINTERFACE
#define DFG_EDGEOPINTERFACE

include "mlir/IR/OpBase.td"

def Dfg_EdgeOpInterface : OpInterface<"EdgeOpInterface"> {
    let cppNamespace = "::mlir::dfg";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the node the input port connects to.}],
            /*retTy=*/"::mlir::Operation*",
            /*methodName=*/"getInputConnection",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the node the output port connects to.}],
            /*retTy=*/"::mlir::Operation*",
            /*methodName=*/"getOutputConnection",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >
    ];
}

#endif // DFG_EDGEOPINTERFACE
