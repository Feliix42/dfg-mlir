//===- NodeOp.td - DFG node interface ---------------*- tablegen -*-===//
//
// This is the definitions file for the DFG node interface.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_NODEOPINTERFACE
#define DFG_NODEOPINTERFACE

include "mlir/IR/OpBase.td"

def Dfg_NodeOpInterface : OpInterface<"NodeOpInterface"> {
    let cppNamespace = "::mlir::dfg";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the name of this node.}],
            /*retTy=*/"std::string",
            /*methodName=*/"getNodeName",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the number of input ports.}],
            /*retTy=*/"unsigned",
            /*methodName=*/"getNumInputPorts",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the input port with the given index.}],
            /*retTy=*/"::mlir::Value",
            /*methodName=*/"getInputPort",
            /*args=*/(ins "unsigned":$index),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the range of all input port types.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Type>",
            /*methodName=*/"getInputPortTypes",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the range of all input ports.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
            /*methodName=*/"getInputPorts",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the number of output ports.}],
            /*retTy=*/"unsigned",
            /*methodName=*/"getNumOutputPorts",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the output port with the given index.}],
            /*retTy=*/"::mlir::Value",
            /*methodName=*/"getOutputPort",
            /*args=*/(ins "unsigned":$index),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the range of all output port types.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Type>",
            /*methodName=*/"getOutputPortTypes",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the range of all output ports.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
            /*methodName=*/"getOutputPorts",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >
    ];
}

#endif // DFG_NODEOPINTERFACE
