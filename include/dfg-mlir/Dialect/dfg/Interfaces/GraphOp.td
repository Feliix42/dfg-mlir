//===- GraphOp.td - DFG graph interface ---------------*- tablegen -*-===//
//
// This is the definitions file for the DFG graph interface.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_GRAPHOPINTERFACE
#define DFG_GRAPHOPINTERFACE

include "mlir/IR/OpBase.td"

def Dfg_GraphOpInterface : OpInterface<"GraphOpInterface"> {
    let cppNamespace = "::mlir::dfg";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the name of the graph.}],
            /*retTy=*/"std::string",
            /*methodName=*/"getGraphName",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Checks if the graph is a sub-graph.}],
            /*retTy=*/"bool",
            /*methodName=*/"isSubGraph",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the nodes of a graph.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Operation*>",
            /*methodName=*/"getGraphNodes",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                // Return an empty block by default
                return ::llvm::SmallVector<::mlir::Operation*>();
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the subgraphs in a graph.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Operation*>",
            /*methodName=*/"getGraphSubGs",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                // Return an empty block by default
                return ::llvm::SmallVector<::mlir::Operation*>();
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the edges of a graph.}],
            /*retTy=*/"::llvm::SmallVector<::mlir::Operation*>",
            /*methodName=*/"getGraphEdges",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                // Return an empty block by default
                return ::llvm::SmallVector<::mlir::Operation*>();
            }]
        >
    ];
}

#endif // DFG_GRAPHOPINTERFACE
