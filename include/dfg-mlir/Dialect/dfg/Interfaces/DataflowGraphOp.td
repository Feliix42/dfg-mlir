//===- DataflowGraphOp.td -----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg DataflowGraphOp interface.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_INTERFACES_DATAFLOWGRAPHOP
#define DFG_INTERFACES_DATAFLOWGRAPHOP

include "mlir/IR/OpBase.td"

def Dfg_DataflowGraphOp : OpInterface<"DataflowGraphOp">
{
    let cppNamespace = "::mlir::dfg";
    let description = [{
        ``` TODO
        ```
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                dummy foo
            }],
            /*retTy=*/"void",
            /*methodName=*/"foo",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">
        // InterfaceMethod<
        //     /*desc=*/[{}],
        //     /*retTy=*/"",
        //     /*methodName=*/"",
        //     /*args=*/(ins),
        //     /*methodBody=*/"",
        //     /*defaultImplementation=*/"">
    ];
}

#endif // DFG_INTERFACES_DATAFLOWGRAPHOP
