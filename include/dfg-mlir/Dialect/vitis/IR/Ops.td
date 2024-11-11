//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_OPS
#define VITIS_OPS

include "dfg-mlir/Dialect/vitis/IR/Types.td"
include "dfg-mlir/Dialect/vitis/IR/Dialect.td"
include "dfg-mlir/Dialect/vitis/IR/PragmaOps.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

#endif // VITIS_OPS
