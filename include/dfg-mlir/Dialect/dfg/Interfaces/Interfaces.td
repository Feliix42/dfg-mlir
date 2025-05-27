//===- Interfaces.td ---------------------------------------*- tablegen -*-===//
//
// This is the convenience include for all dfg dialect interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_INTERFACES
#define DFG_INTERFACES

include "dfg-mlir/Dialect/dfg/Interfaces/EdgeOp.td"
include "dfg-mlir/Dialect/dfg/Interfaces/GraphOp.td"
include "dfg-mlir/Dialect/dfg/Interfaces/NodeOp.td"

#endif // DFG_INTERFACES