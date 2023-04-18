/// Implements the dfg dialect base.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Base.h"

// #include "dfg-mlir/Dialect/dfg/IR/Ops.h"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#include "dfg-mlir/Dialect/dfg/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

void Base2Dialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();
}
