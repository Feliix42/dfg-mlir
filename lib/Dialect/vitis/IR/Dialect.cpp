/// Implements the vitis dialect base.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"

#include "dfg-mlir/Dialect/vitis/IR/Ops.h"

using namespace mlir;
using namespace mlir::vitis;

//===- Generated implementation -------------------------------------------===//

#include "dfg-mlir/Dialect/vitis/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VitisDialect
//===----------------------------------------------------------------------===//

void VitisDialect::initialize()
{
    // registerAttributes();
    registerOps();
    registerTypes();
}
