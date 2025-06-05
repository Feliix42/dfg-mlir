/// Implements the emitHLS dialect base.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/emitHLS/IR/Dialect.h"

#include "dfg-mlir/Dialect/emitHLS/IR/Ops.h"

using namespace mlir;
using namespace mlir::emitHLS;

//===- Generated implementation -------------------------------------------===//

#include "dfg-mlir/Dialect/emitHLS/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// emitHLSDialect
//===----------------------------------------------------------------------===//

void emitHLSDialect::initialize()
{
    // registerAttributes();
    registerOps();
    registerTypes();
}
