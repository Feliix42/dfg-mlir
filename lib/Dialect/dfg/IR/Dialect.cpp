/// Implements the dfg dialect base.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"

#include "dfg-mlir/Dialect/dfg/IR/Ops.h"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "dfg-mlir/Dialect/dfg/IR/Attributes.cpp.inc"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DfgDialect
//===----------------------------------------------------------------------===//

void DfgDialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();
}

void DfgDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "dfg-mlir/Dialect/dfg/IR/Attributes.cpp.inc"
        >();
}
