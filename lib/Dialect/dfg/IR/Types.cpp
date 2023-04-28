/// Implements the dfg dialect types.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFixedPoint.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "dfg-types"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dfg-mlir/Dialect/dfg/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Base2Dialect
//===----------------------------------------------------------------------===//

void DfgDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "dfg-mlir/Dialect/dfg/IR/Types.cpp.inc"
        >();
}