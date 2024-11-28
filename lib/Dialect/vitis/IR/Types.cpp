/// Implements the vitis dialect types.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/vitis/IR/Types.h"

#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFixedPoint.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "vitis-types"

using namespace mlir;
using namespace mlir::vitis;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dfg-mlir/Dialect/vitis/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

LogicalResult StreamType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type stream_type)
{
    if (!llvm::isa<APAxiUType>(stream_type)
        && !llvm::isa<APAxiSType>(stream_type))
        return emitError()
               << "Only ap_axi type or alias is supported in stream";
    return success();
}

//===----------------------------------------------------------------------===//
// VitisDialect
//===----------------------------------------------------------------------===//

void VitisDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "dfg-mlir/Dialect/vitis/IR/Types.cpp.inc"
        >();
}
