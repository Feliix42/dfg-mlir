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

#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "vitis-types"

using namespace mlir;
using namespace mlir::vitis;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dfg-mlir/Dialect/vitis/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

ArrayType ArrayType::cloneWith(
    std::optional<ArrayRef<int64_t>> shape,
    Type elementType) const
{
    return ArrayType::get(shape.value_or(getShape()), elementType);
}

Type ArrayType::parse(AsmParser &parser)
{

    SmallVector<int64_t> shape;
    Type elementType;
    if (parser.parseLess()
        || parser.parseDimensionList(shape, /*allowDynamic=*/false)
        || parser.parseType(elementType) || parser.parseGreater())
        return Type();

    return ArrayType::get(shape, elementType);
}

void ArrayType::print(AsmPrinter &p) const
{
    p << "<";
    p.printDimensionList(getShape());
    p << "x";
    p << getElementType();
    p << ">";
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

LogicalResult StreamType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type stream_type)
{
    if (!llvm::isa<IntegerType, FloatType>(stream_type))
        return emitError() << "Only scalar type is supported in stream";
    return success();
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult PointerType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type pointer_type)
{
    if (!llvm::isa<IntegerType, FloatType>(pointer_type))
        return emitError() << "Only scalar type is supported in stream";
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
