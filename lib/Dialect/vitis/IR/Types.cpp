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
// APUIntType
//===----------------------------------------------------------------------===//

Type APUIntType::get(mlir::TypedAttr datawidth)
{
    auto widthWidth = llvm::dyn_cast<IntegerType>(datawidth.getType());
    assert(
        widthWidth && widthWidth.getWidth() == 32
        && "!hw.int width must be 32-bits");
    (void)widthWidth;

    if (auto cstWidth = llvm::dyn_cast<IntegerAttr>(datawidth))
        return IntegerType::get(
            datawidth.getContext(),
            cstWidth.getValue().getZExtValue());

    return Base::get(datawidth.getContext(), datawidth);
}

Type APUIntType::parse(AsmParser &p)
{
    auto i32Ty = p.getBuilder().getIntegerType(32);
    mlir::TypedAttr datawidth;
    if (p.parseLess() || p.parseAttribute(datawidth, i32Ty) || p.parseGreater())
        return Type();
    return get(datawidth);
}

void APUIntType::print(AsmPrinter &p) const
{
    p << "<";
    p.printAttributeWithoutType(getDatawidth());
    p << '>';
}

//===----------------------------------------------------------------------===//
// APSIntType
//===----------------------------------------------------------------------===//

Type APSIntType::get(mlir::TypedAttr datawidth)
{
    auto widthWidth = llvm::dyn_cast<IntegerType>(datawidth.getType());
    assert(
        widthWidth && widthWidth.getWidth() == 32
        && "!hw.int width must be 32-bits");
    (void)widthWidth;

    if (auto cstWidth = llvm::dyn_cast<IntegerAttr>(datawidth))
        return IntegerType::get(
            datawidth.getContext(),
            cstWidth.getValue().getZExtValue());

    return Base::get(datawidth.getContext(), datawidth);
}

Type APSIntType::parse(AsmParser &p)
{
    auto i32Ty = p.getBuilder().getIntegerType(32);
    mlir::TypedAttr datawidth;
    if (p.parseLess() || p.parseAttribute(datawidth, i32Ty) || p.parseGreater())
        return Type();
    return get(datawidth);
}

void APSIntType::print(AsmPrinter &p) const
{
    p << "<";
    p.printAttributeWithoutType(getDatawidth());
    p << '>';
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

LogicalResult StreamType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type stream_type)
{
    if (!llvm::isa<APAxiUType>(stream_type)
        && !llvm::isa<APAxiSType>(stream_type)
        && !llvm::isa<AliasType>(stream_type))
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
