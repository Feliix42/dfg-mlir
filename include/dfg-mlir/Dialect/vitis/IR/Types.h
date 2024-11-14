/// Declaration of the vitis dialect types.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Types.h"

//===- Generated Includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dfg-mlir/Dialect/vitis/IR/Types.h.inc"

//===----------------------------------------------------------------------===//
namespace mlir {
namespace vitis {

template<typename... Types>
class TypeVariant
        : public ::mlir::Type::
              TypeBase<TypeVariant<Types...>, mlir::Type, mlir::TypeStorage> {
    using mlir::Type::TypeBase<
        TypeVariant<Types...>,
        mlir::Type,
        mlir::TypeStorage>::Base::Base;

public:
    // Support LLVM isa/cast/dyn_cast to one of the possible types.
    static bool classof(Type other) { return type_isa<Types...>(other); }
};

} // namespace vitis
} // namespace mlir