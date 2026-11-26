/// Declaration of the dfg dialect attributes.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

//===----------------------------------------------------------------------===//
// Generated includes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "dfg-mlir/Dialect/dfg/IR/Attributes.h.inc"

namespace mlir {
namespace dfg {

/// Returns the fusion group name carried by the given op (via the
/// discardable `fusion` attribute), or an empty StringRef if the op is not
/// annotated.
inline StringRef getFusionGroup(Operation *op) {
    if (auto attr = op->getAttrOfType<FusionAttr>("fusion"))
        return attr.getGroup().getValue();
    return {};
}

/// Attaches a fusion group annotation to the given op.
inline void setFusionGroup(Operation *op, StringRef group) {
    auto *ctx = op->getContext();
    op->setAttr("fusion", FusionAttr::get(ctx, StringAttr::get(ctx, group)));
}

/// Copies the fusion annotation from `src` to `dst`, if present. Used by
/// conversions that replace an annotated op to propagate the annotation onto
/// the replacement op.
inline void copyFusionAttr(Operation *src, Operation *dst) {
    if (Attribute attr = src->getAttr("fusion"))
        dst->setAttr("fusion", attr);
}

} // namespace dfg
} // namespace mlir
