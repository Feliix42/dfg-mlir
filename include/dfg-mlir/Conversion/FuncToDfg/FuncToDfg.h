/// Declaration of the Func to Dfg conversion pass that lowers functions
/// to the dfg dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTFUNCTODFG
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateFuncToDfgConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertFuncToDfgPass();

} // namespace mlir
