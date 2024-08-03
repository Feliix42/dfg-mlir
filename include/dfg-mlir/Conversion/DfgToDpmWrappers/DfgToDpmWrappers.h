/// @file
/// @author     Fabius Mayer-Uhma (fabius.mayer-uhma@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTDFGTODPMWRAPPERS
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateDfgToDpmWrappersConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDfgToDpmWrappersPass();

} // namespace mlir
