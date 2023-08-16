/// Declaration of the Dfg to Async lowering pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

void populateDfgToAsyncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDfgToAsyncPass();

} // namespace mlir
