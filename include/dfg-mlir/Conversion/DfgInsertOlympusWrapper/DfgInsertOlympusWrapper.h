/// Declaration of the Dfg to Olympus lowering pass that lowers offloaded nodes
/// to individual wrappers for Olympus
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTDFGINSERTOLYMPUSWRAPPER
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateDfgInsertOlympusWrapperConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDfgInsertOlympusWrapperPass();

} // namespace mlir
