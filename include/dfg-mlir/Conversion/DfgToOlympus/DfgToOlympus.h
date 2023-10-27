/// Declaration of the Dfg to Olympus lowering pass that lowers offloaded nodes
/// to the olympus dialect.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTDFGTOOLYMPUS
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateDfgToOlympusConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDfgToOlympusPass();

} // namespace mlir
