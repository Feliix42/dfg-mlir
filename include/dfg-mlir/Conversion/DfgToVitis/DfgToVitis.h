/// Declaration of the Dfg to Vitis lowering pass that lowers dfg programs
/// to the Vitis HLS dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTDFGTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateDfgToVitisConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDfgToVitisPass();

namespace dfg {
void registerConvertToVitisPipelines();
void addConvertToVitisPasses(OpPassManager &pm);

void registerPrepareForVivadoPipelines();
void addPrepareForVivadoPasses(OpPassManager &pm);
} // namespace dfg

} // namespace mlir
