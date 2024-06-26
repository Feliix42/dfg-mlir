/// Declaration of dfg process to yaml transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGPrintOperatorToYaml
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populatePrintOperatorToYamlConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgPrintOperatorToYamlPass();

} // namespace dfg
} // namespace mlir
