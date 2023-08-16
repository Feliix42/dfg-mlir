/// Declaration of the standard to Circt lowering pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTSTDTOCIRCT
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateStdToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertStdToCirctPass();

} // namespace mlir
