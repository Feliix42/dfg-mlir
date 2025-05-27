/// Implementation of LinalgToDfg pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/LinalgToDfg/LinalgToDfg.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTODFG
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {} // namespace

void mlir::populateLinalgToDfgConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{}

// TODO: To systolic array
namespace {
struct ConvertLinalgToDfgPass
        : public impl::ConvertLinalgToDfgBase<ConvertLinalgToDfgPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertLinalgToDfgPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateLinalgToDfgConversionPatterns(converter, patterns);

    // target.addLegalOp<func::FuncOp>();
    target.addLegalDialect<DfgDialect, arith::ArithDialect>();
    target.addIllegalDialect<linalg::LinalgDialect, func::FuncDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertLinalgToDfgPass()
{
    return std::make_unique<ConvertLinalgToDfgPass>();
}
