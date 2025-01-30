/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Bufferize/Bufferize.h"
#include "dfg-mlir/Dialect/dfg/Transforms/FlattenMemref/FlattenMemref.h"
#include "dfg-mlir/Dialect/dfg/Transforms/InlineRegion/InlineRegion.h"
#include "dfg-mlir/Dialect/dfg/Transforms/OpereatorToProcess/OperatorToProcess.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <mlir/Conversion/TosaToLinalg/TosaToLinalg.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGLOWERINSIDETOLINALG
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

namespace {
struct DfgLowerInsideToLinalgPass
        : public dfg::impl::DfgLowerInsideToLinalgBase<
              DfgLowerInsideToLinalgPass> {
    void runOnOperation() override
    {

        TypeConverter converter;
        tosa::populateTosaTypeConversion(converter);

        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        // From tosa to linalg.
        tosa::populateTosaDecomposeConv2D(&getContext(), patterns);
        tosa::populateTosaDecomposeTransposeConv(&getContext(), patterns);
        tosa::populateTosaDecomposeDepthwise(&getContext(), patterns);
        tosa::populateTosaToLinalgConversionPatterns(converter, &patterns);

        target.addIllegalDialect<tosa::TosaDialect>();
        target.addLegalDialect<DfgDialect>();
        target.addIllegalOp<OperatorOp>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        if (failed(applyPartialConversion(
                getOperation(),
                target,
                std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createDfgLowerInsideToLinalgPass()
{
    return std::make_unique<DfgLowerInsideToLinalgPass>();
}

void mlir::dfg::addDfgLowerInsideToScfPasses(OpPassManager &pm)
{
    pm.addPass(dfg::createDfgOperatorToProcessPass());
    pm.addPass(dfg::createDfgInlineRegionPass());
    pm.addPass(dfg::createDfgBufferizePass());
    pm.addPass(dfg::createDfgLowerInsideToLinalgPass());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(dfg::createDfgFlattenMemrefPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
}

void mlir::dfg::registerDfgLowerInsideToScfPipelines()
{
    PassPipelineRegistration<>(
        "dfg-lower-inside-to-scf",
        "Lower inside to scf level",
        [](OpPassManager &pm) { addDfgLowerInsideToScfPasses(pm); });
}
