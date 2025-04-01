/// Implementation of ScfToVitis pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/ScfToVitis/ScfToVitis.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTSCFTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::vitis;

namespace {
struct ConvertForOp : OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;

    ConvertForOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<scf::ForOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        scf::ForOp op,
        scf::ForOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto lb = op.getLowerBound();
        auto ub = op.getUpperBound();
        auto step = op.getStep();

        Attribute lbVal, ubVal, stepVal;
        if (auto defOp = dyn_cast<vitis::VariableOp>(lb.getDefiningOp()))
            lbVal = defOp.getInitAttr();
        if (auto defOp = dyn_cast<vitis::VariableOp>(ub.getDefiningOp()))
            ubVal = defOp.getInitAttr();
        if (auto defOp = dyn_cast<vitis::VariableOp>(step.getDefiningOp()))
            stepVal = defOp.getInitAttr();

        if (op.getResults().size() != 0)
            return rewriter.notifyMatchFailure(
                loc,
                "No support for scf.for op with results");

        auto vitisFor = rewriter.create<ForOp>(loc, lbVal, ubVal, stepVal);
        rewriter.setInsertionPointToEnd(&vitisFor.getBody().front());
        IRMapping mapper;
        mapper.map(op.getInductionVar(), vitisFor.getInductionVar());

        // Copy scf.for contents into new vitis.for
        for (auto &opi : op.getRegion().getOps()) {
            if (isa<scf::YieldOp>(opi)) continue;
            auto clonedOp = rewriter.clone(opi, mapper);
            if (auto nestedFor = dyn_cast<scf::ForOp>(clonedOp)) {
                scf::ForOpAdaptor nestedAdaptor(nestedFor);
                if (failed(matchAndRewrite(nestedFor, nestedAdaptor, rewriter)))
                    return rewriter.notifyMatchFailure(
                        nestedFor.getLoc(),
                        "Failed to process nested for.");
            }
        }

        rewriter.replaceOp(op, vitisFor);
        return success();
    }
};
} // namespace

void mlir::populateScfToVitisConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertForOp>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertScfToVitisPass
        : public impl::ConvertScfToVitisBase<ConvertScfToVitisPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertScfToVitisPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateScfToVitisConversionPatterns(converter, patterns);

    target.addLegalDialect<vitis::VitisDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertScfToVitisPass()
{
    return std::make_unique<ConvertScfToVitisPass>();
}
