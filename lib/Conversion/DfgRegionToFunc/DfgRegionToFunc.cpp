/// Implementation of the DfgRegionToFunc pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgRegionToFunc/DfgRegionToFunc.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/IR/Diagnostics.h>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGREGIONTOFUNC
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

// ========================================================
// Lowerings
// ========================================================

namespace {
struct ConvertDfgRegionToFuncPass
        : public mlir::impl::ConvertDfgRegionToFuncBase<
              ConvertDfgRegionToFuncPass> {
    void runOnOperation() final;
};
} // namespace

struct RegionOpLowering : public OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    RegionOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        RegionOp op,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* context = rewriter.getContext();
        StringRef operatorName = adaptor.getSymName();
        FunctionType funcTy = adaptor.getFunctionType();

        std::vector<Type> new_inputs;
        for (auto ty : funcTy.getInputs()) {
            new_inputs.push_back(cast<OutputType>(ty).getElementType());
        }
        std::vector<Type> new_outputs;
        for (auto ty : funcTy.getResults()) {
            new_inputs.push_back(cast<InputType>(ty).getElementType());
        }

        FunctionType newFuncTy = FunctionType::get(context, new_inputs, new_outputs);

        auto genFuncOp =
            func::FuncOp::create(rewriter, loc, operatorName, newFuncTy);

        // - inline the region from `op` into the newly generated func op
        rewriter.inlineRegionBefore(
            adaptor.getBody(),
            genFuncOp.getBody(),
            genFuncOp.end());

        Block &entryBlock = genFuncOp.getBody().front();
        std::vector<Value> retVals;
        auto inputCount = new_inputs.size();
        while (entryBlock.getNumArguments() > inputCount) {
            // - remove the return values one by one, add a pull and append it to the return
            BlockArgument out = entryBlock.getArgument(inputCount);

            if (!out.hasOneUse())
                return emitError(out.getLoc(), "Expected output to have exactly one use");

            ConnectOutputOp o = cast<ConnectOutputOp>(*out.getUsers().begin());

            rewriter.setInsertionPoint(o);
            PullOp newPull = rewriter.replaceOpWithNewOp<PullOp>(o, o.getChannelPort());
            retVals.push_back(newPull.getResult());

            entryBlock.eraseArgument(inputCount);
        }

        // - insert the actual return
        rewriter.setInsertionPointToEnd(&entryBlock);
        func::ReturnOp::create(rewriter, op.getLoc(), retVals);

        // - replace the block argument types with the newly generated types of new_inputs and new_outputs
        for (auto arg: entryBlock.getArguments()) {
            if (!arg.hasOneUse())
                return emitError(arg.getLoc(), "Expected input to have exactly one use");

            ConnectInputOp i = cast<ConnectInputOp>(*arg.getUsers().begin());

            OutputType ty = cast<OutputType>(arg.getType());
            arg.setType(ty.getElementType());

            rewriter.setInsertionPoint(i);
            rewriter.replaceOpWithNewOp<PushOp>(i, arg, i.getChannelPort());
        }

        rewriter.eraseOp(op);

        return success();
    }
};


void mlir::populateDfgRegionToFuncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.region -> func.func
    patterns.add<RegionOpLowering>(typeConverter, patterns.getContext());

    // dfg.connect.input -> [ dfg.push ]
    // patterns.add<ConnectInputOpLowering>(typeConverter, patterns.getContext());

    // dfg.connect.output -> [ dfg.pull ]
    // patterns.add<ConnectOutputOpLowering>(typeConverter, patterns.getContext());
}

void ConvertDfgRegionToFuncPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgRegionToFuncConversionPatterns(converter, patterns);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        omp::OpenMPDialect,
        LLVM::LLVMDialect>();

    target.addLegalDialect<DfgDialect>();

    // It is intentional that `EmbedOp` is illegal from here on but no lowering
    // is supplied. It's expected that all embeddings have been removed by means of inlining.
    target.addIllegalOp<
        ProcessOp,
        LoopOp,
        OperatorOp,
        InstantiateOp,
        ConnectInputOp,
        ConnectOutputOp,
        RegionOp,
        EmbedOp>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgRegionToFuncPass()
{ return std::make_unique<ConvertDfgRegionToFuncPass>(); }
