/// Implementation of the DfgToFunc pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOFUNC
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

// ========================================================
// Lowerings
// ========================================================

namespace {
struct ConvertDfgToFuncPass
        : public mlir::impl::ConvertDfgToFuncBase<ConvertDfgToFuncPass> {
    void runOnOperation() final;
};
} // namespace

struct OperatorOpLowering : public OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    OperatorOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp op,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* context = rewriter.getContext();
        StringRef operatorName = adaptor.getSymName();
        FunctionType funcTy = adaptor.getFunctionType();
        auto old_inputs = funcTy.getInputs();
        auto old_outputs = funcTy.getResults();

        std::vector<Type> inputs = old_inputs.vec();
        inputs.insert(inputs.end(), old_outputs.begin(), old_outputs.end());
        FunctionType newFuncTy = FunctionType::get(context, inputs, {});

        auto genFuncOp =
            rewriter.create<func::FuncOp>(loc, operatorName, newFuncTy);

        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                adaptor.getBody(),
                genFuncOp.getBody(),
                genFuncOp.end());

            // add a terminator block to the function
            Block* terminatorBlock = rewriter.createBlock(
                &genFuncOp.getBody(),
                genFuncOp.getBody().end());

            // insert the cf.br operation at the end of the previous block
            rewriter.setInsertionPointToEnd(&genFuncOp.getBody().front());
            auto branchOp = rewriter.create<cf::BranchOp>(
                genFuncOp.getBody().front().back().getLoc(),
                terminatorBlock);

            // populate the Terminator Block (for now) with a return only
            rewriter.setInsertionPointToEnd(terminatorBlock);
            rewriter.create<func::ReturnOp>(branchOp.getLoc());
        }

        rewriter.eraseOp(op);

        return success();
    }
};

struct InstantiateOpLowering : public OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // don't lower offloaded functions
        if (adaptor.getOffloaded()) return failure();

        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            adaptor.getCallee(),
            ArrayRef<Type>(),
            adaptor.getOperands());

        return success();
    }
};

void mlir::populateDfgToFuncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.operator -> func.func
    patterns.add<OperatorOpLowering>(typeConverter, patterns.getContext());

    // dfg.instantiate -> func.call
    patterns.add<InstantiateOpLowering>(typeConverter, patterns.getContext());
}

void ConvertDfgToFuncPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([](Type t) { return t; });

    // TODO(feliix42): add type conversion here
    // converter.addConversion([&](Type type) {
    //     if (isa<IntegerType>(type)) return type;
    //     return Type();
    // });

    // TODO:
    // look at the places where the populate Functions for builtin ops are
    // defined to copy the dynamic legality constraints and type rewriter
    // patterns for these ops

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToFuncConversionPatterns(converter, patterns);
    // TODO: remove below??
    // populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    // populateReturnOpTypeConversionPattern(patterns, converter);
    // populateCallOpTypeConversionPattern(patterns, converter);
    // populateBranchOpInterfaceTypeConversionPattern(patterns, converter);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        LLVM::LLVMDialect>();

    target.addLegalDialect<DfgDialect>();
    target.addIllegalOp<OperatorOp>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return op.getOffloaded(); });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToFuncPass()
{
    return std::make_unique<ConvertDfgToFuncPass>();
}

// STEPS
// - [x] rewrite to use the populate function
// - [x] use adaptor where possible
// - [x] single out OperatorOpLowering and InstantiateOpLowering
// - [x] expand OperatorOpLowering to include cf logic already
// - [ ] modify the pull/push lowerings to include the necessary logic for
// breaking
// - [ ] insert the logic for closing channels in the break block
//       - Make all FuncOps with channels in the signature illegal -> alter the
//       type & insert
// - [ ] rewrite ChannelOp
// - [ ] rewrite the LoopOp
// - [ ] check the type rewriter thingy