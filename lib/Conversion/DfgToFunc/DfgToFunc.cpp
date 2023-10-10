/// Implementation of the DfgToFunc pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"

// TODO(feliix42): Switch to per-pass classes, look into upstream :-)
#include "../PassDetails.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::dfg;

// ========================================================
// Helper Functions
// ========================================================

/// Return a symbol reference to the requested function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertFunc(
    PatternRewriter &rewriter,
    ModuleOp module,
    std::string funcName,
    Type result,
    ValueRange arguments)
{
    auto* context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
        return SymbolRefAttr::get(context, funcName);

    // convert the OperandRange into a list of types
    auto argIterator = arguments.getTypes();
    std::vector<Type> tyVec(argIterator.begin(), argIterator.end());

    // Create a function declaration for the desired function
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        result,
        tyVec,
        /*isVarArg=*/true);

    // Insert the function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, llvmFnType);
    return SymbolRefAttr::get(context, funcName);
}

// ========================================================
// Lowerings
// ========================================================

namespace {
struct ConvertDfgToFuncPass
        : public ConvertDfgToFuncBase<ConvertDfgToFuncPass> {
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
        // NOTE(feliix42): Is this check even necessary?
        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                adaptor.getBody(),
                genFuncOp.getBody(),
                genFuncOp.end());

            // add return op at the bottom of the body
            rewriter.setInsertionPointToEnd(&genFuncOp.getBody().back());
            rewriter.create<func::ReturnOp>(
                genFuncOp.getBody().back().back().getLoc(),
                ValueRange());
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

struct PushOpLowering : public OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp op,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // create function name for PushOp in LLVM IR
        std::string funcName = "push_";
        llvm::raw_string_ostream funcNameStream(funcName);
        // 0 is the type of the item sent
        adaptor.getOperands()[0].getType().print(funcNameStream);

        // return value
        Type boolReturnVal = rewriter.getI1Type();

        // fetch or create the FlatSymbolRefAttr for the called function
        ModuleOp parentModule = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
            rewriter,
            parentModule,
            funcNameStream.str(),
            boolReturnVal,
            adaptor.getOperands());

        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            ArrayRef<Type>(boolReturnVal),
            pushFuncName,
            adaptor.getOperands());

        // TODO: Replace users of the results

        rewriter.eraseOp(op);

        return success();
    }
};

struct PullOpLowering : public OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    PullOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // create function name for PullOp in LLVM IR
        std::string funcName = "pull_";
        llvm::raw_string_ostream funcNameStream(funcName);
        op.getType().print(funcNameStream);

        // create the struct type that models the result
        Type boolReturnVal = rewriter.getI1Type();
        std::vector<Type> structTypes{op.getType(), boolReturnVal};
        Type returnedStruct = LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            structTypes);

        // fetch or create the FlatSymbolRefAttr for the called function
        ModuleOp parentModule = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr pullFuncName = getOrInsertFunc(
            rewriter,
            parentModule,
            funcNameStream.str(),
            returnedStruct,
            adaptor.getOperands());

        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            ArrayRef<Type>(returnedStruct),
            pullFuncName,
            adaptor.getChan());

        // TODO: Replace users of the results

        rewriter.eraseOp(op);

        return success();
    }
};

// struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
//     using OpConversionPattern<ChannelOp>::OpConversionPattern;

//     ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
//             : OpConversionPattern<ChannelOp>(typeConverter, context){};

//     LogicalResult matchAndRewrite(
//         ChannelOp op,
//         ChannelOpAdaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         // NOTE(feliix42): Conceptually, what I probably want to do, is:
//         // 1. create a new CallOp calling the channel creation function
//         // 2. insert the appropriate definition at the top
//         // 3. replace all channel inputs and outputs with the newly created
//         // result
//         // 4. delete the old op
//         // 5. insert the lowering down there
//         return success();
//     }
// };

void mlir::populateDfgToFuncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.operator -> func.func
    patterns.add<OperatorOpLowering>(typeConverter, patterns.getContext());

    // dfg.instantiate -> func.call
    patterns.add<InstantiateOpLowering>(typeConverter, patterns.getContext());

    // dfg.push -> llvm.call + logic
    patterns.add<PushOpLowering>(typeConverter, patterns.getContext());

    // dfg.pull -> llvm.call + logic
    patterns.add<PullOpLowering>(typeConverter, patterns.getContext());

    // dfg.channel -> !llvm.ptr
    // patterns.add<ChannelOp>(typeConverter, patterns.getContext());
}

void ConvertDfgToFuncPass::runOnOperation()
{
    TypeConverter converter;

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
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        LLVM::LLVMDialect>();

    target.addLegalDialect<DfgDialect>();
    target.addIllegalOp<OperatorOp, PushOp, PullOp>();
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
// - [ ] single out OperatorOpLowering and InstantiateOpLowering
// - [ ] expand OperatorOpLowering to include cf logic already
// - [ ] modify the pull/push lowerings
// - [ ] rewrite ChannelOp
// - [ ] rewrite the LoopOp
// - [ ] check the type rewriter thingy