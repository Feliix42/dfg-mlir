/// Implementation of the DfgToLLVM pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToLLVM/DfgToLLVM.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOLLVM
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

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
struct ConvertDfgToLLVMPass
        : public mlir::impl::ConvertDfgToLLVMBase<ConvertDfgToLLVMPass> {
    void runOnOperation() final;
};
} // namespace

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

void mlir::populateDfgToLLVMConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.push -> llvm.call + logic
    patterns.add<PushOpLowering>(typeConverter, patterns.getContext());

    // dfg.pull -> llvm.call + logic
    patterns.add<PullOpLowering>(typeConverter, patterns.getContext());

    // dfg.channel -> !llvm.ptr
    // patterns.add<ChannelOp>(typeConverter, patterns.getContext());
}

void ConvertDfgToLLVMPass::runOnOperation()
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

    populateDfgToLLVMConversionPatterns(converter, patterns);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        LLVM::LLVMDialect>();

    target.addLegalDialect<DfgDialect>();
    // NOTE(feliix42): Keep InstantiateOp and OperatorOp illegal here as they
    // should've been removed in a previous pass.
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

std::unique_ptr<Pass> mlir::createConvertDfgToLLVMPass()
{
    return std::make_unique<ConvertDfgToLLVMPass>();
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