/// Implementation of the DfgToLLVM pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToLLVM/DfgToLLVM.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

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
    if (module.lookupSymbol<func::FuncOp>(funcName))
        return SymbolRefAttr::get(context, funcName);

    // convert the OperandRange into a list of types
    auto argIterator = arguments.getTypes();
    std::vector<Type> tyVec(argIterator.begin(), argIterator.end());

    // Create a function declaration for the desired function
    auto fnType = rewriter.getFunctionType(tyVec, result);

    // Insert the function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<func::FuncOp>(module.getLoc(), funcName, fnType)
        .setPrivate();
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

LogicalResult insertTeardownFunctionFromPush(
    ModuleOp parentModule,
    PushOp pushOp,
    ConversionPatternRewriter &rewriter)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "close_";
    llvm::raw_string_ostream funcNameStream(funcName);
    // 0 is the type of the item sent
    // TODO(feliix42): Add name mangling here
    funcNameStream << pushOp.getOperands()[0].getType();

    auto llvmVoid = LLVM::LLVMVoidType::get(pushOp.getContext());

    // fetch or create the FlatSymbolRefAttr for the called function
    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        llvmVoid,
        pushOp.getChan());

    rewriter.create<func::CallOp>(
        pushOp.getLoc(),
        pushFuncName,
        ArrayRef<Type>(llvmVoid),
        pushOp.getChan());

    return success();
}

LogicalResult insertTeardownFunctionFromPull(
    ModuleOp parentModule,
    PullOp pullOp,
    ConversionPatternRewriter &rewriter)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "close_";
    llvm::raw_string_ostream funcNameStream(funcName);
    // 0 is the type of the item sent
    // TODO(feliix42): Add name mangling here
    funcNameStream << pullOp.getType();

    auto llvmVoid = LLVM::LLVMVoidType::get(pullOp.getContext());

    // Translate to InputType, insert UnrealizedConversionCastOp
    UnrealizedConversionCastOp inputType =
        rewriter.create<UnrealizedConversionCastOp>(
            pullOp.getLoc(),
            rewriter.getType<InputType>(pullOp.getType()),
            pullOp.getChan());

    // fetch or create the FlatSymbolRefAttr for the called function
    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        llvmVoid,
        inputType.getResult(0));

    rewriter.create<func::CallOp>(
        pullOp.getLoc(),
        pushFuncName,
        ArrayRef<Type>(llvmVoid),
        inputType.getResult(0));

    return success();
}

LogicalResult rewritePushOp(
    PushOp op,
    Block* terminatorBlock,
    ConversionPatternRewriter &rewriter)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "push_";
    llvm::raw_string_ostream funcNameStream(funcName);
    // 0 is the type of the item sent
    // TODO(feliix42): Add name mangling here
    funcNameStream << op->getOperands()[0].getType();

    // return value
    Type boolReturnVal = rewriter.getI1Type();

    // fetch or create the FlatSymbolRefAttr for the called function
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        boolReturnVal,
        op->getOperands());

    func::CallOp pushOperation = rewriter.create<func::CallOp>(
        op->getLoc(),
        pushFuncName,
        ArrayRef<Type>(boolReturnVal),
        op->getOperands());

    // to handle the result of the push, we must do the following:
    // - split the current block after this instruction
    // - conditionally either continue processing the body or break the
    // computation

    Block* currentBlock = pushOperation->getBlock();

    // create the new block with the same argument list
    Block* newBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // insert conditional jump to the break block
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::CondBranchOp>(
        pushOperation.getLoc(),
        pushOperation.getResult(0),
        newBlock,
        /* trueArgs = */ ArrayRef<Value>(),
        terminatorBlock,
        /* falseArgs = */ ArrayRef<Value>());

    op->erase();

    return success();
}

LogicalResult rewritePullOp(
    PullOp op,
    Block* terminatorBlock,
    ConversionPatternRewriter &rewriter)
{
    // create function name for PullOp in LLVM IR
    std::string funcName = "pull_";
    llvm::raw_string_ostream funcNameStream(funcName);
    op.getType().print(funcNameStream);

    // create the struct type that models the result
    Type boolReturnVal = rewriter.getI1Type();
    std::vector<Type> structTypes{op.getType(), boolReturnVal};
    Type returnedStruct =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), structTypes);

    // fetch or create the FlatSymbolRefAttr for the called function
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr pullFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcNameStream.str(),
        returnedStruct,
        op.getOperand());

    func::CallOp pullOperation = rewriter.create<func::CallOp>(
        op.getLoc(),
        ArrayRef<Type>(returnedStruct),
        pullFuncName,
        op.getChan());

    LLVM::ExtractValueOp value = rewriter.create<LLVM::ExtractValueOp>(
        pullOperation.getLoc(),
        pullOperation.getResult(0),
        0);
    LLVM::ExtractValueOp valid = rewriter.create<LLVM::ExtractValueOp>(
        value.getLoc(),
        pullOperation.getResult(0),
        1);

    op.getResult().replaceAllUsesWith(value.getResult());

    Block* currentBlock = pullOperation->getBlock();

    // create the new block with everything following this pull op
    Block* newBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    // Originally, I added the Block arguments of the current block to the next
    // block. But because of the nature of SSA value scoping, that's not
    // necessary. All values of a block are visible in the successor blocks.

    // insert conditional jump to the break block
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::CondBranchOp>(
        valid.getLoc(),
        valid.getResult(),
        newBlock,
        ArrayRef<Value>(),
        // blockArgs,
        terminatorBlock,
        ArrayRef<Value>());

    op->erase();

    return success();
}

struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context){};
    // struct ChannelOpLowering : public ConvertOpToLLVMPattern<ChannelOp> {
    // using ConvertOpToLLVMPattern<ChannelOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // NOTE(feliix42): Conceptually, what I probably want to do, is:
        // 1. create a new CallOp calling the channel creation function
        // 2. insert the appropriate definition at the top
        // 3. replace all channel inputs and outputs with the newly created
        // result
        // 4. delete the old op
        // 5. insert the lowering down there
        std::string funcName = "channel_";
        llvm::raw_string_ostream funcNameStream(funcName);
        funcNameStream << op.getEncapsulatedType();

        // Do I need an unrealized cast?

        return success();
    }
};

void mlir::populateDfgToLLVMConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.channel -> !llvm.ptr
    patterns.add<ChannelOpLowering>(typeConverter, patterns.getContext());
}

Type getLLVMStructfromChannelType(const Type &elementType, MLIRContext* ctx)
{
    Type elementPtr = LLVM::LLVMPointerType::get(elementType);
    Type i64Type = IntegerType::get(ctx, 64);
    std::vector<Type> structTypes{
        elementPtr,
        i64Type,
        i64Type,
        i64Type,
        i64Type,
        IntegerType::get(ctx, 8)};
    return LLVM::LLVMStructType::getLiteral(ctx, structTypes);
}

void ConvertDfgToLLVMPass::runOnOperation()
{
    Operation* op = getOperation();

    TypeConverter converter;

    // this is a stack!! Hence, the most generic conversion lives at the bottom
    converter.addConversion([](Type t) { return t; });
    converter.addConversion([](OutputType t) {
        return getLLVMStructfromChannelType(t.getElementType(), t.getContext());
    });
    converter.addConversion([](InputType t) {
        return getLLVMStructfromChannelType(t.getElementType(), t.getContext());
    });
    converter.addSourceMaterialization(
        [&](OpBuilder &builder,
            Type resultType,
            ValueRange inputs,
            Location loc) -> std::optional<Value> {
            if (inputs.size() != 1) return std::nullopt;

            return builder
                .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                .getResult(0);
        });
    converter.addTargetMaterialization(
        [&](OpBuilder &builder,
            Type resultType,
            ValueRange inputs,
            Location loc) -> std::optional<Value> {
            if (inputs.size() != 1) return std::nullopt;

            return builder
                .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                .getResult(0);
        });

    // =========================================================================================
    // In the following block, we do multiple things in one go:
    // 0. create an empty terminator block for any functions with dfg.pull/push
    // ops.
    // 1. lower `PushOp` and `PullOp` into their respective `LLVM::CallOp`s
    // 2. insert the necessary teardown logic for channels
    //
    // The reasoning behind doing this with a walk pattern instead of a "normal"
    // ConversionRewritePattern is the irreducible entangledness between the
    // three aforementioned tasks. We cannot "just" create an empty termination
    // block in one pass and find it again later. Canonicalization passes will
    // immediately remove it. Similarly, it's hard to find a non-empty block
    // again later without retaining a reference all the time. In the end, it
    // was easier to solve this problem by doing all three things in one go.

    // per func, collect all push/pull ops
    WalkResult res = op->walk([&](func::FuncOp funcOp) -> WalkResult {
        SmallVector<PushOp> pushOps;
        SmallVector<PullOp> pullOps;
        funcOp->walk([&pushOps](PushOp push) {
            pushOps.push_back(push);
            return WalkResult::advance();
        });
        funcOp->walk([&pullOps](PullOp pull) {
            pullOps.push_back(pull);
            return WalkResult::advance();
        });

        if (pullOps.empty() && pushOps.empty()) return WalkResult::advance();
        // if (pushOps.empty()) return WalkResult::advance();

        Block* terminatorBlock = funcOp.addBlock();

        // search all returns, forward to TerminatorBlock
        funcOp->walk([&](func::ReturnOp ret) {
            ConversionPatternRewriter localRewriter(ret->getContext());
            localRewriter.setInsertionPoint(ret);
            // NOTE(feliix42): replaceWithNewOp does *not* work here!
            localRewriter.create<cf::BranchOp>(ret->getLoc(), terminatorBlock);
            ret->erase();
        });

        // insert Return into terminatorBlock
        ConversionPatternRewriter rewriter(funcOp->getContext());
        rewriter.setInsertionPointToEnd(terminatorBlock);
        // insert teardown function calls for push and pull ops
        ModuleOp parent = funcOp->getParentOfType<ModuleOp>();
        for (PushOp push : pushOps)
            if (failed(insertTeardownFunctionFromPush(parent, push, rewriter)))
                return WalkResult::interrupt();
        for (PullOp pull : pullOps)
            if (failed(insertTeardownFunctionFromPull(parent, pull, rewriter)))
                return WalkResult::interrupt();
        rewriter.create<func::ReturnOp>(funcOp->getLoc());

        for (PushOp replace : pushOps) {
            ConversionPatternRewriter pushRewriter(replace->getContext());
            pushRewriter.setInsertionPointAfter(replace);
            if (failed(rewritePushOp(replace, terminatorBlock, pushRewriter)))
                return WalkResult::interrupt();
        }

        for (PullOp replace : pullOps) {
            ConversionPatternRewriter pullRewriter(replace->getContext());
            pullRewriter.setInsertionPointAfter(replace);
            if (failed(rewritePullOp(replace, terminatorBlock, pullRewriter)))
                return WalkResult::interrupt();
        }

        return WalkResult::advance();
    });

    if (res.wasInterrupted()) signalPassFailure();

    // =======================================================================

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToLLVMConversionPatterns(converter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns,
        converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        LLVM::LLVMDialect>();

    target.addIllegalDialect<DfgDialect>();
    // FIXME: Remove this hack
    target.addLegalOp<ChannelOp>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return op.getOffloaded(); });

    // mark func dialect ops as illegal when they contain a dfg type
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
        return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
        return converter.isLegal(op.getOperandTypes());
    });
    // Mark BranchOps as illegal when they contain a dfg type
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
        return !llvm::isa<BranchOpInterface>(op)
               || converter.isLegal(op->getOperandTypes());
    });

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertDfgToLLVMPass()
{
    return std::make_unique<ConvertDfgToLLVMPass>();
}
