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
    OpBuilder &rewriter,
    ModuleOp module,
    std::string funcName,
    std::optional<Type> result,
    ValueRange arguments)
{
    auto* context = module.getContext();
    if (module.lookupSymbol<func::FuncOp>(funcName))
        return SymbolRefAttr::get(context, funcName);

    // convert the OperandRange into a list of types
    auto argIterator = arguments.getTypes();
    std::vector<Type> tyVec(argIterator.begin(), argIterator.end());

    // Create a function declaration for the desired function
    FunctionType fnType;
    if (result.has_value())
        fnType = rewriter.getFunctionType(tyVec, result.value());
    else
        fnType = rewriter.getFunctionType(tyVec, {});

    // Insert the function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<func::FuncOp>(module.getLoc(), funcName, fnType)
        .setPrivate();
    return SymbolRefAttr::get(context, funcName);
}

/// Returns the appropriate LLVM pointer representation for a channel.
Type getLLVMPointerFromChannelType(const Type &elementType)
{
    return LLVM::LLVMPointerType::get(elementType.getContext());
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
    OpBuilder &rewriter)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "close_channel";

    // fetch or create the FlatSymbolRefAttr for the called function
    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        std::nullopt,
        pushOp.getChan());

    rewriter.create<func::CallOp>(
        pushOp.getLoc(),
        pushFuncName,
        ArrayRef<Type>(),
        pushOp.getChan());

    return success();
}

LogicalResult insertTeardownFunctionFromPull(
    ModuleOp parentModule,
    PullOp pullOp,
    OpBuilder &rewriter)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "close_channel";

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
        std::nullopt,
        inputType.getResult(0));

    rewriter.create<func::CallOp>(
        pullOp.getLoc(),
        pushFuncName,
        ArrayRef<Type>(),
        inputType.getResult(0));

    return success();
}

LogicalResult
rewritePushOp(PushOp op, Block* terminatorBlock, IRRewriter &rewriter)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "push";

    Location loc = op.getLoc();
    // create a stack allocated data segment, write the data into it and call
    // the channel send function with this pointer
    auto one = rewriter.create<LLVM::ConstantOp>(
        loc,
        rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(1));
    auto ptrType =
        LLVM::LLVMPointerType::get(op.getInp().getType().getContext());
    auto allocated = rewriter.create<LLVM::AllocaOp>(
        loc,
        ptrType,
        op.getInp().getType(),
        one);
    rewriter.create<LLVM::StoreOp>(loc, op.getInp(), allocated);

    // return value
    Type boolReturnVal = rewriter.getI1Type();

    // TODO: The function declaration can be static: 2 pointers. Hence, this can
    // be simplified, here and elsewhere

    // fetch or create the FlatSymbolRefAttr for the called function
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    SmallVector<Value> arguments;
    arguments.push_back(op.getChan());
    arguments.push_back(op.getChan());
    // arguments.push_back(allocated.getResult());

    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        boolReturnVal,
        arguments);

    LLVM::BitcastOp casted = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(),
        LLVM::LLVMPointerType::get(op.getContext()),
        allocated);

    SmallVector<Value> args2;
    args2.push_back(op.getChan());
    args2.push_back(casted);

    func::CallOp pushOperation = rewriter.create<func::CallOp>(
        loc,
        pushFuncName,
        ArrayRef<Type>(boolReturnVal),
        args2);

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

    // if the terminator block has arguments, insert poison values
    SmallVector<Value> terminatorArgs;
    if (terminatorBlock->getNumArguments() > 0) {
        for (auto ty : terminatorBlock->getArgumentTypes()) {
            LLVM::PoisonOp poison = rewriter.create<LLVM::PoisonOp>(loc, ty);
            terminatorArgs.push_back(poison);
        }
    }
    rewriter.create<cf::CondBranchOp>(
        pushOperation.getLoc(),
        pushOperation.getResult(0),
        newBlock,
        /* trueArgs = */ ArrayRef<Value>(),
        terminatorBlock,
        /* falseArgs = */ terminatorArgs);

    op->erase();

    return success();
}

LogicalResult
rewritePullOp(PullOp op, Block* terminatorBlock, IRRewriter &rewriter)
{
    // create function name for PullOp in LLVM IR
    std::string funcName = "pull";

    Location loc = op.getLoc();
    // create a stack allocated data segment, write the data into it and call
    // the channel send function with this pointer
    auto one = rewriter.create<LLVM::ConstantOp>(
        loc,
        rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(1));
    auto ptrType = LLVM::LLVMPointerType::get(op.getType().getContext());
    auto allocated =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, op.getType(), one);

    // create the struct type that models the result
    Type boolReturnVal = rewriter.getI1Type();

    SmallVector<Value> arguments;
    arguments.push_back(op.getChan());
    // arguments.push_back(allocated);
    arguments.push_back(op.getChan());

    // fetch or create the FlatSymbolRefAttr for the called function
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr pullFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        boolReturnVal,
        arguments);

    LLVM::BitcastOp casted = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(),
        LLVM::LLVMPointerType::get(op.getContext()),
        allocated);

    SmallVector<Value> args2;
    args2.push_back(op.getChan());
    args2.push_back(casted);

    func::CallOp valid = rewriter.create<func::CallOp>(
        loc,
        ArrayRef<Type>(boolReturnVal),
        pullFuncName,
        args2);

    LLVM::LoadOp value =
        rewriter.create<LLVM::LoadOp>(loc, op.getType(), allocated);

    op.getResult().replaceAllUsesWith(value.getResult());

    Block* currentBlock = value->getBlock();

    // create the new block with everything following this pull op
    Block* newBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    // Originally, I added the Block arguments of the current block to the next
    // block. But because of the nature of SSA value scoping, that's not
    // necessary. All values of a block are visible in the successor blocks.

    // insert conditional jump to the break block
    rewriter.setInsertionPointToEnd(currentBlock);

    // if the terminator block has arguments, insert poison values
    SmallVector<Value> terminatorArgs;
    if (terminatorBlock->getNumArguments() > 0) {
        for (auto ty : terminatorBlock->getArgumentTypes()) {
            LLVM::PoisonOp poison = rewriter.create<LLVM::PoisonOp>(loc, ty);
            terminatorArgs.push_back(poison);
        }
    }
    rewriter.create<cf::CondBranchOp>(
        valid.getLoc(),
        valid.getResult(0),
        newBlock,
        ArrayRef<Value>(),
        // blockArgs,
        terminatorBlock,
        terminatorArgs);

    op->erase();

    return success();
}

struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // 1. create a new CallOp calling the channel creation function
        Type encapsulatedType = adaptor.getEncapsulatedType();
        std::string funcName = "channel";

        // Compute the byte width of the channel
        LLVM::LLVMPointerType nullPtrTy =
            LLVM::LLVMPointerType::get(encapsulatedType.getContext());

        LLVM::ZeroOp nullOp =
            rewriter.create<LLVM::ZeroOp>(op.getLoc(), nullPtrTy);
        LLVM::GEPOp gepResult = rewriter.create<LLVM::GEPOp>(
            op.getLoc(),
            nullPtrTy,
            nullPtrTy,
            nullOp.getResult(),
            ArrayRef<LLVM::GEPArg>{1});
        LLVM::PtrToIntOp bytewidthOp = rewriter.create<LLVM::PtrToIntOp>(
            op.getLoc(),
            rewriter.getI64Type(),
            gepResult.getResult());

        // I want to use the original encapsulated type here
        Type returnedPtr = getLLVMPointerFromChannelType(encapsulatedType);

        ModuleOp parentModule = op->getParentOfType<ModuleOp>();
        // 2. insert the appropriate definition at the top
        FlatSymbolRefAttr chanFunctionName = getOrInsertFunc(
            rewriter,
            parentModule,
            funcName,
            returnedPtr,
            ArrayRef<Value>(bytewidthOp.getResult()));

        func::CallOp channelCreation = rewriter.create<func::CallOp>(
            op.getLoc(),
            ArrayRef<Type>{returnedPtr},
            chanFunctionName,
            ArrayRef<Value>(bytewidthOp.getResult()));

        // 2.1 create 2 UnrealizedConversionCastOps that cast the resulting llvm
        // struct to an input and an output
        // 3. replace all channel inputs and outputs with the newly created
        // results
        UnrealizedConversionCastOp convertedInput =
            rewriter.create<UnrealizedConversionCastOp>(
                op.getLoc(),
                rewriter.getType<InputType>(encapsulatedType),
                channelCreation.getResult(0));
        op.getInChan().replaceAllUsesWith(convertedInput.getResult(0));

        UnrealizedConversionCastOp convertedOutput =
            rewriter.create<UnrealizedConversionCastOp>(
                op.getLoc(),
                rewriter.getType<OutputType>(encapsulatedType),
                channelCreation.getResult(0));
        op.getOutChan().replaceAllUsesWith(convertedOutput.getResult(0));

        // 4. delete the old op
        rewriter.eraseOp(op);

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

void ConvertDfgToLLVMPass::runOnOperation()
{
    Operation* op = getOperation();

    TypeConverter converter;

    // this is a stack!! Hence, the most generic conversion lives at the bottom
    converter.addConversion([](Type t) { return t; });
    converter.addConversion([](OutputType t) {
        return getLLVMPointerFromChannelType(t.getElementType());
    });
    converter.addConversion([](InputType t) {
        return getLLVMPointerFromChannelType(t.getElementType());
    });
    converter.addSourceMaterialization(
        [&](OpBuilder &builder,
            Type resultType,
            ValueRange inputs,
            Location loc) -> Value {
            if (inputs.size() != 1) return Value{};

            return builder
                .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                .getResult(0);
        });
    converter.addTargetMaterialization(
        [&](OpBuilder &builder,
            Type resultType,
            ValueRange inputs,
            Location loc) -> Value {
            if (inputs.size() != 1) return Value{};

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
        if (funcOp.getNumResults() > 0) {
            SmallVector<Location> locs;
            for (unsigned i = 0; i < funcOp.getNumResults(); i++)
                locs.push_back(funcOp.getLoc());
            terminatorBlock->addArguments(funcOp.getResultTypes(), locs);
        }

        // search all returns, forward to TerminatorBlock
        funcOp->walk([&](func::ReturnOp ret) {
            // ConversionPatternRewriter localRewriter(ret->getContext());
            OpBuilder localRewriter(ret->getContext());
            localRewriter.setInsertionPoint(ret);

            // NOTE(feliix42): replaceWithNewOp does *not* work here!
            // localRewriter.create<cf::BranchOp>(ret->getLoc(),
            // terminatorBlock);
            localRewriter.create<cf::BranchOp>(
                ret->getLoc(),
                ret.getOperands(),
                terminatorBlock);
            ret->erase();
        });

        // insert Return into terminatorBlock
        // ConversionPatternRewriter rewriter(funcOp->getContext());
        OpBuilder rewriter(funcOp->getContext());
        rewriter.setInsertionPointToEnd(terminatorBlock);
        // insert teardown function calls for push and pull ops
        ModuleOp parent = funcOp->getParentOfType<ModuleOp>();

        SmallPtrSet<Value, 16> inpSet;
        for (PushOp push : pushOps) {
            Value in = push.getChan();
            if (!inpSet.contains(in)) {
                inpSet.insert(in);
                if (failed(
                        insertTeardownFunctionFromPush(parent, push, rewriter)))
                    return WalkResult::interrupt();
            }
        }
        SmallPtrSet<Value, 16> outSet;
        for (PullOp pull : pullOps) {
            Value out = pull.getChan();
            if (!outSet.contains(out)) {
                outSet.insert(out);
                if (failed(
                        insertTeardownFunctionFromPull(parent, pull, rewriter)))
                    return WalkResult::interrupt();
            }
        }
        rewriter.create<func::ReturnOp>(
            funcOp->getLoc(),
            terminatorBlock->getArguments());

        for (PushOp replace : pushOps) {
            // ConversionPatternRewriter pushRewriter(replace->getContext());
            IRRewriter pushRewriter(replace->getContext());
            pushRewriter.setInsertionPointAfter(replace);
            if (failed(rewritePushOp(replace, terminatorBlock, pushRewriter)))
                return WalkResult::interrupt();
        }

        for (PullOp replace : pullOps) {
            // ConversionPatternRewriter pullRewriter(replace->getContext());
            IRRewriter pullRewriter(replace->getContext());
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
    // target.addDynamicallyLegalOp<InstantiateOp>(
    //     [](InstantiateOp op) { return op.getOffloaded(); });

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
