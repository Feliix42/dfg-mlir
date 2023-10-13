/// Implementation of the DfgToLLVM pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToLLVM/DfgToLLVM.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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

    LLVM::CallOp pushOperation = rewriter.create<LLVM::CallOp>(
        pushOp.getLoc(),
        ArrayRef<Type>(llvmVoid),
        pushFuncName,
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

    // fetch or create the FlatSymbolRefAttr for the called function
    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        llvmVoid,
        pullOp.getChan());

    LLVM::CallOp pushOperation = rewriter.create<LLVM::CallOp>(
        pullOp.getLoc(),
        ArrayRef<Type>(llvmVoid),
        pushFuncName,
        pullOp.getChan());

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

    LLVM::CallOp pushOperation = rewriter.create<LLVM::CallOp>(
        op->getLoc(),
        ArrayRef<Type>(boolReturnVal),
        pushFuncName,
        op->getOperands());

    // to handle the result of the push, we must do the following:
    // - split the current block after this instruction
    // - conditionally either continue processing the body or break the
    // computation

    Block* currentBlock = pushOperation->getBlock();

    // create the new block with the same argument list
    Block* newBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    SmallVector<Location> argLocs;
    for (auto arg : currentBlock->getArguments())
        argLocs.push_back(arg.getLoc());
    newBlock->addArguments(currentBlock->getArgumentTypes(), argLocs);

    // insert conditional jump to the break block
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::CondBranchOp>(
        pushOperation.getLoc(),
        pushOperation.getResult(),
        newBlock,
        terminatorBlock,
        currentBlock->getArguments());

    op->erase();

    return success();
}

// struct PushOpLowering : public OpConversionPattern<PushOp> {
//     using OpConversionPattern<PushOp>::OpConversionPattern;

//     PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
//             : OpConversionPattern<PushOp>(typeConverter, context){};

//     LogicalResult matchAndRewrite(
//         PushOp op,
//         PushOpAdaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         // create function name for PushOp in LLVM IR
//         std::string funcName = "push_";
//         llvm::raw_string_ostream funcNameStream(funcName);
//         // 0 is the type of the item sent
//         // TODO(feliix42): Add name mangling here
//         funcNameStream << adaptor.getOperands()[0].getType();

//         // return value
//         Type boolReturnVal = rewriter.getI1Type();

//         // fetch or create the FlatSymbolRefAttr for the called function
//         ModuleOp parentModule = op->getParentOfType<ModuleOp>();
//         FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
//             rewriter,
//             parentModule,
//             funcName,
//             boolReturnVal,
//             adaptor.getOperands());

//         LLVM::CallOp pushOperation = rewriter.create<LLVM::CallOp>(
//             op.getLoc(),
//             ArrayRef<Type>(boolReturnVal),
//             pushFuncName,
//             adaptor.getOperands());

//         // to handle the result of the push, we must do the following:
//         // - split the current block after this instruction
//         // - conditionally either continue processing the body or break the
//         // computation

//         Block* currentBlock = pushOperation->getBlock();

//         // create the new block with the same argument list
//         Block* newBlock =
//             rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
//         SmallVector<Location> argLocs;
//         for (auto arg : currentBlock->getArguments())
//             argLocs.push_back(arg.getLoc());
//         newBlock->addArguments(currentBlock->getArgumentTypes(), argLocs);

//         // insert conditional jump to the break block
//         Block* terminatorBlock = pushOperation->getSuccessors().back();
//         rewriter.setInsertionPointToEnd(currentBlock);
//         rewriter.create<cf::CondBranchOp>(
//             pushOperation.getLoc(),
//             pushOperation.getResult(),
//             newBlock,
//             terminatorBlock);
//         // TODO: If the above is not working: Is the way I get the
//         // terminatorBlock wrong?

//         rewriter.eraseOp(op);

//         return success();
//     }
// };

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

    LLVM::CallOp pullOperation = rewriter.create<LLVM::CallOp>(
        op.getLoc(),
        ArrayRef<Type>(returnedStruct),
        pullFuncName,
        op.getChan());

    // ===================================================================
    // TODO: Replace users of the results
    // insert into old block: unpacking
    // split block
    // block arguments: arguments of this block + result of unpacked value
    // conditional jump
    // REPLACE ALL USES
    // ===================================================================

    // this was taken from the pushOp lowering
    // Block* currentBlock = pullOperation->getBlock();

    // // create the new block with the same argument list
    // Block* newBlock =
    //     rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    // SmallVector<Location> argLocs;
    // for (auto arg : currentBlock->getArguments())
    //     argLocs.push_back(arg.getLoc());
    // newBlock->addArguments(currentBlock->getArgumentTypes(), argLocs);

    // // insert conditional jump to the break block
    // rewriter.setInsertionPointToEnd(currentBlock);
    // rewriter.create<cf::CondBranchOp>(
    //     pullOperation.getLoc(),
    //     pullOperation.getResult(),
    //     newBlock,
    //     terminatorBlock,
    //     currentBlock->getArguments());

    op->erase();

    return success();
}

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
        // insert into old block: unpacking
        // split block
        // block arguments: arguments of this block + result of unpacked value
        // conditional jump

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
    // patterns.add<PushOpLowering>(typeConverter, patterns.getContext());

    // dfg.pull -> llvm.call + logic
    patterns.add<PullOpLowering>(typeConverter, patterns.getContext());

    // dfg.channel -> !llvm.ptr
    // patterns.add<ChannelOp>(typeConverter, patterns.getContext());
}

void ConvertDfgToLLVMPass::runOnOperation()
{
    Operation* op = getOperation();

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
    // target.addIllegalOp<OperatorOp, PushOp, PullOp>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return op.getOffloaded(); });

    // if (failed(applyPartialConversion(op, target, std::move(patterns))))
    //     signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertDfgToLLVMPass()
{
    return std::make_unique<ConvertDfgToLLVMPass>();
}

// TODO:
// I changed the whole structure of this.
// - implement the rewritePullOp function
// - remove the pullOp lowering
// - fix the typerewrite to remove the channel types
// - continue with the TODOs from the other list