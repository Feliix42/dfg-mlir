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
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
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
    func::FuncOp::create(rewriter, module.getLoc(), funcName, fnType)
        .setPrivate();
    return SymbolRefAttr::get(context, funcName);
}

/// Returns the appropriate LLVM pointer representation for a channel.
Type getLLVMPointerFromChannelType(const Type &elementType)
{
    return LLVM::LLVMPointerType::get(elementType.getContext());
}

/// Returns the element type encapsulated by a channel type.
static Type getChannelElementType(Type channelType)
{
    if (auto inputType = dyn_cast<InputType>(channelType))
        return inputType.getElementType();
    return cast<OutputType>(channelType).getElementType();
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
    Value chan,
    Location loc,
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
        chan);

    func::CallOp::create(
        rewriter,
        loc,
        pushFuncName,
        ArrayRef<Type>(),
        chan);

    return success();
}

LogicalResult insertTeardownFunctionFromPull(
    ModuleOp parentModule,
    Value chan,
    Type elemType,
    Location loc,
    OpBuilder &rewriter)
{
    // create function name for PullOp in LLVM IR
    std::string funcName = "close_channel";

    // Translate to InputType, insert UnrealizedConversionCastOp
    UnrealizedConversionCastOp inputType =
        UnrealizedConversionCastOp::create(
            rewriter,
            loc,
            rewriter.getType<InputType>(elemType),
            chan);

    // fetch or create the FlatSymbolRefAttr for the called function
    FlatSymbolRefAttr pushFuncName = getOrInsertFunc(
        rewriter,
        parentModule,
        funcName,
        std::nullopt,
        inputType.getResult(0));

    func::CallOp::create(
        rewriter,
        loc,
        pushFuncName,
        ArrayRef<Type>(),
        inputType.getResult(0));

    return success();
}

/// Builds the operand list for a branch to the terminator block: poison
/// values for the function result arguments, and for each channel that is
/// passed as a block argument either the channel value (where it dominates
/// the branch site) or a null pointer (close_channel handles NULL).
static SmallVector<Value> buildTerminatorArgs(
    Block* terminatorBlock,
    unsigned numFuncResults,
    ArrayRef<Value> chanSrcs,
    ArrayRef<bool> chanUseValue,
    Location loc,
    OpBuilder &rewriter)
{
    SmallVector<Value> args;
    for (unsigned i = 0; i < numFuncResults; i++) {
        LLVM::PoisonOp poison = LLVM::PoisonOp::create(
            rewriter, loc, terminatorBlock->getArgument(i).getType());
        args.push_back(poison);
    }
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    for (auto [src, useValue] : llvm::zip(chanSrcs, chanUseValue)) {
        if (useValue) {
            auto cast =
                UnrealizedConversionCastOp::create(rewriter, loc, ptrType, src);
            args.push_back(cast.getResult(0));
        } else {
            args.push_back(
                LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult());
        }
    }
    return args;
}

LogicalResult
rewritePushOp(PushOp op, Block* terminatorBlock, unsigned numFuncResults,
              ArrayRef<Value> chanSrcs, ArrayRef<bool> chanUseValue,
              IRRewriter &rewriter, bool closeAfterPush = false)
{
    // create function name for PushOp in LLVM IR
    std::string funcName = "push";

    Location loc = op.getLoc();
    // create a stack allocated data segment, write the data into it and call
    // the channel send function with this pointer
    auto one = LLVM::ConstantOp::create(
        rewriter,
        loc,
        rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(1));
    auto ptrType =
        LLVM::LLVMPointerType::get(op.getInp().getType().getContext());
    auto allocated = LLVM::AllocaOp::create(
        rewriter,
        loc,
        ptrType,
        op.getInp().getType(),
        one);
    LLVM::StoreOp::create(rewriter, loc, op.getInp(), allocated);

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

    LLVM::BitcastOp casted = LLVM::BitcastOp::create(
        rewriter,
        op.getLoc(),
        LLVM::LLVMPointerType::get(op.getContext()),
        allocated);

    SmallVector<Value> args2;
    args2.push_back(op.getChan());
    args2.push_back(casted);

    func::CallOp pushOperation = func::CallOp::create(
        rewriter,
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

    SmallVector<Value> terminatorArgs =
        buildTerminatorArgs(terminatorBlock, numFuncResults, chanSrcs,
                            chanUseValue, loc, rewriter);
    cf::CondBranchOp::create(
        rewriter,
        pushOperation.getLoc(),
        pushOperation.getResult(0),
        newBlock,
        /* trueArgs = */ ArrayRef<Value>(),
        terminatorBlock,
        /* falseArgs = */ terminatorArgs);

    if (closeAfterPush) {
        // One-shot push in the graph-instantiating function: terminate the
        // channel input right after the send has happened (i.e. on the
        // success path of the push, at the start of the continuation block),
        // instead of deferring the close to the terminator block at the end
        // of the function.
        rewriter.setInsertionPointToStart(newBlock);
        if (failed(insertTeardownFunctionFromPush(
                parentModule, op.getChan(), op.getLoc(), rewriter)))
            return failure();
    }

    op->erase();

    return success();
}

LogicalResult
rewritePullOp(PullOp op, Block* terminatorBlock, unsigned numFuncResults,
              ArrayRef<Value> chanSrcs, ArrayRef<bool> chanUseValue,
              IRRewriter &rewriter)
{
    // create function name for PullOp in LLVM IR
    std::string funcName = "pull";

    Location loc = op.getLoc();
    // create a stack allocated data segment, write the data into it and call
    // the channel send function with this pointer
    auto one = LLVM::ConstantOp::create(
        rewriter,
        loc,
        rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(1));
    auto ptrType = LLVM::LLVMPointerType::get(op.getType().getContext());
    auto allocated =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, op.getType(), one);

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

    LLVM::BitcastOp casted = LLVM::BitcastOp::create(
        rewriter,
        op.getLoc(),
        LLVM::LLVMPointerType::get(op.getContext()),
        allocated);

    SmallVector<Value> args2;
    args2.push_back(op.getChan());
    args2.push_back(casted);

    func::CallOp valid = func::CallOp::create(
        rewriter,
        loc,
        ArrayRef<Type>(boolReturnVal),
        pullFuncName,
        args2);

    LLVM::LoadOp value =
        LLVM::LoadOp::create(rewriter, loc, op.getType(), allocated);

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

    SmallVector<Value> terminatorArgs =
        buildTerminatorArgs(terminatorBlock, numFuncResults, chanSrcs,
                            chanUseValue, loc, rewriter);
    cf::CondBranchOp::create(
        rewriter,
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
            LLVM::ZeroOp::create(rewriter, op.getLoc(), nullPtrTy);
        LLVM::GEPOp gepResult = LLVM::GEPOp::create(
            rewriter,
            op.getLoc(),
            nullPtrTy,
            encapsulatedType,
            nullOp.getResult(),
            ArrayRef<LLVM::GEPArg>{1});
        LLVM::PtrToIntOp bytewidthOp = LLVM::PtrToIntOp::create(
            rewriter,
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

        func::CallOp channelCreation = func::CallOp::create(
            rewriter,
            op.getLoc(),
            ArrayRef<Type>{returnedPtr},
            chanFunctionName,
            ArrayRef<Value>(bytewidthOp.getResult()));

        // 2.1 create 2 UnrealizedConversionCastOps that cast the resulting llvm
        // struct to an input and an output
        // 3. replace all channel inputs and outputs with the newly created
        // results
        UnrealizedConversionCastOp convertedInput =
            UnrealizedConversionCastOp::create(
                rewriter,
                op.getLoc(),
                rewriter.getType<InputType>(encapsulatedType),
                channelCreation.getResult(0));
        op.getInChan().replaceAllUsesWith(convertedInput.getResult(0));

        UnrealizedConversionCastOp convertedOutput =
            UnrealizedConversionCastOp::create(
                rewriter,
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

            return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
                .getResult(0);
        });
    converter.addTargetMaterialization(
        [&](OpBuilder &builder,
            Type resultType,
            ValueRange inputs,
            Location loc) -> Value {
            if (inputs.size() != 1) return Value{};

            return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
                .getResult(0);
        });

    // =========================================================================================
    // In the following block, we do multiple things in one go:
    // 0. create an empty terminator block for any functions with dfg.pull/push
    // ops.
    // 1. lower `PushOp` and `PullOp` into their respective `LLVM::CallOp`s
    // 2. insert the necessary teardown logic for channels
    //    - channels pulled from / pushed to inside the nodes are closed in the
    //      terminator block (i.e. when the function exits)
    //    - "one-shot" push channels in the top-level function that instantiates
    //      the graph (pushes before the first instantiation) are closed right
    //      after the send has happened
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
        // Channels whose input side is only pushed *before* the graph is
        // instantiated in this function ("one-shot" pushes in the top-level
        // function, e.g. after the channels were created and before the
        // instantiation starts). Those inputs are terminated right after the
        // send instead of in the terminator block.
        SmallPtrSet<Value, 16> oneShotChans;
        bool seenInstantiation = false;
        funcOp->walk([&](Operation* innerOp) -> WalkResult {
            // In the top-level function, the instantiation of the graph is
            // either still present as dfg.instantiate, already lowered to
            // an omp.parallel region (by convert-dfg-nodes-to-func with the
            // default omp backend), or lowered to raw pthread_create calls
            // (parallel-backend=pthread). Pushes before that point are
            // one-shot pushes.
            if (isa<InstantiateOp, omp::ParallelOp>(innerOp)) {
                seenInstantiation = true;
            } else if (auto call = dyn_cast<LLVM::CallOp>(innerOp)) {
                // pthread backend: the graph nodes are spawned via
                // pthread_create calls emitted by PthreadSpawnStrategy
                FlatSymbolRefAttr callee = call.getCalleeAttr();
                if (callee && callee.getValue() == "pthread_create")
                    seenInstantiation = true;
            } else if (auto push = dyn_cast<PushOp>(innerOp)) {
                pushOps.push_back(push);
                if (seenInstantiation)
                    oneShotChans.erase(push.getChan());
                else
                    oneShotChans.insert(push.getChan());
            } else if (auto pull = dyn_cast<PullOp>(innerOp)) {
                pullOps.push_back(pull);
            }
            return WalkResult::advance();
        });
        // only a function that actually instantiates the graph has one-shot
        // pushes
        if (!seenInstantiation) oneShotChans.clear();

        if (pullOps.empty() && pushOps.empty()) return WalkResult::advance();
        // if (pushOps.empty()) return WalkResult::advance();

        // Collect the channels whose teardown is inserted into the
        // terminator block (deduplicated; one-shot push channels are closed
        // at their push site instead).
        SmallVector<Value> teardownChans;
        SmallPtrSet<Value, 16> seenChans;
        for (PushOp push : pushOps) {
            Value in = push.getChan();
            if (oneShotChans.contains(in)) continue;
            if (seenChans.insert(in).second) teardownChans.push_back(in);
        }
        for (PullOp pull : pullOps) {
            Value out = pull.getChan();
            if (seenChans.insert(out).second) teardownChans.push_back(out);
        }

        // A teardown channel can be referenced directly inside the
        // terminator block only if its definition dominates every block that
        // branches to the terminator block (i.e. every push/pull site and
        // every return). This does not hold when a channel is created only
        // after a push that may fail (e.g. loops with input channels create
        // their output channel after the graph was instantiated). Such
        // channels are passed as additional terminator block arguments
        // instead; edges where the channel is not yet available pass a
        // null pointer (close_channel handles NULL).
        DominanceInfo domInfo(funcOp);
        SmallVector<Operation*> sites;
        for (PushOp push : pushOps) sites.push_back(push);
        for (PullOp pull : pullOps) sites.push_back(pull);
        funcOp->walk([&](func::ReturnOp ret) { sites.push_back(ret); });

        // does the channel value dominate the given site? (block arguments,
        // e.g. lowered scf.for results used as channels, only dominate from
        // their own block onwards)
        auto dominatesSite = [&](Value chan, Operation* site) -> bool {
            if (Operation* def = chan.getDefiningOp())
                return domInfo.properlyDominates(def, site);
            if (auto arg = dyn_cast<BlockArgument>(chan))
                return domInfo.dominates(arg.getOwner(), site->getBlock());
            return true; // function argument
        };

        auto dominatesAllSites = [&](Value chan) -> bool {
            for (Operation* site : sites)
                if (!dominatesSite(chan, site)) return false;
            return true;
        };
        // channels that need to be passed as terminator block arguments
        SmallVector<Value> chanSrcs;
        SmallVector<Type> chanArgTypes;
        for (Value chan : teardownChans) {
            if (dominatesAllSites(chan)) continue;
            chanSrcs.push_back(chan);
            chanArgTypes.push_back(
                getLLVMPointerFromChannelType(getChannelElementType(chan.getType())));
        }

        auto useValueAt = [&](Operation* site) {
            SmallVector<bool> flags;
            for (Value src : chanSrcs)
                flags.push_back(dominatesSite(src, site));
            return flags;
        };

        Block* terminatorBlock = funcOp.addBlock();
        unsigned numFuncResults = funcOp.getNumResults();
        if (numFuncResults > 0) {
            SmallVector<Location> locs;
            for (unsigned i = 0; i < numFuncResults; i++)
                locs.push_back(funcOp.getLoc());
            terminatorBlock->addArguments(funcOp.getResultTypes(), locs);
        }
        if (!chanArgTypes.empty()) {
            SmallVector<Location> locs(chanArgTypes.size(), funcOp.getLoc());
            terminatorBlock->addArguments(chanArgTypes, locs);
        }
        SmallVector<Value> chanArgs;
        for (unsigned i = 0; i < chanSrcs.size(); i++)
            chanArgs.push_back(terminatorBlock->getArgument(numFuncResults + i));

        // search all returns, forward to TerminatorBlock
        funcOp->walk([&](func::ReturnOp ret) {
            // ConversionPatternRewriter localRewriter(ret->getContext());
            OpBuilder localRewriter(ret->getContext());
            localRewriter.setInsertionPoint(ret);

            // NOTE(feliix42): replaceWithNewOp does *not* work here!
            // localRewriter.create<cf::BranchOp>(ret->getLoc(),
            // terminatorBlock);
            SmallVector<Value> operands(ret.getOperands());
            auto ptrType = LLVM::LLVMPointerType::get(ret->getContext());
            for (auto [src, useValue] :
                 llvm::zip(chanSrcs, useValueAt(ret))) {
                if (useValue) {
                    operands.push_back(
                        UnrealizedConversionCastOp::create(
                            localRewriter, ret->getLoc(), ptrType, src)
                            .getResult(0));
                } else {
                    operands.push_back(LLVM::ZeroOp::create(
                                           localRewriter, ret->getLoc(), ptrType)
                                           .getResult());
                }
            }
            cf::BranchOp::create(
                localRewriter,
                ret->getLoc(),
                operands,
                terminatorBlock);
            ret->erase();
        });

        // insert Return into terminatorBlock
        // ConversionPatternRewriter rewriter(funcOp->getContext());
        OpBuilder rewriter(funcOp->getContext());
        rewriter.setInsertionPointToEnd(terminatorBlock);
        // insert teardown function calls for push and pull ops
        ModuleOp parent = funcOp->getParentOfType<ModuleOp>();

        // for channels passed as block arguments, close the argument (cast
        // back to the channel type); all other channels are closed directly
        auto closeValueFor = [&](Value chan) -> Value {
            for (auto [src, arg] : llvm::zip(chanSrcs, chanArgs)) {
                if (src == chan) {
                    return UnrealizedConversionCastOp::create(
                               rewriter, funcOp.getLoc(), chan.getType(), arg)
                        .getResult(0);
                }
            }
            return chan;
        };

        SmallPtrSet<Value, 16> inpSet;
        for (PushOp push : pushOps) {
            Value in = push.getChan();
            // one-shot channels are closed right after their push instead
            if (oneShotChans.contains(in)) continue;
            if (!inpSet.contains(in)) {
                inpSet.insert(in);
                if (failed(insertTeardownFunctionFromPush(
                        parent, closeValueFor(in), push.getLoc(), rewriter)))
                    return WalkResult::interrupt();
            }
        }
        SmallPtrSet<Value, 16> outSet;
        for (PullOp pull : pullOps) {
            Value out = pull.getChan();
            if (!outSet.contains(out)) {
                outSet.insert(out);
                if (failed(insertTeardownFunctionFromPull(
                        parent, closeValueFor(out), pull.getType(),
                        pull.getLoc(), rewriter)))
                    return WalkResult::interrupt();
            }
        }
        SmallVector<Value> retArgs;
        for (unsigned i = 0; i < numFuncResults; i++)
            retArgs.push_back(terminatorBlock->getArgument(i));
        func::ReturnOp::create(rewriter, funcOp->getLoc(), retArgs);

        for (PushOp replace : pushOps) {
            // ConversionPatternRewriter pushRewriter(replace->getContext());
            IRRewriter pushRewriter(replace->getContext());
            pushRewriter.setInsertionPointAfter(replace);
            bool closeAfterPush = oneShotChans.contains(replace.getChan());
            if (failed(rewritePushOp(replace, terminatorBlock, numFuncResults,
                                     chanSrcs, useValueAt(replace), pushRewriter,
                                     closeAfterPush)))
                return WalkResult::interrupt();
        }

        for (PullOp replace : pullOps) {
            // ConversionPatternRewriter pullRewriter(replace->getContext());
            IRRewriter pullRewriter(replace->getContext());
            pullRewriter.setInsertionPointAfter(replace);
            if (failed(rewritePullOp(replace, terminatorBlock, numFuncResults,
                                     chanSrcs, useValueAt(replace),
                                     pullRewriter)))
                return WalkResult::interrupt();
        }

        return WalkResult::advance();
    });

    if (res.wasInterrupted()) signalPassFailure();

    // =======================================================================
    // Pre-convert dfg-typed values that cross block boundaries.
    //
    // The dialect conversion framework's branch pattern cannot build a
    // replacement for channel-typed branch operands (remapValues fails on
    // convertType of the operand value), so loops whose scf.for carried a
    // state channel (lowered to cf blocks with channel-typed block
    // arguments and branch operands) would not legalize. Convert these
    // values manually with unrealized_conversion_cast, mirroring what
    // ChannelOpLowering does for channel results: non-entry block arguments
    // of dfg type become !llvm.ptr (with a cast back to the dfg type at the
    // start of the block), and dfg-typed branch operands are cast to
    // !llvm.ptr right before the branch.
    {
        SmallVector<Block *> blocksToFix;
        struct BranchOperandFix {
            Operation *op;
            unsigned idx;
            Type newType;
        };
        SmallVector<BranchOperandFix> branchFixes;
        op->walk([&](Operation *inner) {
            for (Region &region : inner->getRegions())
                for (Block &block : region)
                    if (!block.isEntryBlock() &&
                        llvm::any_of(block.getArguments(),
                                     [](BlockArgument arg) {
                                         return isa<InputType, OutputType>(
                                             arg.getType());
                                     }))
                        blocksToFix.push_back(&block);
            if (auto branch = dyn_cast<BranchOpInterface>(inner))
                for (auto [idx, operand] : llvm::enumerate(inner->getOperands()))
                    if (isa<InputType, OutputType>(operand.getType()))
                        branchFixes.push_back(
                            {branch.getOperation(), (unsigned)idx,
                             getLLVMPointerFromChannelType(
                                 getChannelElementType(operand.getType()))});
            return WalkResult::advance();
        });

        IRRewriter rewriter(op->getContext());
        for (Block *block : blocksToFix) {
            for (BlockArgument arg : block->getArguments()) {
                Type oldType = arg.getType();
                if (!isa<InputType, OutputType>(oldType))
                    continue;
                Type newType =
                    getLLVMPointerFromChannelType(getChannelElementType(oldType));
                arg.setType(newType);
                rewriter.setInsertionPointToStart(block);
                Value cast = UnrealizedConversionCastOp::create(
                    rewriter, arg.getLoc(), oldType, arg).getResult(0);
                arg.replaceAllUsesExcept(cast, cast.getDefiningOp());
            }
        }
        for (const BranchOperandFix &fix : branchFixes) {
            rewriter.setInsertionPoint(fix.op);
            Value cast = UnrealizedConversionCastOp::create(
                rewriter, fix.op->getLoc(), fix.newType,
                fix.op->getOperand(fix.idx)).getResult(0);
            fix.op->setOperand(fix.idx, cast);
        }
    }

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
