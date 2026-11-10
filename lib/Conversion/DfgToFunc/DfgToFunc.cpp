/// Implementation of the DfgToFunc pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "dfg-mlir/Conversion/PassesEnums.cpp.inc"

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
    ConvertDfgToFuncPass() = default;
    explicit ConvertDfgToFuncPass(ConvertDfgToFuncOptions options)
            : mlir::impl::ConvertDfgToFuncBase<ConvertDfgToFuncPass>(
                  std::move(options)) {}

    void runOnOperation() final;
};

/// Spawn strategy that lowers parallel graph execution to OpenMP dialect
/// operations: each group of successive instantiations is wrapped in an
/// `omp.parallel` (with num_threads set to the group size) containing
/// `omp.sections`, and each instantiation becomes a `func.call` inside an
/// `omp.section`.
struct OmpSpawnStrategy final : public SpawnStrategy {
    void emitParallelGroup(OpBuilder &builder, Location loc,
                           ArrayRef<InstantiateOp> group) const override
    {
        arith::ConstantOp threadCount = arith::ConstantOp::create(
            builder,
            loc,
            builder.getI32Type(),
            builder.getI32IntegerAttr(group.size()));

        omp::ParallelOp par = omp::ParallelOp::create(builder, loc);
        par.getNumThreadsMutable().assign(threadCount.getResult());
        Block* parBlock = builder.createBlock(&par.getRegion());

        omp::SectionsOp sections = omp::SectionsOp::create(
            builder,
            loc,
            ArrayRef<Type>(),
            ValueRange());
        sections.setNowait(true);
        Block* sectionsBlock = builder.createBlock(&sections.getRegion());

        // move instantiates here
        for (auto op : group) {
            InstantiateOp::create(
                builder,
                op->getLoc(),
                op.getNodeName(),
                op.getInputs(),
                op.getOutputs(),
                op.getOffloadedAttr().getValue());
            op->erase();
        }

        builder.setInsertionPointToEnd(sectionsBlock);
        omp::TerminatorOp::create(builder, loc);

        builder.setInsertionPointToEnd(parBlock);
        omp::TerminatorOp::create(builder, loc);
    }

    LogicalResult emitInstantiation(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // don't lower offloaded functions
        if (adaptor.getOffloaded()) {
            emitWarning(
                op.getLoc(),
                "This lowering does not handle the conversion of offloaded "
                "operators. Please use a different conversion pattern "
                "beforehand to translate these operations into legal inputs.");
            return failure();
        }

        omp::SectionOp sec = omp::SectionOp::create(rewriter, op.getLoc());
        Region &sectionRegion = sec.getRegion();
        Block* secBlock = rewriter.createBlock(&sectionRegion);

        rewriter.setInsertionPointToStart(secBlock);
        func::CallOp::create(
            rewriter,
            op.getLoc(),
            adaptor.getCallee(),
            ArrayRef<Type>(),
            adaptor.getOperands());
        omp::TerminatorOp::create(rewriter, op.getLoc());

        rewriter.eraseOp(op);

        return success();
    }
};

/// Get (or insert at module level) an external `llvm.func` declaration.
static LLVM::LLVMFuncOp getOrInsertLLVMFunc(ModuleOp module, StringRef name,
                                            LLVM::LLVMFunctionType type)
{
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return fn;
    OpBuilder b(module.getBody(), module.getBody()->end());
    return LLVM::LLVMFuncOp::create(b, module.getLoc(), name, type,
                                   LLVM::Linkage::External);
}

/// Spawn strategy that lowers parallel graph execution to raw pthread calls
/// (LLVM dialect), one thread per instantiation, all emitted in this pass:
///
/// - Per instantiation, a thunk is emitted at module level:
///   `func.func @dfg_thunk_<node>_<n>(%packed: !llvm.ptr)` whose body unpacks the
///   packed channel pointers (extractvalue + unrealized_conversion_cast back to
///   the dfg channel types), calls the node, and returns (returning from the
///   start routine is equivalent to pthread_exit(NULL)).
/// - At the group site, per instantiation: the channel args are cast
///   (!dfg.* -> !llvm.ptr) and packed into an alloca'd `!llvm.struct<(!llvm.ptr, ...)>`;
///   then `llvm.call @pthread_create(tid*, null, llvm.addressof @thunk, args*)`.
/// - After all spawns of the group: one `llvm.call @pthread_join(tid, null)` per
///   thread (pthread has no group join; this matches the implicit barrier at the
///   end of omp.parallel).
///
/// Notes: `pthread_t` is treated as i64 (glibc LP64). The unrealized casts are
/// folded away by DfgToLLVM's type converter (!dfg.* -> !llvm.ptr) later.
struct PthreadSpawnStrategy final : public SpawnStrategy {
    void emitParallelGroup(OpBuilder &builder, Location loc,
                           ArrayRef<InstantiateOp> group) const override
    {
        MLIRContext *ctx = builder.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i64Ty = builder.getI64Type();
        ModuleOp module = group[0]->getParentOfType<ModuleOp>();

        LLVM::LLVMFuncOp createFn = getOrInsertLLVMFunc(
            module, "pthread_create",
            LLVM::LLVMFunctionType::get(builder.getI32Type(),
                                        {ptrTy, ptrTy, ptrTy, ptrTy}));
        LLVM::LLVMFuncOp joinFn = getOrInsertLLVMFunc(
            module, "pthread_join",
            LLVM::LLVMFunctionType::get(builder.getI32Type(), {i64Ty, ptrTy}));

        // emit all spawns before the first group op, then erase the group ops
        Block *block = group[0]->getBlock();
        Operation *afterGroup = group.back()->getNextNode();
        builder.setInsertionPoint(group[0]);

        SmallVector<Value> tidSlots;
        SmallVector<InstantiateOp> spawned;
        for (InstantiateOp op : group) {
            // offloaded instantiates are left in place; emitInstantiation
            // rejects them during conversion (same as the omp backend)
            if (op.getOffloaded()) continue;
            tidSlots.push_back(emitSpawn(builder, module, loc, op, createFn));
            spawned.push_back(op);
        }
        for (InstantiateOp op : spawned) op->erase();

        // join all spawned threads, in spawn order
        if (afterGroup)
            builder.setInsertionPoint(afterGroup);
        else
            builder.setInsertionPointToEnd(block);
        Value nullPtr = LLVM::ZeroOp::create(builder, loc, ptrTy);
        for (Value tidSlot : tidSlots) {
            Value tid = LLVM::LoadOp::create(builder, loc, i64Ty, tidSlot);
            LLVM::CallOp::create(builder, loc, joinFn, {tid, nullPtr});
        }
    }

    LogicalResult emitInstantiation(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // don't lower offloaded functions
        if (adaptor.getOffloaded()) {
            emitWarning(
                op.getLoc(),
                "This lowering does not handle the conversion of offloaded "
                "operators. Please use a different conversion pattern "
                "beforehand to translate these operations into legal inputs.");
            return failure();
        }

        // Non-offloaded instantiates are consumed by emitParallelGroup; this
        // is only a sequential fallback.
        func::CallOp::create(
            rewriter,
            op.getLoc(),
            adaptor.getCallee(),
            ArrayRef<Type>(),
            adaptor.getOperands());
        rewriter.eraseOp(op);

        return success();
    }

  private:
    /// Emit the thunk (module level) and the pthread_create call (group site)
    /// for one instantiation. Returns the alloca'd pthread_t slot.
    Value emitSpawn(OpBuilder &builder, ModuleOp module, Location loc,
                    InstantiateOp op, LLVM::LLVMFuncOp createFn) const
    {
        MLIRContext *ctx = builder.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i32Ty = builder.getI32Type();

        SmallVector<Value> chans(op.getInputs().begin(), op.getInputs().end());
        chans.append(op.getOutputs().begin(), op.getOutputs().end());
        SmallVector<Type> chanTys;
        for (Value v : chans) chanTys.push_back(v.getType());

        auto structTy = LLVM::LLVMStructType::getLiteral(
            ctx, SmallVector<Type>(chans.size(), ptrTy));

        // Thunk at module level (must be an llvm.func: llvm.addressof only
        // references llvm dialect symbols). Signature is the raw pthread start
        // routine `void* (*)(void*)`: one packed-args pointer in, null out.
        std::string thunkName =
            (llvm::Twine("dfg_thunk_") + op.getNodeName() + "_" +
             std::to_string(thunkCounter++))
                .str();
        OpBuilder mb(module.getBody(), module.getBody()->end());
        auto thunk = LLVM::LLVMFuncOp::create(
            mb, loc, thunkName,
            LLVM::LLVMFunctionType::get(ptrTy, {ptrTy}),
            LLVM::Linkage::Internal);
        Block *entry = mb.createBlock(&thunk.getBody(), {}, {ptrTy}, {loc});
        SmallVector<Value> callArgs;
        if (!chans.empty()) {
            Value packed =
                LLVM::LoadOp::create(mb, loc, structTy, entry->getArgument(0));
            for (size_t i = 0; i < chans.size(); ++i) {
                Value elem = LLVM::ExtractValueOp::create(
                    mb, loc, ptrTy, packed, {static_cast<int64_t>(i)});
                callArgs.push_back(UnrealizedConversionCastOp::create(
                                       mb, loc, chanTys[i], elem)
                                       .getResult(0));
            }
        }
        func::CallOp::create(
            mb, loc, FlatSymbolRefAttr::get(ctx, op.getNodeName()),
            TypeRange{}, callArgs);
        LLVM::ReturnOp::create(
            mb, loc, ValueRange{LLVM::ZeroOp::create(mb, loc, ptrTy)});

        // spawn at the group site: pack args, pthread_create(...)
        Value one =
            LLVM::ConstantOp::create(builder, loc, i32Ty, builder.getI32IntegerAttr(1));
        Value argsPtr;
        if (chans.empty()) {
            argsPtr = LLVM::ZeroOp::create(builder, loc, ptrTy);
        } else {
            Value argsSlot =
                LLVM::AllocaOp::create(builder, loc, ptrTy, structTy, one);
            Value packed = LLVM::UndefOp::create(builder, loc, structTy);
            for (size_t i = 0; i < chans.size(); ++i) {
                Value p = UnrealizedConversionCastOp::create(
                              builder, loc, ptrTy, chans[i])
                              .getResult(0);
                packed = LLVM::InsertValueOp::create(
                    builder, loc, structTy, packed, p,
                    {static_cast<int64_t>(i)});
            }
            LLVM::StoreOp::create(builder, loc, packed, argsSlot);
            argsPtr = argsSlot;
        }

        Value fnPtr = LLVM::AddressOfOp::create(
            builder, loc, ptrTy, FlatSymbolRefAttr::get(ctx, thunkName));
        Value tidSlot =
            LLVM::AllocaOp::create(builder, loc, ptrTy, builder.getI64Type(), one);
        Value nullPtr = LLVM::ZeroOp::create(builder, loc, ptrTy);
        LLVM::CallOp::create(builder, loc, createFn,
                             {tidSlot, nullPtr, fnPtr, argsPtr});
        return tidSlot;
    }

    mutable unsigned thunkCounter = 0;
};
} // namespace

std::unique_ptr<SpawnStrategy> mlir::createSpawnStrategy(ParallelBackend backend)
{
    switch (backend) {
    case ParallelBackend::Omp:
        return std::make_unique<OmpSpawnStrategy>();
    case ParallelBackend::Pthread:
        return std::make_unique<PthreadSpawnStrategy>();
    }
    llvm_unreachable("unknown parallel backend");
}

struct ProcessOpLowering : public OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ProcessOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
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
            func::FuncOp::create(rewriter, loc, operatorName, newFuncTy);

        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                adaptor.getBody(),
                genFuncOp.getBody(),
                genFuncOp.end());

            // populate the Terminator Block (for now) with a return only
            rewriter.setInsertionPointToEnd(&genFuncOp.getRegion().back());
            func::ReturnOp::create(rewriter, op.getLoc());
        } else {
            // set the function private for linking
            genFuncOp.setPrivate();
        }

        rewriter.eraseOp(op);

        return success();
    }
};

struct InstantiateOpLowering : public OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering(TypeConverter &typeConverter, MLIRContext* context,
                          const SpawnStrategy* spawnStrategy)
            : OpConversionPattern<InstantiateOp>(typeConverter, context),
              spawnStrategy(spawnStrategy){};

    LogicalResult matchAndRewrite(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        return spawnStrategy->emitInstantiation(op, adaptor, rewriter);
    }

    const SpawnStrategy* spawnStrategy;
};

struct LoopOpLowering : public OpConversionPattern<LoopOp> {
    using OpConversionPattern<LoopOp>::OpConversionPattern;

    LoopOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<LoopOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        LoopOp op,
        LoopOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO(feliix42):
        // - test that this warning is triggered!

        // Warn when not all channels are watched
        llvm::DenseSet<Value> regionArgs;
        Region &parentRegion = op.getRegion();
        for (auto item : parentRegion.getArguments()) regionArgs.insert(item);

        // FIXME(feliix42): What about channels created in the Loop (before the
        // loop is not possible)
        for (auto inItem : adaptor.getInChans()) regionArgs.erase(inItem);
        for (auto outItem : adaptor.getOutChans()) regionArgs.erase(outItem);

        if (!regionArgs.empty())
            emitWarning(
                op.getLoc(),
                "This loop does not watch all channels used by the operator. "
                "However, in the software lowering the closure of *any* "
                "channel in the executed code path will lead to the "
                "termination of the loop and therefore the operator. This may "
                "lead to behavior that differs from the Hardware lowering");

        // 1. get teardown block (for later reference)
        // maybe remove this assert? I'm depending on it for finding the
        // terminator block, though
        assert(parentRegion.getBlocks().size() == 1);
        Block &terminatorBlock = parentRegion.getBlocks().back();

        // 2. split after the loopOp -> finalizerBlock
        Block* currentBlock = op->getBlock();
        rewriter.setInsertionPointAfter(op);
        Block* finalizerBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        // 3. re-wire the cond_br in the loop to point to the finalizer
        SmallVector<cf::CondBranchOp> rewritableBranches;
        op->walk([&](cf::CondBranchOp branchOp) {
            if (branchOp.getFalseDest() == &terminatorBlock)
                rewritableBranches.push_back(branchOp);
        });

        for (auto rewriteOp : rewritableBranches) {
            rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
                rewriteOp,
                rewriteOp.getCondition(),
                rewriteOp.getTrueDest(),
                rewriteOp.getTrueOperands(),
                finalizerBlock,
                ArrayRef<Value>());
        }

        // 3.5 insert the back-jump at the end of the loop (while I still can
        // find it)
        Region &loopRegion = op.getRegion();
        Block* loopEntry = &loopRegion.front();
        rewriter.setInsertionPointToEnd(&loopRegion.back());
        cf::BranchOp::create(rewriter, op.getLoc(), &loopRegion.front());

        // 4. move the blocks from the region to after the LoopOp
        rewriter.inlineRegionBefore(loopRegion, finalizerBlock);

        // 5. insert a br before the loop
        rewriter.setInsertionPointToEnd(currentBlock);
        cf::BranchOp::create(rewriter, op.getLoc(), loopEntry);

        // 6. delete the current op
        rewriter.eraseOp(op);

        return success();
    }
};

void mlir::populateDfgToFuncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    const SpawnStrategy &spawnStrategy)
{
    // dfg.process -> func.func
    patterns.add<ProcessOpLowering>(typeConverter, patterns.getContext());

    // dfg.instantiate -> backend-specific invocation (e.g. func.call in an
    // omp.section)
    patterns.add<InstantiateOpLowering>(
        typeConverter,
        patterns.getContext(),
        &spawnStrategy);

    // dfg.loop -> cf
    patterns.add<LoopOpLowering>(typeConverter, patterns.getContext());
}

void ConvertDfgToFuncPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });

    std::unique_ptr<SpawnStrategy> spawnStrategy = createSpawnStrategy(backend);

    // wrap groups of successive instantiations in the parallel backend's
    // spawning construct (e.g. omp::ParallelOp + omp::SectionsOp, or
    // pthread_create/pthread_join calls)

    Operation* op = getOperation();

    SmallVector<SmallVector<InstantiateOp>> rewriteGroups;

    // whether an open group is being filled
    bool openGroup = false;
    WalkResult res = op->walk([&](InstantiateOp instantiation) -> WalkResult {
        if (!openGroup) {
            // create a new rewrite group
            SmallVector<InstantiateOp> group;
            rewriteGroups.push_back(group);
            openGroup = true;
        }

        SmallVector<InstantiateOp> &group = rewriteGroups.back();
        group.push_back(instantiation);

        if (!isa_and_nonnull<InstantiateOp>(instantiation->getNextNode())) {
            // the next node is NOT an instantiation, so break here
            openGroup = false;
        }

        return WalkResult::advance();
    });

    for (auto &group : rewriteGroups) {
        OpBuilder localRewriter(group[0]->getContext());
        localRewriter.setInsertionPoint(group[0]);
        spawnStrategy->emitParallelGroup(localRewriter, group[0]->getLoc(), group);
    }

    if (res.wasInterrupted()) signalPassFailure();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToFuncConversionPatterns(converter, patterns, *spawnStrategy);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        LLVM::LLVMDialect>();

    // the OpenMP dialect is only legal when the omp backend is used (it is
    // the backend that emits these ops)
    if (backend == ParallelBackend::Omp)
        target.addLegalDialect<omp::OpenMPDialect>();

    target.addLegalDialect<DfgDialect>();
    // NOTE(feliix42): Offloaded InstantiateOp operations are *not* covered by
    // this lowering but are illegal.
    target.addIllegalOp<ProcessOp, LoopOp, OperatorOp, InstantiateOp>();
    // target.addDynamicallyLegalOp<InstantiateOp>(
    //     [](InstantiateOp op) { return op.getOffloaded(); });

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

std::unique_ptr<Pass> mlir::createConvertDfgToFuncPass(ConvertDfgToFuncOptions options)
{
    return std::make_unique<ConvertDfgToFuncPass>(std::move(options));
}
