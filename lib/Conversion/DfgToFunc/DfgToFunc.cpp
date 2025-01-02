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
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
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

struct ProcessOpLowering : public OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ProcessOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context) {};

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
            rewriter.create<func::FuncOp>(loc, operatorName, newFuncTy);

        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                adaptor.getBody(),
                genFuncOp.getBody(),
                genFuncOp.end());

            // populate the Terminator Block (for now) with a return only
            rewriter.setInsertionPointToEnd(&genFuncOp.getRegion().back());
            rewriter.create<func::ReturnOp>(op.getLoc());
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

    InstantiateOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
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

        omp::SectionOp sec = rewriter.create<omp::SectionOp>(op.getLoc());
        Region &sectionRegion = sec.getRegion();
        Block* secBlock = rewriter.createBlock(&sectionRegion);

        rewriter.setInsertionPointToStart(secBlock);
        rewriter.create<func::CallOp>(
            op.getLoc(),
            adaptor.getCallee(),
            ArrayRef<Type>(),
            adaptor.getOperands());
        rewriter.create<omp::TerminatorOp>(op.getLoc());

        rewriter.eraseOp(op);

        return success();
    }
};

struct LoopOpLowering : public OpConversionPattern<LoopOp> {
    using OpConversionPattern<LoopOp>::OpConversionPattern;

    LoopOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<LoopOp>(typeConverter, context) {};

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
        rewriter.create<cf::BranchOp>(op.getLoc(), &loopRegion.front());

        // 4. move the blocks from the region to after the LoopOp
        rewriter.inlineRegionBefore(loopRegion, finalizerBlock);

        // 5. insert a br before the loop
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<cf::BranchOp>(op.getLoc(), loopEntry);

        // 6. delete the current op
        rewriter.eraseOp(op);

        return success();
    }
};

void mlir::populateDfgToFuncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.process -> func.func
    patterns.add<ProcessOpLowering>(typeConverter, patterns.getContext());

    // dfg.instantiate -> func.call
    patterns.add<InstantiateOpLowering>(typeConverter, patterns.getContext());

    // dfg.loop -> cf
    patterns.add<LoopOpLowering>(typeConverter, patterns.getContext());
}

void ConvertDfgToFuncPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });

    // insert omp::SectionsOp and omp::ParallelOp in the code
    // do so by counting the number of successive instantiations and wrapping
    // them

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

        if (!isa<InstantiateOp>(instantiation->getNextNode())) {
            // the next node is NOT an instantiation, so break here
            openGroup = false;
        }

        return WalkResult::advance();
    });

    for (auto group : rewriteGroups) {
        size_t groupSize = group.size();

        Location loc = group[0]->getLoc();

        // ConversionPatternRewriter localRewriter(group[0]->getContext());
        OpBuilder localRewriter(group[0]->getContext());
        localRewriter.setInsertionPoint(group[0]);

        arith::ConstantOp threadCount = localRewriter.create<arith::ConstantOp>(
            loc,
            localRewriter.getI32Type(),
            localRewriter.getI32IntegerAttr(groupSize));

        omp::ParallelOp par = localRewriter.create<omp::ParallelOp>(
            loc,
            ValueRange(),
            ValueRange(),
            Value{},
            threadCount.getResult(),
            ValueRange(),
            ArrayAttr{},
            omp::ClauseProcBindKindAttr{},
            ValueRange(),
            DenseBoolArrayAttr{},
            nullptr);
        Block* parBlock = localRewriter.createBlock(&par.getRegion());

        omp::SectionsOp sections = localRewriter.create<omp::SectionsOp>(
            loc,
            ArrayRef<Type>(),
            ValueRange());
        sections.setNowait(true);
        Block* sectionsBlock = localRewriter.createBlock(&sections.getRegion());

        // move instantiates here
        for (auto op : group) {
            localRewriter.create<InstantiateOp>(
                op->getLoc(),
                op.getCallee(),
                op.getInputs(),
                op.getOutputs(),
                op.getOffloadedAttr());
            // localRewriter.eraseOp(op);
            op->erase();
        }

        localRewriter.setInsertionPointToEnd(sectionsBlock);
        localRewriter.create<omp::TerminatorOp>(loc);

        localRewriter.setInsertionPointToEnd(parBlock);
        localRewriter.create<omp::TerminatorOp>(loc);
    }

    if (res.wasInterrupted()) signalPassFailure();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToFuncConversionPatterns(converter, patterns);

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        omp::OpenMPDialect,
        LLVM::LLVMDialect>();

    target.addLegalDialect<DfgDialect>();
    // NOTE(feliix42): Offloaded InstantiateOp operations are *not* covered by
    // this lowering but are illegal.
    target.addIllegalOp<ProcessOp, LoopOp, InstantiateOp>();
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
