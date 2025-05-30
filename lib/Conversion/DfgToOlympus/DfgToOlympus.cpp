/// Implementation of the DfgToOlympus pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToOlympus/DfgToOlympus.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOOLYMPUS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

// ========================================================
// Utility functions
// ========================================================

static std::string OLYMPUS_CONFIG = "olympus.config";

SmallVector<NamedAttribute>
deriveAttributesFromConfig(ConversionPatternRewriter &rewriter, StringRef cfg)
{
    SmallVector<StringRef> splitValues;
    SmallVector<NamedAttribute> attributes;

    cfg.split(splitValues, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

    for (auto pair : splitValues) {
        if (pair.contains('=')) {
            auto [name, val] = pair.split('=');
            NamedAttribute a(
                rewriter.getStringAttr(name),
                rewriter.getStringAttr(val));
            attributes.push_back(a);
        } else {
            NamedAttribute a(
                rewriter.getStringAttr(pair),
                rewriter.getBoolAttr(true));
            attributes.push_back(a);
        }
    }

    return attributes;
}

// ========================================================
// Lowerings
// ========================================================

namespace {
struct ConvertDfgToOlympusPass
        : public mlir::impl::ConvertDfgToOlympusBase<ConvertDfgToOlympusPass> {
    void runOnOperation() final;
};
} // namespace

/// This lowering creates an olympus wrapper for each offloaded node
/// instantiation. It creates a new FuncOp which contains all relevant Olympus
/// operations
struct OffloadedInstantiateOpLowering
        : public OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    OffloadedInstantiateOpLowering(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // don't lower offloaded functions
        if (!adaptor.getOffloaded()) {
            emitError(
                op.getLoc(),
                "This lowering is supposed to run on offloaded instantiations "
                "only");
            return failure();
        }

        ModuleOp parent = op->getParentOfType<ModuleOp>();

        // find associated ProcessOp
        ProcessOp kernelDefinition =
            parent.lookupSymbol<ProcessOp>(adaptor.getCallee());

        if (kernelDefinition == NULL)
            return ::emitError(
                parent.lookupSymbol(adaptor.getCallee())->getLoc(),
                "Expected this to be a ProcessOp");

        // verify that both argument lengths match
        size_t kernelArgLength =
            kernelDefinition.getFunctionType().getNumInputs()
            + kernelDefinition.getFunctionType().getNumResults();
        if (kernelArgLength != op.getOperands().size()) {
            emitError(
                kernelDefinition.getLoc(),
                "The kernel declaration that corresponds to this instantiation "
                "does not have a matching argument list length");
            return failure();
        }

        // ====================================================================
        // Olympus wrapper insertion
        // ====================================================================
        SymbolRefAttr kernel_name = adaptor.getCallee();
        std::string wrapperName = kernel_name.getRootReference().str();
        wrapperName.append("_wrapper");

        // construct the function type for the wrapper
        auto fnType = rewriter.getFunctionType(op.getOperands().getTypes(), {});

        // TODO(feliix42): Fix the name generation to not error on duplicate
        // names
        if (parent.lookupSymbol<func::FuncOp>(wrapperName)) {
            emitError(
                op.getLoc(),
                "Wrapper function name already exists. This is an issue that "
                "needs to be addressed in the lowering code.");
            return failure();
        }

        rewriter.setInsertionPointToStart(parent.getBody());
        func::FuncOp olympusWrapper =
            rewriter.create<func::FuncOp>(op.getLoc(), wrapperName, fnType);
        Block* entryBlock = olympusWrapper.addEntryBlock();
        rewriter.setInsertionPointToEnd(entryBlock);

        // insert olympus.channels
        StringAttr olympusDialectName =
            StringAttr::get(op->getContext(), "olympus");
        llvm::SmallVector<Value> chans;
        int i = 0;
        ArrayRef<int64_t> multiplicities = kernelDefinition.getMultiplicity();

        if (multiplicities.empty()) {
            // initialize with all 1's
            SmallVector<int64_t> newMults;
            for (size_t i = 0; i < op.getOperands().size(); i++)
                newMults.push_back(1);

            multiplicities = newMults;
        }
        // verify that the number of arguments matches the defined multiplicity
        if (op.getOperands().size() != multiplicities.size()) {
            emitError(
                op.getLoc(),
                "The multiplicity argument of the kernel definition does not "
                "match the number of operands supplied to the instantiation "
                "function");
            return failure();
        }

        for (auto arg : op.getOperands()) {
            OperationState chanOpState(op.getLoc(), "olympus.channel");
            chanOpState.addAttribute(
                "depth",
                rewriter.getI64IntegerAttr(multiplicities[i]));
            chanOpState.addAttribute(
                "paramType",
                rewriter.getStringAttr("small"));

            std::string chanTy = "channel<";
            llvm::raw_string_ostream tyStream(chanTy);
            if (isa<OutputType>(arg.getType()))
                tyStream << cast<OutputType>(arg.getType()).getElementType()
                         << ">";
            if (isa<InputType>(arg.getType()))
                tyStream << cast<InputType>(arg.getType()).getElementType()
                         << ">";
            OpaqueType channelType =
                OpaqueType::get(olympusDialectName, chanTy);
            chanOpState.addTypes({channelType});

            Operation* chan = rewriter.create(chanOpState);
            chans.push_back(chan->getResult(0));
            i++;
        }

        // create the olympus kernel op, supply it with all generated channels
        // as inputs
        OperationState kernelOpState(op.getLoc(), "olympus.kernel");
        kernelOpState.addAttribute(
            "callee",
            adaptor.getCallee().getRootReference());
        kernelOpState.addAttribute(
            op.getOperandSegmentSizeAttr(),
            op->getAttr(op.getOperandSegmentSizesAttrName()));

        // propagate all olympus configuration attributes
        if (op->hasAttrOfType<StringAttr>(OLYMPUS_CONFIG)) {
            StringAttr olympusConfig =
                op->getAttrOfType<StringAttr>(OLYMPUS_CONFIG);
            SmallVector<NamedAttribute> cfgAttrs =
                deriveAttributesFromConfig(rewriter, olympusConfig.getValue());
            kernelOpState.addAttributes(cfgAttrs);
        } else {
            if (op->hasAttr(OLYMPUS_CONFIG))
                ::emitWarning(
                    op.getLoc(),
                    "The Olympus configuration attribute was found but ignored "
                    "as it was not a string attribute");
        }

        std::string pathAttrName("path");
        if (kernelOpState.attributes.get("path") == NULL) {
            Attribute pathAttr = kernelDefinition->getAttr(pathAttrName);
            if (!pathAttr) {
                return ::emitError(
                    kernelDefinition.getLoc(),
                    "Kernel declaration must have an `path` argument pointing "
                    "to the kernel source file");
            }
            kernelOpState.addAttribute(pathAttrName, pathAttr);
        }

        // add channels as arguments
        kernelOpState.addOperands(chans);

        rewriter.create(kernelOpState);

        // return
        rewriter.create<func::ReturnOp>(op.getLoc());

        // ====================================================================
        // Olympus wrapper call
        // ====================================================================
        // Technically, we don't care about any dfg past this point, but this
        // seems safe(r)
        op.setOffloaded(false);
        // rewriter.setInsertionPoint(op);
        // rewriter.replaceOpWithNewOp<func::CallOp>(
        //     op,
        //     wrapperName,
        //     op->getResultTypes(),
        //     adaptor.getOperands());

        return success();
    }
};

void mlir::populateDfgToOlympusConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.instantiate -> func.call & olympus
    patterns.add<OffloadedInstantiateOpLowering>(
        typeConverter,
        patterns.getContext());
}

void ConvertDfgToOlympusPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToOlympusConversionPatterns(converter, patterns);

    target.addLegalDialect<BuiltinDialect, func::FuncDialect>();

    target.addLegalDialect<DfgDialect>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return !op.getOffloaded(); });
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
        return (op->getName().getDialectNamespace() == "olympus")
               || converter.isLegal(op->getOperandTypes());
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToOlympusPass()
{
    return std::make_unique<ConvertDfgToOlympusPass>();
}
