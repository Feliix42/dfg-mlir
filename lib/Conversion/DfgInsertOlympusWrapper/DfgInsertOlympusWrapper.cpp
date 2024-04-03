/// Implementation of the DfgInsertOlympusWrapper pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgInsertOlympusWrapper/DfgInsertOlympusWrapper.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGINSERTOLYMPUSWRAPPER
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

// ========================================================
// Constants
// ========================================================
std::string alveoHostWrapperName = "createAlveoHostObjectWrapper";
std::string alveoHostFunctionName = "createAlveoHost";
std::string alveoHostCall = "olympus_wrapper";

uint64_t dataWidth = 80;

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

// ========================================================
// Lowerings
// ========================================================

namespace {
struct ConvertDfgInsertOlympusWrapperPass
        : public mlir::impl::ConvertDfgInsertOlympusWrapperBase<
              ConvertDfgInsertOlympusWrapperPass> {
    void runOnOperation() final;
};
} // namespace

/// This lowering creates an olympus wrapper for each offloaded node
/// instantiation. It creates a new Operator which contains all relevant Olympus
/// calls
struct OffloadedInstantiateOpLowering
        : public OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    OffloadedInstantiateOpLowering(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context){};

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

        //
        ModuleOp parent = op->getParentOfType<ModuleOp>();

        // find associated OperatorOp
        OperatorOp kernelDefinition =
            parent.lookupSymbol<OperatorOp>(adaptor.getCallee());

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

        std::string pathAttrName("evp.path");
        Attribute pathAttr = kernelDefinition->getAttr(pathAttrName);
        if (!pathAttr) {
            emitError(
                kernelDefinition.getLoc(),
                "Kernel declaration must have an `evp.path` argument pointing "
                "to the kernel source file");
            return failure();
        }
        kernelOpState.addAttribute(pathAttrName, pathAttr);
        // add channels as arguments
        kernelOpState.addOperands(chans);

        rewriter.create(kernelOpState);

        // return
        rewriter.create<func::ReturnOp>(op.getLoc());

        // ====================================================================
        // Olympus wrapper call
        // ====================================================================
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            wrapperName,
            op->getResultTypes(),
            adaptor.getOperands());

        return success();
    }
};

LogicalResult createAlveoHostObject(SmallVector<InstantiateOp> &instantiations)
{
    if (instantiations.empty()) return success();

    // ensure that we don't do this for nothing
    size_t numInstantiations = instantiations.size();

    // get module
    ModuleOp module = instantiations[0]->getParentOfType<ModuleOp>();
    MLIRContext* moduleCtx = module->getContext();
    // ConversionPatternRewriter rewriter(moduleCtx);
    OpBuilder rewriter(moduleCtx);
    rewriter.setInsertionPointToStart(module.getBody());

    Type alveoHostObjectType = LLVM::LLVMPointerType::get(module->getContext());

    // ensure that the host object has not been created yet (which should be
    // impossible)
    OperatorOp opExists = module.lookupSymbol<OperatorOp>(alveoHostWrapperName);
    if (opExists) {
        emitError(
            opExists.getLoc(),
            "A wrapper for creating an Alveo Host object already exists.");
        return failure();
    }

    // create the OperatorOp
    std::vector<Type> tyVec;
    std::vector<Location> locs;
    for (size_t i = 0; i < numInstantiations; i++) {
        tyVec.push_back(InputType::get(moduleCtx, alveoHostObjectType));
        locs.push_back(module.getLoc());
    }

    FunctionType opSignature = FunctionType::get(moduleCtx, {}, tyVec);
    OperatorOp hostCreationOp = rewriter.create<OperatorOp>(
        instantiations[0].getLoc(),
        alveoHostWrapperName,
        opSignature);

    // create the body: generate the item, send it
    Region &opBody = hostCreationOp.getRegion();

    FlatSymbolRefAttr hostCreationRef = getOrInsertFunc(
        rewriter,
        module,
        alveoHostFunctionName,
        alveoHostObjectType,
        {});
    assert(numInstantiations != 0);
    Block* entryBlock = rewriter.createBlock(&opBody);
    entryBlock->addArguments(tyVec, locs);
    rewriter.setInsertionPointToEnd(entryBlock);
    func::CallOp hostObject = rewriter.create<func::CallOp>(
        hostCreationOp->getLoc(),
        hostCreationRef,
        ArrayRef<Type>(alveoHostObjectType),
        ArrayRef<Value>());

    for (Value output : opBody.getArguments()) {
        if (!isa<InputType>(output.getType())) {
            emitError(
                output.getLoc(),
                "A non InputType type in the arguments list. How did THAT "
                "happen");
            return failure();
        }

        rewriter.create<PushOp>(
            output.getLoc(),
            hostObject.getResult(0),
            output);
    }

    return success();
}

OperatorOp insertOlympusWrapperOp(InstantiateOp instantiation)
{
    ModuleOp module = instantiation->getParentOfType<ModuleOp>();
    // ConversionPatternRewriter rewriter(module.getContext());
    OpBuilder rewriter(module.getContext());
    rewriter.setInsertionPointToStart(module.getBody());

    std::string wrapperOpName =
        instantiation.getCallee().getRootReference().str() + "_wrapper";
    Type alveoHostObjectType =
        LLVM::LLVMPointerType::get(instantiation.getContext());
    OutputType alveoHostOutput =
        OutputType::get(instantiation.getContext(), alveoHostObjectType);

    // verification & data extraction
    // find associated OperatorOp
    OperatorOp kernelDefinition =
        module.lookupSymbol<OperatorOp>(instantiation.getCallee());

    // verify that both argument lengths match
    size_t kernelArgLength =
        kernelDefinition.getFunctionType().getNumInputs()
        + kernelDefinition.getFunctionType().getNumResults();
    if (kernelArgLength != instantiation.getOperands().size()) {
        emitError(
            kernelDefinition.getLoc(),
            "The kernel declaration that corresponds to this instantiation "
            "does not have a matching argument list length");
        return OperatorOp();
    }

    ArrayRef<int64_t> multiplicities = kernelDefinition.getMultiplicity();

    // verify that the number of arguments matches the defined multiplicity
    if (instantiation.getOperands().size() != multiplicities.size()) {
        emitError(
            instantiation.getLoc(),
            "The multiplicity argument of the kernel definition does not "
            "match the number of operands supplied to the instantiation "
            "function");
        return OperatorOp();
    }

    // create the OperatorOp
    SmallVector<Type> inputs, outputs, allArgs;
    SmallVector<Location> locs;
    for (Value arg : instantiation.getInputs()) {
        inputs.push_back(arg.getType());
        allArgs.push_back(arg.getType());
        locs.push_back(arg.getLoc());
    }
    inputs.push_back(alveoHostOutput);
    allArgs.push_back(alveoHostOutput);
    locs.push_back(instantiation.getLoc());

    for (Value arg : instantiation.getOutputs()) {
        outputs.push_back(arg.getType());
        allArgs.push_back(arg.getType());
        locs.push_back(arg.getLoc());
    }

    // manually add the AlveoHost object
    FunctionType opSignature =
        FunctionType::get(instantiation.getContext(), inputs, outputs);

    OperatorOp wrapperOp = rewriter.create<OperatorOp>(
        instantiation.getLoc(),
        wrapperOpName,
        opSignature);

    Location loc = wrapperOp.getLoc();

    // create the body: generate the item, send it
    Region &opBody = wrapperOp.getRegion();
    Block* entryBlock = rewriter.createBlock(&opBody);
    entryBlock->addArguments(allArgs, locs);
    rewriter.setInsertionPointToEnd(entryBlock);

    // pull the host object
    Value alveoInput =
        entryBlock->getArgument(wrapperOp.getFunctionType().getNumInputs() - 1);
    PullOp hostObject = rewriter.create<PullOp>(loc, alveoInput);

    // TODO: Draw the rest of the fuckin owl
    // initialize the two buffers for sx/rx
    // -> alloca buffers
    SmallVector<Value> ioChans;
    for (size_t i = 0; i < multiplicities.size(); i++) {
        // TODO: #iterations will go here!!
        int64_t m = multiplicities[i] * dataWidth;
        LLVM::ConstantOp bufSize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(m));

        Type elementType;
        if (i < instantiation.getInputs().size()) {
            // input
            elementType =
                llvm::cast<OutputType>(instantiation.getOperand(i).getType())
                    .getElementType();
        } else {
            // output
            elementType =
                llvm::cast<InputType>(instantiation.getOperand(i).getType())
                    .getElementType();
        }

        LLVM::LLVMPointerType ptrType =
            LLVM::LLVMPointerType::get(elementType.getContext());
        LLVM::AllocaOp allocated =
            rewriter.create<LLVM::AllocaOp>(loc, ptrType, ValueRange{bufSize});

        ioChans.push_back(allocated);
    }

    SmallVector<Value> inVals;
    SmallVector<Value> outVals;
    for (size_t i = 0; i < entryBlock->getNumArguments(); i++)
        if (i < inputs.size())
            inVals.push_back(entryBlock->getArgument(i));
        else
            outVals.push_back(entryBlock->getArgument(i));

    // loopOp
    LoopOp loopOp = rewriter.create<LoopOp>(loc, inVals, outVals);
    Block* loopEntryBlock = rewriter.createBlock(&loopOp.getBody());
    rewriter.setInsertionPointToStart(loopEntryBlock);

    Type llvmChannelPointer = LLVM::LLVMPointerType::get(module->getContext());

    //   accumulate items
    for (size_t i = 0; i < instantiation.getInputs().size(); i++) {
        int64_t m = multiplicities[i] * dataWidth;
        LLVM::ConstantOp numData = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(m));
        LLVM::BitcastOp casted = rewriter.create<LLVM::BitcastOp>(
            loc,
            LLVM::LLVMPointerType::get(instantiation.getContext()),
            ioChans[i]);
        UnrealizedConversionCastOp inputType =
            rewriter.create<UnrealizedConversionCastOp>(
                loc,
                llvmChannelPointer,
                entryBlock->getArgument(i));

        SmallVector<Value> callValues;
        callValues.push_back(inputType.getResult(0));
        callValues.push_back(casted);
        callValues.push_back(numData);
        FlatSymbolRefAttr pullNFunc = getOrInsertFunc(
            rewriter,
            module,
            "pull_n",
            rewriter.getI1Type(),
            callValues);

        func::CallOp pulledResult = rewriter.create<func::CallOp>(
            loc,
            ArrayRef<Type>(rewriter.getI1Type()),
            pullNFunc,
            callValues);
        // TODO: Figure out what to do with return value
    }

    ioChans.insert(ioChans.begin(), hostObject.getOutp());

    //   call host
    FlatSymbolRefAttr alveoHostFunc =
        getOrInsertFunc(rewriter, module, alveoHostCall, std::nullopt, ioChans);
    rewriter
        .create<func::CallOp>(loc, ArrayRef<Type>(), alveoHostFunc, ioChans);

    //   retrieve results
    for (size_t i = 0; i < instantiation.getOutputs().size(); i++) {
        size_t j = instantiation.getInputs().size() + i;
        int64_t m = multiplicities[j] * dataWidth;

        LLVM::ConstantOp numData = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(m));
        LLVM::BitcastOp casted = rewriter.create<LLVM::BitcastOp>(
            loc,
            LLVM::LLVMPointerType::get(instantiation.getContext()),
            ioChans[j + 1]);
        UnrealizedConversionCastOp inputType =
            rewriter.create<UnrealizedConversionCastOp>(
                loc,
                llvmChannelPointer,
                entryBlock->getArgument(j + 1));

        SmallVector<Value> callValues;
        callValues.push_back(inputType.getResult(0));
        callValues.push_back(casted);
        callValues.push_back(numData);
        FlatSymbolRefAttr pushNFunc = getOrInsertFunc(
            rewriter,
            module,
            "push_n",
            rewriter.getI1Type(),
            callValues);

        func::CallOp pushedResult = rewriter.create<func::CallOp>(
            loc,
            ArrayRef<Type>(rewriter.getI1Type()),
            pushNFunc,
            callValues);
        // TODO: Figure out what to do with return value
    }

    return wrapperOp;
}

SmallVector<ChannelOp>
createAlveoHostChannels(func::FuncOp topLevel, size_t numChannels)
{
    // ConversionPatternRewriter rewriter(topLevel.getContext());
    OpBuilder rewriter(topLevel.getContext());
    rewriter.setInsertionPointToStart(&topLevel.getBody().front());

    Type alveoHostObjectType =
        LLVM::LLVMPointerType::get(topLevel.getContext());

    SmallVector<ChannelOp> newChans;
    for (size_t i = 0; i < numChannels; i++) {
        ChannelOp chan = rewriter.create<ChannelOp>(
            topLevel.getLoc(),
            InputType::get(topLevel.getContext(), alveoHostObjectType),
            OutputType::get(topLevel.getContext(), alveoHostObjectType),
            TypeAttr::get(alveoHostObjectType),
            /* channel size */ nullptr);

        newChans.push_back(chan);
    }

    return newChans;
}

LogicalResult replaceInstantiations(
    SmallVector<InstantiateOp> &instantiations,
    SmallVector<OperatorOp> &newOps,
    SmallVector<ChannelOp> &newChans)
{
    ModuleOp module = instantiations[0]->getParentOfType<ModuleOp>();
    MLIRContext* moduleCtx = module->getContext();
    // ConversionPatternRewriter rewriter(moduleCtx);
    OpBuilder rewriter(moduleCtx);

    SmallVector<Value> alveoInputs, alveoOutputs;
    for (ChannelOp chanPair : newChans) {
        alveoInputs.push_back(chanPair.getResult(0));
        alveoOutputs.push_back(chanPair.getResult(1));
    }

    // insert the instantiation for the AlveoHostObject above the first
    // instantiation
    if (!module.lookupSymbol<OperatorOp>(alveoHostWrapperName)) {
        emitError(
            module.getLoc(),
            "The Alveo Host wrapper generation operator does not exist. This "
            "is an internal error.");
        return failure();
    }
    SymbolRefAttr alveoHostOpRef =
        SymbolRefAttr::get(moduleCtx, alveoHostWrapperName);

    rewriter.setInsertionPoint(instantiations[0]);
    rewriter.create<InstantiateOp>(
        instantiations[0].getLoc(),
        alveoHostOpRef,
        ValueRange(),
        alveoInputs);

    // replace all existing instantiations from the list with the wrapper
    // instantiations that also contain the AlveoHostObject
    for (size_t i = 0; i < newOps.size(); i++) {
        SmallVector<Value> ins = instantiations[i].getInputs();
        ins.push_back(alveoOutputs[i]);

        SymbolRefAttr instantiateName =
            SymbolRefAttr::get(newOps[i].getContext(), newOps[i].getSymName());
        rewriter.setInsertionPoint(instantiations[i]);
        rewriter.create<InstantiateOp>(
            instantiations[i].getLoc(),
            instantiateName,
            ins,
            instantiations[i].getOutputs());

        instantiations[i].erase();
    }

    return success();
}

void mlir::populateDfgInsertOlympusWrapperConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // dfg.instantiate offloaded -> dfg.instantiate & dfg.operator
    patterns.add<OffloadedInstantiateOpLowering>(
        typeConverter,
        patterns.getContext());
}

void ConvertDfgInsertOlympusWrapperPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });

    // TODO:
    // 1. collect all offloaded instantiates (in a list, ensure all of them are
    // in the same place (a func))
    // 2. insert creation of AlveoHost (needs # instantiates)
    // 3. for all instantiates, insert a new operator op.
    // 4. replace all instantiates with the new ones
    Operation* op = getOperation();
    SmallVector<InstantiateOp> instantiations;
    func::FuncOp topLevel;
    bool sawFuncWithInstantiates = false;

    // Step 1: Collect all offloaded instantiations
    WalkResult res = op->walk([&](func::FuncOp funcOp) -> WalkResult {
        WalkResult intRes = funcOp->walk([&](InstantiateOp instantiation) {
            // skip non-offloaded instantiates
            if (!instantiation.getOffloaded()) return WalkResult::skip();

            // FIXME(feliix42): Handling of offloaded instantiations in
            // different FuncOps. How should it be handled? two Host objects?
            // Generally forbidden?
            if (!sawFuncWithInstantiates) {
                sawFuncWithInstantiates = true;
            } else {
                emitError(
                    instantiation->getLoc(),
                    "Kernel Instantiations are split between multiple FuncOps. "
                    "While this is allowed behavior, it is not implemented "
                    "yet.");
                return WalkResult::interrupt();
            }

            instantiations.push_back(instantiation);

            return WalkResult::advance();
        });

        if (intRes.wasInterrupted())
            return WalkResult::interrupt();
        else
            return WalkResult::advance();
    });

    if (res.wasInterrupted()) signalPassFailure();

    if (instantiations.size() > 0) {
        topLevel = instantiations[0]->getParentOfType<func::FuncOp>();

        // 2. Insert the creation of the AlveoHost object
        if (failed(createAlveoHostObject(instantiations))) signalPassFailure();

        // 3. Insert an OperatorOp for each offloaded instantiate
        // TODO -> re-use the olympus lowering
        SmallVector<OperatorOp> newOps;
        for (InstantiateOp instantiation : instantiations)
            newOps.push_back(insertOlympusWrapperOp(instantiation));
        assert(newOps.size() == instantiations.size());

        // 4. Insert new channels (one per offloaded instantiate)
        SmallVector<ChannelOp> newChans =
            createAlveoHostChannels(topLevel, newOps.size());
        assert(newOps.size() == newChans.size());

        // 5. Replace the existing instantiations & add AlveoInstantiate
        if (failed(replaceInstantiations(instantiations, newOps, newChans)))
            signalPassFailure();
        // NOTE(feliix42): Do we want to return the op name from 2 and use here?
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // populateDfgInsertOlympusWrapperConversionPatterns(converter, patterns);

    target.addLegalDialect<BuiltinDialect, func::FuncDialect>();

    target.addLegalDialect<DfgDialect>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return !op.getOffloaded(); });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgInsertOlympusWrapperPass()
{
    return std::make_unique<ConvertDfgInsertOlympusWrapperPass>();
}
