/// Implementation of the DfgInsertOlympusWrapper pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgInsertOlympusWrapper/DfgInsertOlympusWrapper.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

// ========================================================
// Helper Functions
// ========================================================

/// Return a symbol reference to the requested function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertFunc(
    PatternRewriter &rewriter,
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

/// This lowering transforms push_n ops into runtime channel code, compatible
/// with the Olympus memory layout.
struct OlympusPushNLowering
        : public OpConversionPattern<PushNOp> {
    using OpConversionPattern<PushNOp>::OpConversionPattern;

    OlympusPushNLowering(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<PushNOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushNOp op,
        PushNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO
        
        return success();
    }
};

/// This lowering transforms pull_n ops into runtime channel code, compatible
/// with the Olympus memory layout.
struct OlympusPullNLowering
        : public OpConversionPattern<PullNOp> {
    using OpConversionPattern<PullNOp>::OpConversionPattern;

    OlympusPullNLowering(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<PullNOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullNOp op,
        PullNOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO
        
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
    ConversionPatternRewriter rewriter(moduleCtx);
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
    ConversionPatternRewriter rewriter(module.getContext());
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

    SmallVector<Value> getNumTimesArgs;
    getNumTimesArgs.push_back(hostObject);

    FlatSymbolRefAttr getNumTimesFunc = getOrInsertFunc(
        rewriter,
        module,
        "get_num_times",
        rewriter.getI64Type(),
        getNumTimesArgs);

    func::CallOp dataWidth = rewriter.create<func::CallOp>(
        loc,
        ArrayRef<Type>(rewriter.getI64Type()),
        getNumTimesFunc,
        getNumTimesArgs);

    // TODO: Draw the rest of the fuckin owl
    // initialize the two buffers for sx/rx
    // -> alloca buffers
    SmallVector<Value> ioChans;
    SmallVector<Value> inputDepth;
    for (size_t i = 0; i < instantiation.getInputs().size(); i++) {
        // TODO: #iterations will go here!!
        arith::ConstantOp multI = rewriter.create<arith::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(multiplicities[i]));
        arith::MulIOp bufSize = rewriter.create<arith::MulIOp>(
            loc,
            dataWidth.getResults()[0],
            multI);
        inputDepth.push_back(bufSize);

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

        LLVM::LLVMPointerType ptrType = LLVM::LLVMPointerType::get(elementType);
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
        callValues.push_back(inputDepth[i]);
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

    SmallVector<Value> alveoInputs(ioChans.begin(), ioChans.begin() + 1 + instantiation.getInputs().size());

    //   call host
    FlatSymbolRefAttr alveoHostFunc =
        getOrInsertFunc(rewriter, module, alveoHostCall, std::nullopt, alveoInputs);
    rewriter
        .create<func::CallOp>(loc, ArrayRef<Type>(), alveoHostFunc, alveoInputs);

    //   retrieve results
    for (size_t i = 0; i < instantiation.getOutputs().size(); i++) {
        size_t j = instantiation.getInputs().size() + i;

        arith::ConstantOp multJ = rewriter.create<arith::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(multiplicities[j]));
        arith::MulIOp numData = rewriter.create<arith::MulIOp>(
            loc,
            dataWidth.getResults()[0],
            multJ);
        LLVM::BitcastOp casted = rewriter.create<LLVM::BitcastOp>(
            loc,
            LLVM::LLVMPointerType::get(instantiation.getContext()),
            ioChans[j+1 - instantiation.getOutputs().size()]);
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
    ConversionPatternRewriter rewriter(topLevel.getContext());
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
    ConversionPatternRewriter rewriter(moduleCtx);

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
    patterns.add<OlympusPushNLowering>(
        typeConverter,
        patterns.getContext());

    patterns.add<OlympusPullNLowering>(
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

    SmallVector<OperatorOp> newOps;
    if (instantiations.size() > 0) {
        topLevel = instantiations[0]->getParentOfType<func::FuncOp>();

        // 2. Insert the creation of the AlveoHost object
        if (failed(createAlveoHostObject(instantiations))) signalPassFailure();

        // 3. Insert an OperatorOp for each offloaded instantiate
        // TODO -> re-use the olympus lowering
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

    populateDfgInsertOlympusWrapperConversionPatterns(converter, patterns);

    target.addLegalDialect<BuiltinDialect, func::FuncDialect>();

    target.addLegalDialect<DfgDialect>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return !op.getOffloaded(); });
    target.addDynamicallyLegalOp<PushNOp>(
        [&](PushNOp op) {
            // NOTE(feliix42): This might fail for items with parent != OperatorOp. Is that legal?
            OperatorOp parent = op->getParentOfType<OperatorOp>();
            bool legal = true;
            for (OperatorOp newOp : newOps) {
                legal = legal && (newOp != parent);
            }

            return legal;
        });
    target.addDynamicallyLegalOp<PullNOp>(
        [&](PullNOp op) {
            // NOTE(feliix42): This might fail for items with parent != OperatorOp. Is that legal?
            OperatorOp parent = op->getParentOfType<OperatorOp>();
            bool legal = true;
            for (OperatorOp newOp : newOps) {
                legal = legal && (newOp != parent);
            }

            return legal;
        });

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
