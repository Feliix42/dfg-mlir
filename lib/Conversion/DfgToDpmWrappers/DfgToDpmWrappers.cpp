/// Implementation of the DfgToDpmWrappers pass.
///
/// @file
/// @author     Fabius Mayer-Uhma (fabius.mayer-uhma@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToDpmWrappers/DfgToDpmWrappers.h"

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
#include "mlir/Dialect/EmitC/IR/EmitC.h"

#include <regex>
#include <functional>

#include "dfg-mlir/Conversion/Utils.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTODPMWRAPPERS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

// ========================================================
// Lowerings
// ========================================================

// IMPORTANT: THIS HAS TO BE IN AN UNIQUE NAMESPACE!! IF NOT YOUR CONVERSION
// PATTERNS MIGHT BE NAMED LIKE AN ALREADY EXISTING ONE WHICH LEADS TO
// UNTRACABLE ERRORS
// TODO think about how to actually generate unique adapter names

namespace {

struct ConvertDfgToDpmWrappersPass : public mlir::impl::ConvertDfgToDpmWrappersBase<
                                      ConvertDfgToDpmWrappersPass> {
    void runOnOperation() final;
};

std::vector<std::tuple<std::string, std::vector<Type>, std::vector<Type>, SymbolRefAttr, emitc::PointerType, std::string>> collectedInstantiateOps;
std::vector<std::tuple<std::string, std::vector<Type>, std::vector<Type>, SymbolRefAttr, emitc::PointerType>> collectedEmbedOps;
std::vector<std::tuple<std::string, Type, emitc::PointerType>> collectedChannelOps;
std::vector<std::tuple<std::string, FunctionType>> collectedProcessOps;
std::vector<std::tuple<std::string, FunctionType>> collectedOperatorOps;
std::vector<std::tuple<std::string, FunctionType>> collectedRegionOps;
std::vector<std::tuple<std::string, Type>> collectedPullOps;
std::vector<std::tuple<std::string, Type>> collectedPushOps;

// helper functions

// TODO is there a better way to access row and col?
std::string GetLocationString(Location loc){
    std::string locationName;
    auto ostream = llvm::raw_string_ostream(locationName);
    loc.print(ostream);
    std::regex pattern(R"((\d+):(\d+)\)$)");
    std::smatch matches;
    if (std::regex_search(locationName, matches, pattern)){
        if(matches.size() == 3){
            return matches[1].str() + "_" + matches[2].str();
        }
    }
    return "ERROR_ERROR";
}

std::string GetAdapterName(Operation *operation){
    std::string opName = operation->getName().getStringRef().str();
    std::replace(opName.begin(),opName.end(),'.','_');
    std::string adapterName = opName + "_at_" + GetLocationString(operation->getLoc()) + "_dpm_adapter";
    return adapterName;
}

std::vector<Type> lowerAllTypes(std::vector<Type> inputTypes, const TypeConverter *typeConverter){
    std::vector<Type> returnValue;
    for(Type type : inputTypes){
        returnValue.push_back(typeConverter->convertType(type));
    }
    return returnValue;
}

emitc::OpaqueType wrapAllInTemplate(MLIRContext *context, std::string typeName, std::vector<Type> types){
    std::string returnTypeName = typeName + "<";
    for(size_t i = 0 ; i < types.size(); i++){
        auto type = types[i];
        emitc::OpaqueType castedOp = llvm::cast<emitc::OpaqueType>(type);
        if(i > 0){
            returnTypeName += ",";
        }
        returnTypeName += castedOp.getValue().str();
    }
    returnTypeName += ">";
    return emitc::OpaqueType::get(context, returnTypeName);
}

emitc::PointerType getRegionType(MLIRContext *context, emitc::OpaqueType inputChannelsTupleType, emitc::OpaqueType outputChannelsTupleType){
    auto regionType = emitc::OpaqueType::get(context, ("Dppm::Region<" + inputChannelsTupleType.getValue() + "," + outputChannelsTupleType.getValue() + ">").str());
    return emitc::PointerType::get(context, regionType);
}

emitc::PointerType extractTypeFromRegionOp(RegionOp regionOp, const TypeConverter *typeConverter){
    auto functionType = regionOp.getFunctionType();
    emitc::OpaqueType inputChannelsTupleType = wrapAllInTemplate(regionOp.getContext(), "Dppm::InputChannels", lowerAllTypes(functionType.getInputs(), typeConverter));
    emitc::OpaqueType outputChannelsTupleType = wrapAllInTemplate(regionOp.getContext(), "Dppm::OutputChannels", lowerAllTypes(functionType.getResults(), typeConverter));
    return getRegionType(regionOp.getContext(), inputChannelsTupleType, outputChannelsTupleType);
}

emitc::OpaqueType wrapInTemplate(MLIRContext *context, std::string templateName, Type inputType){
    emitc::OpaqueType castedOp = llvm::cast<emitc::OpaqueType>(inputType);
    return emitc::OpaqueType::get(context, (templateName + "<" + castedOp.getValue() + ">").str());
}

emitc::PointerType getChannelPointer(MLIRContext *context, Type elementType){
    return emitc::PointerType::get(wrapInTemplate(context, "Dppm::Channel", elementType));
}

emitc::PointerType getRTChannelPointer(MLIRContext *context, Type elementType){
    return emitc::PointerType::get(wrapInTemplate(context, "Dppm::RTChannel", elementType));
}

class ContextBoundHelper {
public:
    ContextBoundHelper(MLIRContext *_context, ModuleOp *_globalModuleOp) : context(_context), globalModuleOp(_globalModuleOp) {}
    emitc::OpaqueType CreateStdArrayWithSameSize(Type type){
        auto dataLayout = DataLayout(*globalModuleOp);
        auto dataSize = dataLayout.getTypeSize(type);
        std::string typeName = "std::array<uint8_t," + std::to_string(dataSize) + ">";
        auto channelType = emitc::OpaqueType::get(context, typeName);
        return channelType;
    }
    emitc::OpaqueAttr OpaqueAttr(std::string name){
        return emitc::OpaqueAttr::get(context, name);
    }
    emitc::PointerType PointerType(Type t){
        return emitc::PointerType::get(context, t);
    }
    emitc::PointerType getRTChannelPointer(Type elementType){
        return emitc::PointerType::get(wrapInTemplate(context, "Dppm::RTChannel", elementType));
    }

private:
    MLIRContext* context;
    const ModuleOp* globalModuleOp;
};


struct EmbedOpLowering : public mlir::OpConversionPattern<EmbedOp> {
    using OpConversionPattern<EmbedOp>::OpConversionPattern;

    EmbedOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<EmbedOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        EmbedOp embedOp,
        EmbedOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // do nothing but update operator
        rewriter.modifyOpInPlace(embedOp, [](){});

        Operation* parentOp = embedOp->getParentOp();
        RegionOp parentRegionOp = llvm::cast<RegionOp>(parentOp);

        std::vector<Type> inputTypes;
        std::vector<Type> outputTypes;
        for(auto inputValue : adaptor.getInputs()){
            inputTypes.push_back(inputValue.getType());
        }
        for(auto outputValue : adaptor.getOutputs()){
            outputTypes.push_back(outputValue.getType());
        }
        auto regionType = extractTypeFromRegionOp(parentRegionOp, typeConverter);
        collectedEmbedOps.push_back({GetAdapterName(embedOp), inputTypes, outputTypes, embedOp.getCallee(), regionType});
        return success();
    }
};

struct PushOpLowering : public mlir::OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        PushOp pushOp,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // do nothing but update operator
        rewriter.modifyOpInPlace(pushOp, [](){});
        collectedPushOps.push_back({GetAdapterName(pushOp), adaptor.getChan().getType()});
        return success();
    }
};


struct PullOpLowering : public mlir::OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    PullOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        PullOp pullOp,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.modifyOpInPlace(pullOp, [](){});
        collectedPullOps.push_back({GetAdapterName(pullOp), adaptor.getChan().getType()});
        return success();
    }
};

struct ChannelOpLowering : public mlir::OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ChannelOp channelOp,
        ChannelOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        std::string adapterName = GetAdapterName(channelOp);
        Operation* parentOp = channelOp->getParentOp();
        RegionOp parentRegionOp = llvm::cast<RegionOp>(parentOp);
        auto regionType = extractTypeFromRegionOp(parentRegionOp, typeConverter);
        rewriter.modifyOpInPlace(channelOp, [](){});
        collectedChannelOps.push_back({adapterName, typeConverter->convertType(channelOp.getInChan().getType()), regionType});
        return success();
    }
};


struct OperatorOpLowering : public mlir::OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    OperatorOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OperatorOp operatorOp,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // do nothing but update operator
        if(operatorOp.getIterArgs().size() > 0){
            llvm::errs() << "Cannot convert SDF with iterargs\n";
            return failure();
        }
        rewriter.modifyOpInPlace(operatorOp, [](){});
        collectedOperatorOps.push_back({operatorOp.getSymName().str(), operatorOp.getFunctionType()});
        return success();
    }
};


struct ProcessOpLowering : public mlir::OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ProcessOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ProcessOp processOp,
        ProcessOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        rewriter.modifyOpInPlace(processOp, [](){});
        std::string processName = processOp.getSymName().str();
        collectedProcessOps.push_back({processName, processOp.getFunctionType()});
        return success();
    }
};


struct RegionOpLowering : public mlir::OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    RegionOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        RegionOp regionOp,
        RegionOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        rewriter.modifyOpInPlace(regionOp, [](){});
        std::string processName = regionOp.getSymName().str();
        collectedRegionOps.push_back({processName, regionOp.getFunctionType()});
        return success();
    }
};



struct InstantiateOpLowering : public mlir::OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        InstantiateOp instantiateOp,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        std::string processName = SymbolTable::lookupSymbolIn(instantiateOp->getParentOfType<ModuleOp>(), instantiateOp.getCalleeAttr())->getName().getStringRef().str();
        Operation* parentOp = instantiateOp->getParentOp();
        RegionOp parentRegionOp = llvm::cast<RegionOp>(parentOp);
        auto regionType = extractTypeFromRegionOp(parentRegionOp, typeConverter);
        StringRef functionName = instantiateOp.getCallee().getRootReference();
        std::vector<Type> inputTypes;
        std::vector<Type> outputTypes;
        for(auto inputValue : adaptor.getInputs()){
            inputTypes.push_back(inputValue.getType());
        }
        for(auto outputValue : adaptor.getOutputs()){
            outputTypes.push_back(outputValue.getType());
        }
        collectedInstantiateOps.push_back({GetAdapterName(instantiateOp), inputTypes, outputTypes, instantiateOp.getCallee(), regionType, processName});
        rewriter.modifyOpInPlace(instantiateOp, [](){});
        return success();
    }
};


void ConvertDfgToDpmWrappersPass::runOnOperation() {

    Operation* op = getOperation();

    ModuleOp globalModuleOp;
    op->walk([&globalModuleOp](ModuleOp moduleOp){
        globalModuleOp = moduleOp;
        return WalkResult::interrupt();
    });

    TypeConverter highlevelTypeConverter;
    OpBuilder rewriter(&getContext());

    ContextBoundHelper helper(rewriter.getContext(), &globalModuleOp);

    /* Initialize types */
    auto autoType = emitc::OpaqueType::get(&getContext(), "auto");

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // to fit the conversion logic these would actually channel pointers but wrapping the element type is easier than unwrapping
    highlevelTypeConverter.addConversion([](Type t){return t;});
    highlevelTypeConverter.addConversion([&helper](InputType t){
        return helper.CreateStdArrayWithSameSize(t.getElementType());
    });
    highlevelTypeConverter.addConversion([&helper](OutputType t){
        return helper.CreateStdArrayWithSameSize(t.getElementType());
    });

    patterns.add<InstantiateOpLowering, ProcessOpLowering, ChannelOpLowering, PushOpLowering, PullOpLowering, EmbedOpLowering, RegionOpLowering, OperatorOpLowering>(highlevelTypeConverter, patterns.getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    //target.addIllegalDialect<DfgDialect>();

    // use PartialConversion because error logs are better
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();

    // erase everything and add back a main block (cant erase each operator individually because they are linked so delete them all at once)
    globalModuleOp.getBody()->erase();
    globalModuleOp.getRegion().emplaceBlock();

    auto loc = globalModuleOp.getLoc();

    rewriter.setInsertionPointToStart(globalModuleOp.getBody());

    rewriter.create<emitc::IncludeOp>(loc,"Dpm/Manager.h");
    rewriter.create<emitc::IncludeOp>(loc,"includes.h");


    for(auto [funcName, functionType] : collectedOperatorOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        auto outArrayTypes = map(functionType.getResults(), [&helper](Type t) { return (Type)helper.CreateStdArrayWithSameSize(t);});
        auto inArrayTypes = map(functionType.getInputs(), [&helper](Type t){ return (Type)helper.CreateStdArrayWithSameSize(t); });
        auto allArrayTypes = combine(inArrayTypes, outArrayTypes);
        emitc::OpaqueType outputTypesTuple = wrapAllInTemplate(rewriter.getContext(), "std::tuple", outArrayTypes);
        auto arrayPointerTypes = map(allArrayTypes, [&helper](auto t){ return (Type)helper.PointerType(t); });
        rewriter.create<emitc::VerbatimOp>(loc, "extern \"C\" {");
        auto originalFunc = rewriter.create<emitc::FuncOp>(loc, funcName+"_ORIGINAL_dpm_adapter", FunctionType::get(rewriter.getContext(), arrayPointerTypes, {}));
        rewriter.create<emitc::VerbatimOp>(loc, "} // extern C end");
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName + "_dpm_adapter", FunctionType::get(rewriter.getContext(), inArrayTypes, {outputTypesTuple}));
        auto entryBlock = functionHolder.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);
        auto tuples = map(outArrayTypes, [&helper, &loc, &rewriter](auto t){ return rewriter.create<emitc::ConstantOp>(loc, t, helper.OpaqueAttr("{}")).getResult(); });
        auto blockValues = map(entryBlock->getArguments(), [](auto e){ return (Value)e;});
        auto allValues = combine(blockValues, tuples);
        auto arrayPointers = map(allValues, [&loc, &rewriter, &helper](auto v){
            return rewriter.create<emitc::CallOpaqueOp>(loc,TypeRange{helper.PointerType(v.getType())},"&",ValueRange{v}).getResult(0);
        });
        rewriter.create<emitc::CallOp>(loc, originalFunc, arrayPointers);
        std::vector<Value> resultValues;
        for(size_t i = allValues.size() - tuples.size(); i < allValues.size() ; i++){
            Type t = allValues[i].getType();
            resultValues.push_back(rewriter.create<emitc::CallOpaqueOp>(loc,TypeRange{t},"*",ValueRange{arrayPointers[i]}).getResult(0));
        }
        auto returnTuple = rewriter.create<emitc::CallOpaqueOp>(loc, outputTypesTuple, "std::make_tuple", resultValues).getResult(0);
        rewriter.create<emitc::ReturnOp>(loc, returnTuple);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    for(auto [funcName, functionType] : collectedProcessOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        auto loweredInputTypes = lowerAllTypes(functionType.getInputs(), &highlevelTypeConverter);
        auto loweredOutputTypes = lowerAllTypes(functionType.getResults(), &highlevelTypeConverter);
        auto inputChannelsTupleType = wrapAllInTemplate(rewriter.getContext(), "Dppm::InputRTChannels", loweredInputTypes);
        auto outputChannelsTupleType = wrapAllInTemplate(rewriter.getContext(), "Dppm::OutputRTChannels", loweredOutputTypes);
        auto allLoweredTypes = combine(loweredInputTypes, loweredOutputTypes);
        auto declFuncTypes = map(allLoweredTypes, [&helper](auto t){ return (Type)helper.getRTChannelPointer(t); });
        rewriter.create<emitc::VerbatimOp>(loc, "extern \"C\" {");
        auto originalFunc = rewriter.create<emitc::FuncOp>(loc, funcName+"_ORIGINAL_dpm_adapter", FunctionType::get(rewriter.getContext(), declFuncTypes, {}));
        rewriter.create<emitc::VerbatimOp>(loc, "} // extern C end");
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName + "_dpm_adapter", FunctionType::get(&getContext(), {inputChannelsTupleType, outputChannelsTupleType}, {}));
        auto entryBlock = functionHolder.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);
        auto inputChannelsTuple = entryBlock->getArgument(0);
        auto outputChannelsTuple = entryBlock->getArgument(1);
        std::vector<Value> channelValues;
        // TODO if structured binding is added to emitc, replace this
        for(size_t i = 0 ; i < loweredInputTypes.size() ; i++){
            auto getValue = rewriter.create<emitc::CallOpaqueOp>(loc,TypeRange{getRTChannelPointer(rewriter.getContext(), loweredInputTypes[i])}, StringRef{"std::get"}, ValueRange{inputChannelsTuple}, ArrayAttr{}, rewriter.getArrayAttr(rewriter.getI64IntegerAttr(i)));
            channelValues.push_back(getValue.getResult(0));
        }
        for(size_t i = 0 ; i < loweredOutputTypes.size() ; i++){
            auto getValue = rewriter.create<emitc::CallOpaqueOp>(loc,TypeRange{getRTChannelPointer(rewriter.getContext(), loweredOutputTypes[i])}, StringRef{"std::get"}, ValueRange{outputChannelsTuple}, ArrayAttr{}, rewriter.getArrayAttr(rewriter.getI64IntegerAttr(i)));
            channelValues.push_back(getValue.getResult(0));
        }
        rewriter.create<emitc::CallOp>(loc, originalFunc, channelValues);
        rewriter.create<emitc::ReturnOp>(loc, (Value)0);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    rewriter.create<emitc::VerbatimOp>(loc, "extern \"C\" {");

    for(auto [funcName, functionType] : collectedRegionOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        auto loweredInputTypes = lowerAllTypes(functionType.getInputs(), &highlevelTypeConverter);
        auto loweredOutputTypes = lowerAllTypes(functionType.getResults(), &highlevelTypeConverter);
        auto inputChannelsTupleType = wrapAllInTemplate(rewriter.getContext(), "Dppm::InputChannels", loweredInputTypes);
        auto outputChannelsTupleType = wrapAllInTemplate(rewriter.getContext(), "Dppm::OutputChannels", loweredOutputTypes);
        auto regionType = getRegionType(rewriter.getContext(), inputChannelsTupleType, outputChannelsTupleType);
        std::vector<Type> declFuncTypes = {regionType};
        for(auto type : loweredInputTypes){
            declFuncTypes.push_back(getChannelPointer(rewriter.getContext(), type));
        }
        for(auto type : loweredOutputTypes){
            declFuncTypes.push_back(getChannelPointer(rewriter.getContext(), type));
        }
        auto originalFunc = rewriter.create<emitc::FuncOp>(loc, "init_" + funcName, FunctionType::get(rewriter.getContext(), declFuncTypes, {}));
        rewriter.restoreInsertionPoint(savedPosition);
    }

    for(auto [funcName, inputTypes, outputTypes, symbolRefAttr, regionType, processOpName] : collectedInstantiateOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        size_t inputTypesAmount = inputTypes.size();
        size_t outputTypesAmount = outputTypes.size();
        std::vector<Type> functionInputTypes = {regionType};
        for (auto inType : inputTypes){
            functionInputTypes.push_back(getChannelPointer(rewriter.getContext(), inType));
        }
        for (auto outType : outputTypes){
            functionInputTypes.push_back(getChannelPointer(rewriter.getContext(), outType));
        }
        auto functionType = FunctionType::get(&getContext(), functionInputTypes, {});
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName, functionType);
        auto entryBlock = functionHolder.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);
        Value inputRegion = entryBlock->getArgument(0);
        std::vector<Value> inputChannelValues = std::vector<Value>(entryBlock->getArguments().begin()+1, entryBlock->getArguments().begin() + 1 + inputTypesAmount);
        std::vector<Value> outputChannelValues = std::vector<Value>(entryBlock->getArguments().begin()+1+inputTypesAmount, entryBlock->getArguments().end());
        auto inputWrapper = rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{autoType}, "Dppm::InputChannels", inputChannelValues).getResult(0);
        auto outputWrapper = rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{autoType}, "Dppm::OutputChannels", outputChannelValues).getResult(0);
        auto functionName = (StringRef{symbolRefAttr.getRootReference()} + "_dpm_adapter").str();
        auto functionValue = rewriter.create<emitc::ConstantOp>(loc, autoType, emitc::OpaqueAttr::get(&getContext(), functionName));
        std::string kpnOrSdfFunction;
        if(processOpName == "dfg.operator"){
            kpnOrSdfFunction = "AddSdfProcess";
        } else if (processOpName == "dfg.process"){
            kpnOrSdfFunction = "AddKpnProcess";
        }else{
            signalPassFailure();
        }
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, kpnOrSdfFunction, ValueRange{inputRegion, functionValue, inputWrapper, outputWrapper});
        rewriter.create<emitc::ReturnOp>(loc, (Value)0);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    for(auto [funcName, inputTypes, outputTypes, symbolRefAttr, regionType] : collectedEmbedOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        size_t inputTypesAmount = inputTypes.size();
        size_t outputTypesAmount = outputTypes.size();
        std::vector<Type> functionInputTypes = {regionType};
        for (auto inType : inputTypes){
            functionInputTypes.push_back(getChannelPointer(rewriter.getContext(), inType));
        }
        for (auto outType : outputTypes){
            functionInputTypes.push_back(getChannelPointer(rewriter.getContext(), outType));
        }
        auto regionInputType = wrapAllInTemplate(rewriter.getContext(), "Dppm::InputChannels", inputTypes);
        auto regionOutputType = wrapAllInTemplate(rewriter.getContext(), "Dppm::OutputChannels", outputTypes);
        auto newRegionType = getRegionType(rewriter.getContext(), regionInputType, regionOutputType);
        auto functionType = FunctionType::get(&getContext(), functionInputTypes, {});
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName, functionType);
        auto entryBlock = functionHolder.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);
        Value inputRegion = entryBlock->getArgument(0);
        std::vector<Value> inputChannelValues = std::vector<Value>(entryBlock->getArguments().begin()+1, entryBlock->getArguments().begin() + 1 + inputTypesAmount);
        std::vector<Value> outputChannelValues = std::vector<Value>(entryBlock->getArguments().begin()+1+inputTypesAmount, entryBlock->getArguments().end());
        auto inputWrapper = rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{autoType}, "Dppm::InputChannels", inputChannelValues).getResult(0);
        auto outputWrapper = rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{autoType}, "Dppm::OutputChannels", outputChannelValues).getResult(0);
        auto regionPointer = rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{newRegionType}, "AddRegion", ValueRange{inputRegion, inputWrapper, outputWrapper}).getResult(0);
        std::vector<Value> passToNextInit = { regionPointer };
        for(size_t i = 1 ; i < entryBlock->getNumArguments() ; i++){
            passToNextInit.push_back(entryBlock->getArgument(i));
        }
        auto functionName = ("init_" + StringRef{symbolRefAttr.getRootReference()}).str();
        auto functionValue = rewriter.create<emitc::ConstantOp>(loc, autoType, emitc::OpaqueAttr::get(&getContext(), functionName));
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, functionName, passToNextInit);
        rewriter.create<emitc::ReturnOp>(loc, (Value)0);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    for(auto [funcName, channelElementType, regionType] : collectedChannelOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        auto castedchannelElementType = llvm::cast<emitc::OpaqueType>(channelElementType);
        auto channelType = getChannelPointer(rewriter.getContext(), channelElementType);
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName, FunctionType::get(&getContext(), {regionType}, {channelType}));
        auto entryBlock = functionHolder.addEntryBlock();
        Value inputRegion = entryBlock->getArgument(0);
        rewriter.setInsertionPointToStart(entryBlock);
        auto returnValue = rewriter.create<emitc::CallOpaqueOp>(loc, channelType, "AddChannel", ValueRange{inputRegion}, ArrayAttr{}, rewriter.getTypeArrayAttr(channelElementType)).getResult(0);
        rewriter.create<emitc::ReturnOp>(loc, returnValue);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    for(auto [funcName, channelElementType] : collectedPullOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        auto channelType = getRTChannelPointer(rewriter.getContext(), channelElementType);
        auto arrayPointerType = emitc::PointerType::get(rewriter.getContext(), channelElementType);
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName, FunctionType::get(&getContext(), {channelType, arrayPointerType}, {}));
        auto entryBlock = functionHolder.addEntryBlock();
        Value inputChannel = entryBlock->getArgument(0);
        Value inputArrayPointer = entryBlock->getArgument(1);
        rewriter.setInsertionPointToStart(entryBlock);
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "Pull", ValueRange{inputChannel, inputArrayPointer});
        rewriter.create<emitc::ReturnOp>(loc, (Value)0);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    for(auto [funcName, channelElementType] : collectedPushOps){
        auto savedPosition = rewriter.saveInsertionPoint();
        auto channelType = getRTChannelPointer(rewriter.getContext(), channelElementType);
        auto arrayPointerType = emitc::PointerType::get(rewriter.getContext(), channelElementType);
        auto functionHolder = rewriter.create<emitc::FuncOp>(loc, funcName, FunctionType::get(&getContext(), {channelType, arrayPointerType}, {}));
        auto entryBlock = functionHolder.addEntryBlock();
        Value inputChannel = entryBlock->getArgument(0);
        Value inputArrayPointer = entryBlock->getArgument(1);
        rewriter.setInsertionPointToStart(entryBlock);
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "Push", ValueRange{inputChannel, inputArrayPointer});
        rewriter.create<emitc::ReturnOp>(loc, (Value)0);
        rewriter.restoreInsertionPoint(savedPosition);
    }

    rewriter.create<emitc::VerbatimOp>(loc, "} // extern C end");

}

} // namespace

std::unique_ptr<Pass> mlir::createConvertDfgToDpmWrappersPass()
{
    return std::make_unique<ConvertDfgToDpmWrappersPass>();
}
