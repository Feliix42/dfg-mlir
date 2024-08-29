/// Implementation of the DfgToDpmCalls pass.
///
/// @file
/// @author     Fabius Mayer-Uhma (fabius.mayer-uhma@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToDpmCalls/DfgToDpmCalls.h"

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

#include <regex>
#include "dfg-mlir/Conversion/Utils.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTODPMCALLS
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
namespace {

struct ConvertDfgToDpmCallsPass : public mlir::impl::ConvertDfgToDpmCallsBase<
                                      ConvertDfgToDpmCallsPass> {
    void runOnOperation() final;
};

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

std::string GetAdapterLocName(Operation *operation){
    std::string opName = operation->getName().getStringRef().str();
    std::replace(opName.begin(),opName.end(),'.','_');
    std::string adapterName = opName + "_at_" + GetLocationString(operation->getLoc()) + "_dpm_adapter";
    return adapterName;
}

Type genericPointer(MLIRContext *context){
    return LLVM::LLVMPointerType::get(context);
}

Type voidType(MLIRContext *context){
    return LLVM::LLVMVoidType::get(context);
}

struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ChannelOp channelOp,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = channelOp.getLoc();

        std::string adapterName = GetAdapterLocName(channelOp);

        auto parentBlock = channelOp->getBlock();
        auto regionPointer = parentBlock->getArgument(0);

        auto channelPtr = rewriter.create<LLVM::CallOp>(loc, TypeRange{genericPointer(rewriter.getContext())}, adapterName, ValueRange{regionPointer}).getResult();
        rewriter.replaceOp(channelOp, ValueRange{channelPtr, channelPtr});
        return success();
    }
};

struct InstantiateOpLowering : public OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        InstantiateOp instantiateOp,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = instantiateOp.getLoc();
        auto adapterName = GetAdapterLocName(instantiateOp);

        auto parentBlock = instantiateOp->getBlock();
        auto regionPointer = parentBlock->getArgument(0);

        std::vector<Value> functionValues;
        functionValues.push_back(regionPointer);
        for(auto value : adaptor.getInputs()){
            functionValues.push_back(value);
        }
        for(auto value : adaptor.getOutputs()){
            functionValues.push_back(value);
        }

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, adapterName, functionValues);
        rewriter.eraseOp(instantiateOp);
        return success();
    }
};


struct EmbedOpLowering : public OpConversionPattern<EmbedOp> {
    using OpConversionPattern<EmbedOp>::OpConversionPattern;

    EmbedOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<EmbedOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        EmbedOp embedOp,
        EmbedOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = embedOp.getLoc();

        auto parentBlock = embedOp->getBlock();
        auto regionPointer = parentBlock->getArgument(0);

        std::vector<Value> allValues = {regionPointer};
        for(auto value : adaptor.getInputs()){
            allValues.push_back(value);
        }
        for(auto value : adaptor.getOutputs()){
            allValues.push_back(value);
        }
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, GetAdapterLocName(embedOp), allValues);
        rewriter.eraseOp(embedOp);
        return success();
    }
};


struct PushOpLowering : public OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        PushOp pushOp,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = pushOp.getLoc();

		auto dataLayout = DataLayout(pushOp->getParentOfType<ModuleOp>());
		auto dataAlignment = dataLayout.getTypeABIAlignment(pushOp.getInp().getType());

        // Create a local "c-style-array" and store the data

        auto arraySize = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1)).getResult();
		auto dataPointer = rewriter.create<LLVM::AllocaOp>(loc, genericPointer(rewriter.getContext()), pushOp.getInp().getType(), arraySize, dataAlignment).getResult();

        rewriter.create<LLVM::StoreOp>(loc,adaptor.getInp(), dataPointer);

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, GetAdapterLocName(pushOp), ValueRange{adaptor.getChan(), dataPointer});

        rewriter.eraseOp(pushOp);
        return success();
    }
};

struct PullOpLowering : public OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    PullOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        PullOp pullOp,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = pullOp.getLoc();

		auto dataLayout = DataLayout(pullOp->getParentOfType<ModuleOp>());
		auto dataAlignment = dataLayout.getTypeABIAlignment(pullOp.getOutp().getType());

		// Move the ownership of the data into the current LLVMIR frame by allocating a copy

        // TODO check if there are issues when copying the std::array into this "c-style-array"

        auto arraySize = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1)).getResult();
		auto dataPointer = rewriter.create<LLVM::AllocaOp>(loc, genericPointer(rewriter.getContext()), pullOp.getOutp().getType(), arraySize, dataAlignment).getResult();

		rewriter.create<LLVM::CallOp>(
			loc,
            TypeRange{},
            GetAdapterLocName(pullOp),
            ValueRange{adaptor.getChan(), dataPointer});

        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
            pullOp,
            pullOp.getOutp().getType(),
            dataPointer,
			dataAlignment);
        return success();
    }
};

struct OutputOpLowering : public OpConversionPattern<OutputOp> {
    using OpConversionPattern<OutputOp>::OpConversionPattern;

    OutputOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OutputOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OutputOp outputOp,
        OutputOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = outputOp.getLoc();
        auto parentRegion = outputOp->getParentRegion();
        auto operandAmount = outputOp->getOperands().size();
        int index = 0;
        for(auto value : outputOp.getOperands()){
            rewriter.create<LLVM::StoreOp>(loc, value, parentRegion->getArgument(parentRegion->getNumArguments() - operandAmount + index));
            index ++;
        }
        rewriter.eraseOp(outputOp);
        return success();
    }
};

struct ProcessOpLowering : public OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ProcessOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ProcessOp processOp,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = processOp.getLoc();
        // TODO this all only works if the bodies contain exactly one block
        auto processName = processOp.getSymName().str() + "_ORIGINAL_dpm_adapter";
		auto functionType = adaptor.getFunctionType();
        std::vector<Type> allTypes = map(combine(functionType.getInputs(), functionType.getResults()), [this](auto t){ return (Type)typeConverter->convertType(t);});
        auto functionHolder = rewriter.create<LLVM::LLVMFuncOp>(loc, processName, LLVM::LLVMFunctionType::get(rewriter.getContext(), voidType(rewriter.getContext()), allTypes, false));
		bool hasLoop = false;
		// lower loopOps because SSA operands are linked to the process and not to the loopOp
		// if there is a loopOp but the body at the end of the function and add required arguments
		processOp.walk([&](LoopOp loopOp){
			hasLoop = true;
			for(auto &block : loopOp.getBody()){
				for(auto arg : adaptor.getBody().front().getArguments()){
					block.addArgument(arg.getType(), loc);
				}
			}
			rewriter.inlineRegionBefore(loopOp.getBody(), functionHolder.getBody(), functionHolder.getBody().end());
			rewriter.eraseOp(loopOp);
		});
        rewriter.inlineRegionBefore(adaptor.getBody(), functionHolder.getBody(), functionHolder.begin());
        rewriter.convertRegionTypes(&functionHolder.getBody(), *typeConverter);
		if(hasLoop){
			rewriter.setInsertionPointToEnd(&functionHolder.getBody().front());
	        rewriter.create<LLVM::BrOp>(loc, functionHolder.getBody().front().getArguments(), &functionHolder.getBody().back());
			rewriter.setInsertionPointToEnd(&functionHolder.getBody().back());
	        rewriter.create<LLVM::BrOp>(loc, functionHolder.getBody().back().getArguments(), &functionHolder.getBody().back());
		} else {
    	    rewriter.setInsertionPointToEnd(&functionHolder.getBody().back());
	        rewriter.create<LLVM::ReturnOp>(loc, (Value)0);
		}
        rewriter.eraseOp(processOp);
        return success();
    }
};


struct OperatorOpLowering : public OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    OperatorOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OperatorOp operatorOp,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if(operatorOp.getIterArgs().size() > 0){
            llvm::errs() << "Cannot lower iterargs to SDF\n";
            return failure();
        }
        auto loc = operatorOp.getLoc();
        auto functionType = operatorOp.getFunctionType();
        auto inputTypes = functionType.getInputs();
        auto outputTypes = functionType.getResults();
        auto processName = operatorOp.getSymName().str() + "_ORIGINAL_dpm_adapter";
        auto allTypes = combine(inputTypes, outputTypes);
        auto pointerTypes = map(allTypes, [&rewriter](auto){ return genericPointer(rewriter.getContext()); });
        auto functionHolder = rewriter.create<LLVM::LLVMFuncOp>(loc, processName, LLVM::LLVMFunctionType::get(rewriter.getContext(), voidType(rewriter.getContext()), pointerTypes, false));
        auto entryBlock = functionHolder.addEntryBlock(rewriter);
        rewriter.setInsertionPointToEnd(entryBlock);
        std::vector<Value> castedValues;
        for(size_t i = 0 ; i < inputTypes.size(); i++){
            castedValues.push_back(rewriter.create<LLVM::LoadOp>(loc,allTypes[i],entryBlock->getArgument(i)).getResult());
        }
        for(size_t i = 0 ; i < outputTypes.size(); i++){
            castedValues.push_back(entryBlock->getArgument(i + inputTypes.size()));
        }
        // convert output arguments of following regions to PTR
        for(auto &block : adaptor.getBody()){
            block.eraseArguments(block.getNumArguments() - outputTypes.size(), outputTypes.size());
            for(size_t i = 0 ; i < outputTypes.size(); i++){
                block.addArgument(genericPointer(rewriter.getContext()), loc);
            }
        }
        rewriter.create<LLVM::BrOp>(loc, ValueRange{castedValues}, &adaptor.getBody().front());
        rewriter.inlineRegionBefore(adaptor.getBody(), functionHolder.getBody(), functionHolder.end());
        rewriter.setInsertionPointToEnd(&functionHolder.getBody().back());
        rewriter.create<LLVM::ReturnOp>(loc, (Value)0);
        rewriter.eraseOp(operatorOp);
        return success();
    }
};




struct RegionOpLowering : public OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    RegionOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        RegionOp regionOp,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = regionOp.getLoc();

        auto functionArgumentTypes = std::vector<Type>{};
        auto regionUntypedPtrType = genericPointer(rewriter.getContext());
        for(auto inputArg : regionOp.getFunctionType().getInputs()){
            functionArgumentTypes.push_back(typeConverter->convertType(inputArg));
        }
        for(auto outputArg : regionOp.getFunctionType().getResults()){
            functionArgumentTypes.push_back(typeConverter->convertType(outputArg));
        }
        functionArgumentTypes.push_back(regionUntypedPtrType);

        auto functionHolder = rewriter.create<LLVM::LLVMFuncOp>(loc, ("init_"+regionOp.getSymName()).str(), LLVM::LLVMFunctionType::get(rewriter.getContext(), voidType(rewriter.getContext()), functionArgumentTypes, false));
        rewriter.inlineRegionBefore(adaptor.getBody(), functionHolder.getBody(), functionHolder.end());
        rewriter.convertRegionTypes(&functionHolder.getBody(), *typeConverter);

        rewriter.setInsertionPointToEnd(&functionHolder.getBody().back());
        rewriter.create<LLVM::ReturnOp>(loc, (Value)0);

        // TODO check if this works for multiple blocks
        for (Block &block : functionHolder.getBody()){
            block.insertArgument(0u,regionUntypedPtrType, loc);
        }
        rewriter.eraseOp(regionOp);
        return success();
    }
};

void ConvertDfgToDpmCallsPass::runOnOperation()
{
    Operation* op = getOperation();
    TypeConverter highLevelConverter;
    TypeConverter runtimeConverter;

    OpBuilder rewriter(&getContext());

    highLevelConverter.addConversion([](Type t) { return t; });
    highLevelConverter.addConversion([&rewriter](InputType){return genericPointer(rewriter.getContext());});
    highLevelConverter.addConversion([&rewriter](OutputType){return genericPointer(rewriter.getContext());});

    runtimeConverter.addConversion([](Type t) { return t; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<ChannelOpLowering, InstantiateOpLowering, EmbedOpLowering, RegionOpLowering, ProcessOpLowering, PushOpLowering, PullOpLowering, OperatorOpLowering, OutputOpLowering>(highLevelConverter, patterns.getContext());

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        LLVM::LLVMDialect>();

    // target.addIllegalDialect<DfgDialect>();

    // use PartialConversion because error messages are better
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createConvertDfgToDpmCallsPass()
{
    return std::make_unique<ConvertDfgToDpmCallsPass>();
}
