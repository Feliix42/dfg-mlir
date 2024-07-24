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

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTODPMCALLS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

Value mainRegion;
Type voidType;
Type genericPtrType;
Type stringPtrType;
Type wrapperType;
Type wrapperPtrType;
Type rtWrapperType;
Type rtWrapperPtrType;
Type channelUntypedType;
Type channelUntypedPtrType;
Type rtChannelUntypedType;
Type rtChannelUntypedPtrType;
Type regionUntypedPtrType;
Type applicationPtrType;
ModuleOp globalModule;
Block* beginningBlockOfModule;

// ========================================================
// Lowerings
// ========================================================

// IMPORTANT: THIS HAS TO BE IN AN UNIQUE NAMESPACE!! IF NOT YOUR CONVERSION PATTERNS
// MIGHT BE NAMED LIKE AN ALREADY EXISTING ONE WHICH LEADS TO UNTRACABLE ERRORS
namespace {

struct ConvertDfgToDpmCallsPass
        : public mlir::impl::ConvertDfgToDpmCallsBase<ConvertDfgToDpmCallsPass> {
    void runOnOperation() final;
};

struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ChannelOp channelOp,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = channelOp.getLoc();

		auto dataLayout = DataLayout(globalModule);
		auto dataSize = dataLayout.getTypeSize(adaptor.getEncapsulatedType());

        auto constant = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(dataSize));

        Block* parentBlock = channelOp->getBlock();
        auto parentRegionPtr = parentBlock->getArgument(parentBlock->getNumArguments()-1);

        auto newChannel = rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{
                typeConverter->convertType(channelOp.getInChan().getType())},
            rewriter.getStringAttr("C_INTERFACE_AddChannel"),
            ValueRange{parentRegionPtr, constant.getResult()});

        rewriter.replaceOp(
            channelOp,
            ValueRange{newChannel.getResult(), newChannel.getResult()});

        return success();
    }
};


struct
    InstantiateOpLowering
        : public mlir::OpConversionPattern<dfg::InstantiateOp> {
    using OpConversionPattern<dfg::InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<dfg::InstantiateOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        dfg::InstantiateOp instantiateOp,
        dfg::InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        auto loc = instantiateOp.getLoc();

        Block* parentBlock = instantiateOp->getBlock();
        auto parentRegionPtr = parentBlock->getArgument(parentBlock->getNumArguments()-1);

        auto arraySize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(1));

        auto inputWrapper = rewriter.create<LLVM::AllocaOp>(
            loc,
            wrapperPtrType,
            wrapperType,
            arraySize.getResult(),
            8);

        auto inputSize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(adaptor.getInputs().size()));

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            rewriter.getStringAttr("ChannelUntypedPtrsWrapperInit"),
            ValueRange{inputWrapper.getResult(), inputSize});

        size_t index = 0;
        for (auto input : adaptor.getInputs()) {
            rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{},
                rewriter.getStringAttr("ChannelUntypedPtrsWrapperSet"),
                ValueRange{
                    inputWrapper.getResult(),
                    input,
                    rewriter.create<LLVM::ConstantOp>(
                        loc,
                        rewriter.getI64Type(),
                        rewriter.getI64IntegerAttr(index++))});
        }

        auto outputWrapper = rewriter.create<LLVM::AllocaOp>(
            loc,
            wrapperPtrType,
            wrapperType,
            arraySize.getResult(),
            8);

        auto outputSize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(adaptor.getOutputs().size()));

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            rewriter.getStringAttr("ChannelUntypedPtrsWrapperInit"),
            ValueRange{outputWrapper.getResult(), outputSize});

        index = 0;
        for (auto output : adaptor.getOutputs()) {
            rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{},
                rewriter.getStringAttr("ChannelUntypedPtrsWrapperSet"),
                ValueRange{
                    outputWrapper.getResult(),
                    output,
                    rewriter.create<LLVM::ConstantOp>(
                        loc,
                        rewriter.getI64Type(),
                        rewriter.getI64IntegerAttr(index++))});
        }

		auto savedPoint = rewriter.saveInsertionPoint();
		rewriter.setInsertionPointToStart(beginningBlockOfModule);

		// make sure that string is null terminated because were using c interface

		StringRef functionName = instantiateOp.getCalleeAttr().getRootReference();
		std::string fieldName = ("process_" + functionName + "_global_string").str();
		std::string processNameString = ("process_" + functionName).str() + "\00";
		StringRef processName = StringRef(processNameString.c_str(), processNameString.length()+1);

		auto stringGlobalOp = rewriter.create<LLVM::GlobalOp>(
			loc,
			LLVM::LLVMArrayType::get(rewriter.getI8Type(), processName.size()),
			true,
			LLVM::Linkage{},
			fieldName,
			rewriter.getStringAttr(processName));

		rewriter.restoreInsertionPoint(savedPoint);

		auto stringAddress = rewriter.create<LLVM::AddressOfOp>(
			loc,
			stringPtrType, fieldName);

		auto functionAddress = rewriter.create<LLVM::AddressOfOp>(
			loc,
			genericPtrType,
			instantiateOp.getCalleeAttr().getRootReference());

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{LLVM::LLVMPointerType::get(rewriter.getContext())},
            rewriter.getStringAttr("C_INTERFACE_AddKpnProcess"),
            ValueRange{parentRegionPtr, stringAddress.getResult(), functionAddress.getResult(), inputWrapper.getResult(), outputWrapper.getResult()});
        rewriter.eraseOp(instantiateOp);

        return success();
    }
};

struct RegionOpLowering : public OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    RegionOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        RegionOp regionOp,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

		auto loc = regionOp.getLoc();
        auto function = rewriter.create<LLVM::LLVMFuncOp>(loc, ("init_"+regionOp.getSymName()).str(), LLVM::LLVMFunctionType::get(rewriter.getContext(), voidType, {regionUntypedPtrType}, false));

        rewriter.inlineRegionBefore(
            adaptor.getBody(), function.getBody(), function.end());

        rewriter.setInsertionPointToEnd(&function.getBody().back());
        rewriter.create<LLVM::ReturnOp>(loc, (Value)0);

        for(Block &block : function.getBody()){
            block.addArgument(regionUntypedPtrType, loc);
        }

        rewriter.eraseOp(regionOp);
        return success();
    }
};

struct PushOpLowering : public mlir::OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp pushOp,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

		// adaptor.getInput() has arbitrary type
		// store it locally, cast location to generic pointer

		// this has to be done because mlir-translate checks types of pointers
		// this gets optimized in llvmir

		auto loc = pushOp.getLoc();

		auto dataLayout = DataLayout(globalModule);
		auto dataAlignment = dataLayout.getTypeABIAlignment(adaptor.getInp().getType());

        auto outputSize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(1));

		auto copyPointer = rewriter.create<LLVM::AllocaOp>(
			loc,
			LLVM::LLVMPointerType::get(rewriter.getContext()),
			adaptor.getInp().getType(), outputSize.getResult(), dataAlignment);

		rewriter.create<LLVM::StoreOp>(
			loc,
			adaptor.getInp(),
			copyPointer);

		auto castedPointer = rewriter.create<LLVM::BitcastOp>(
			loc, genericPtrType, copyPointer.getResult());

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            pushOp,
            TypeRange{},
            rewriter.getStringAttr("C_INTERFACE_PushBytes"),
            ValueRange{adaptor.getChan(), castedPointer.getResult()});
        return success();
    }
};

struct PullOpLowering : public mlir::OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    PullOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp pullOp,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {


		auto loc = pullOp.getLoc();

		auto dataLayout = DataLayout(globalModule);
		auto dataAlignment = dataLayout.getTypeABIAlignment(pullOp.getOutp().getType());

		auto charPointer = rewriter.create<LLVM::CallOp>(
			loc,
            TypeRange{genericPtrType},
            rewriter.getStringAttr("C_INTERFACE_PopBytes"),
            ValueRange{adaptor.getChan()});

        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
            pullOp,
            pullOp.getOutp().getType(),
            charPointer.getResult(),
			dataAlignment);
        return success();
    }
};

struct ProcessOpLowering
        : public mlir::OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ProcessOpLowering(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ProcessOp processOp,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override

    {

        auto loc = processOp.getLoc();

        LLVM::LLVMFunctionType functionType = LLVM::LLVMFunctionType::get(
            rewriter.getContext(),
            LLVM::LLVMVoidType::get(rewriter.getContext()),
            {applicationPtrType, rtWrapperPtrType, rtWrapperPtrType},
            false);
        auto functionHolder = rewriter.create<LLVM::LLVMFuncOp>(
            loc,
            processOp.getSymName(),
            functionType);
        // TODO AA MEMORY LEAK :(
        Block* newBlock = new Block();

        newBlock->addArgument(applicationPtrType, adaptor.getBody().getLoc());
        newBlock->addArgument(rtWrapperPtrType, adaptor.getBody().getLoc());
        newBlock->addArgument(rtWrapperPtrType, adaptor.getBody().getLoc());

        Value inputWrapper = newBlock->getArgument(1);
        Value outputWrapper = newBlock->getArgument(2);

        rewriter.setInsertionPointToEnd(newBlock);

        std::vector<Value> allValues;

        size_t index = 0;
        for (auto input : adaptor.getFunctionType().getInputs()) {
            auto inputType = typeConverter->convertType(input);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(
                loc,
                rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(index++));
            auto newValue = rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{inputType},
                rewriter.getStringAttr("RTChannelUntypedPtrsWrapperGet"),
                ValueRange{inputWrapper, indexValue.getResult()});
            allValues.push_back(newValue.getResult());
        }

        index = 0;
        for (auto output : adaptor.getFunctionType().getResults()) {
            auto outputType = typeConverter->convertType(output);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(
                loc,
                rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(index++));
            auto newValue = rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{outputType},
                rewriter.getStringAttr("RTChannelUntypedPtrsWrapperGet"),
                ValueRange{outputWrapper, indexValue.getResult()});
            allValues.push_back(newValue.getResult());
        }

        rewriter.create<cf::BranchOp>(
            loc,
            allValues,
            &adaptor.getBody().back());
        rewriter.setInsertionPointToEnd(&adaptor.getBody().back());
        newBlock->insertBefore(&adaptor.getBody().back());

        FailureOr<Block*> convertedBlock =
            rewriter.convertRegionTypes(&adaptor.getBody(), *typeConverter);

        if (*convertedBlock == nullptr) return failure();

        rewriter.inlineRegionBefore(
            adaptor.getBody(),
            functionHolder.getBody(),
            functionHolder.end());

        rewriter.setInsertionPointToEnd(&functionHolder.getRegion().back());
        rewriter.create<LLVM::ReturnOp>(loc, TypeRange({}), ValueRange({}));
        rewriter.eraseOp(processOp);

        return success();
    }
};

void insert_extern_function(OpBuilder localRewriter, Location location, std::string functionName, Type returnType, llvm::ArrayRef<Type> argumentTypes){
    localRewriter.create<LLVM::LLVMFuncOp>(
        location,
        functionName,
        LLVM::LLVMFunctionType::get(localRewriter.getContext(), returnType, argumentTypes, false)
    );
}

void ConvertDfgToDpmCallsPass::runOnOperation() {
    Operation* op = getOperation();
    TypeConverter highLevelConverter;
    TypeConverter runtimeConverter;

	Value mainApplication;
	Value mainManager;

	OpBuilder rewriter(&getContext());

    // all types used in the lowering (TODO rethink)
    regionUntypedPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    channelUntypedPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto managerPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    voidType = LLVM::LLVMVoidType::get(&getContext());
	genericPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    wrapperType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("ChannelUntypedPtrsWrapper"),
        {rewriter.getI64Type(), genericPtrType });
    wrapperPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    rtWrapperType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("RTChannelUntypedPtrsWrapper"),
        {rewriter.getI64Type(), genericPtrType});
    rtWrapperPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    channelUntypedType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("ChannelUntyped"),
        {rewriter.getI32Type()});
    rtChannelUntypedType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("RTChannelUntyped"),
        {rewriter.getI32Type()});
    rtChannelUntypedPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
	auto regionUntypedType = LLVM::LLVMStructType::getNewIdentified(
            &getContext(),
            StringRef("class.Dppm::RegionUntyped"),
            {rewriter.getI32Type()});
	auto applicationType = LLVM::LLVMStructType::getNewIdentified(
            &getContext(),
            StringRef("class.Dppm::Application"),
            {rewriter.getI32Type()});
    applicationPtrType =
        LLVM::LLVMPointerType::get(rewriter.getContext());
	auto managerType = LLVM::LLVMStructType::getNewIdentified(
            &getContext(),
            StringRef("class.Dppm::Manager"),
            {rewriter.getI32Type()});
	stringPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());


	// first search for the begin of the module so we can add all of our global string constants and extern functions here
    op->walk([&](ModuleOp moduleOp) {
        auto loc = moduleOp.getLoc();
		llvm::errs() << "found ModuleOp: " << moduleOp.getName() << "/" << loc << "\n";
		beginningBlockOfModule = &moduleOp.getBodyRegion().front();
		globalModule = moduleOp;
        OpBuilder localRewriter(moduleOp->getContext());
        localRewriter.setInsertionPointToStart(moduleOp.getBody());
        insert_extern_function(localRewriter, loc,"C_INTERFACE_AddChannel",channelUntypedPtrType,{regionUntypedPtrType, rewriter.getI32Type()});
        insert_extern_function(localRewriter, loc,"ChannelUntypedPtrsWrapperSet",voidType,{wrapperPtrType, channelUntypedPtrType, rewriter.getI64Type()});
        insert_extern_function(localRewriter, loc,"RTChannelUntypedPtrsWrapperGet",rtChannelUntypedPtrType,{rtWrapperPtrType, rewriter.getI64Type()});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_PopBytes",genericPtrType,{rtChannelUntypedPtrType});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_PushBytes",voidType,{rtChannelUntypedPtrType, genericPtrType});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_GetMainRegion",regionUntypedPtrType,{applicationPtrType});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_CreateApplication",applicationPtrType,{managerPtrType});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_RunAndWaitForApplication",voidType,{managerPtrType,applicationPtrType});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_ManagerGetInstance",managerPtrType,{});
        insert_extern_function(localRewriter, loc,"ChannelUntypedPtrsWrapperInit",voidType,{wrapperPtrType, rewriter.getI64Type()});
        insert_extern_function(localRewriter, loc,"C_INTERFACE_AddKpnProcess",genericPtrType,{regionUntypedPtrType, stringPtrType, genericPtrType, wrapperPtrType, wrapperPtrType});

        // Insert main function
        auto functionHolder = localRewriter.create<LLVM::LLVMFuncOp>(loc,"main",LLVM::LLVMFunctionType::get(&getContext(), localRewriter.getI32Type(),{}, false));
        auto functionBlock = functionHolder.addEntryBlock(localRewriter);
        localRewriter.setInsertionPointToStart(functionBlock);
        auto manager = localRewriter.create<LLVM::CallOp>(loc,
            TypeRange{managerPtrType},
            localRewriter.getStringAttr("C_INTERFACE_ManagerGetInstance"),
            ValueRange{});
        auto application = localRewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{applicationPtrType},
            localRewriter.getStringAttr("C_INTERFACE_CreateApplication"),
        ValueRange{manager.getResult()});
		mainApplication = application.getResult();
		mainManager = manager.getResult();
        mainRegion = localRewriter.create<LLVM::CallOp>(loc,
                 TypeRange{regionUntypedPtrType},
                 localRewriter.getStringAttr(
                     "C_INTERFACE_GetMainRegion"),
                 ValueRange{application.getResult()})
             .getResult();

        //configure main region
        localRewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            localRewriter.getStringAttr("init_mainRegion"),
            ValueRange{mainRegion});

        // run application
        localRewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            localRewriter.getStringAttr("C_INTERFACE_RunAndWaitForApplication"),
            ValueRange{mainManager, mainApplication});
        // return 0
        auto returnValue = localRewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(0)).getResult();
        localRewriter.create<LLVM::ReturnOp>(loc, returnValue);

	});

    highLevelConverter.addConversion([](Type t) { return t; });
    highLevelConverter.addConversion([](InputType) {
        return channelUntypedPtrType;
    });
    highLevelConverter.addConversion([](OutputType) {
        return channelUntypedPtrType;
    });

    runtimeConverter.addConversion([](Type t) { return t; });
    runtimeConverter.addConversion([](InputType) {
        return rtChannelUntypedPtrType;
    });
    runtimeConverter.addConversion([](OutputType) {
        return rtChannelUntypedPtrType;
    });

	ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

	patterns.add<ChannelOpLowering, RegionOpLowering, InstantiateOpLowering>(
        highLevelConverter,
        patterns.getContext());

    patterns.add<
        PushOpLowering,
        PullOpLowering,
        ProcessOpLowering>(
        runtimeConverter,
        patterns.getContext());

    target.addLegalDialect<
        BuiltinDialect,
        func::FuncDialect,
        cf::ControlFlowDialect,
        LLVM::LLVMDialect>();

    //target.addIllegalDialect<DfgDialect>();

	// use PartialConversion because documentation is better
    if (failed(applyPartialConversion(op, target, std::move(patterns)))){
        signalPassFailure();
	}
}

} // namespace


std::unique_ptr<Pass> mlir::createConvertDfgToDpmCallsPass() {
    return std::make_unique<ConvertDfgToDpmCallsPass>();
}

