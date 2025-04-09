/// Implementation of DfgToVitis pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToVitis/DfgToVitis.h"

#include "dfg-mlir/Conversion/Passes.h"
#include "dfg-mlir/Conversion/ScfToVitis/ScfToVitis.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "dfg-mlir/Dialect/vitis/Enums.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "dfg-mlir/Dialect/vitis/Transforms/InsertIncludes.h"
#include "dfg-mlir/Dialect/vitis/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Process.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {

struct ConvertProcessToFunc : OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ConvertProcessToFunc(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto converter = getTypeConverter();
        auto loc = op->getLoc();
        auto funcTy = cast<FunctionType>(
            converter->convertType(adaptor.getFunctionType()));

        auto funcOp = rewriter.create<vitis::FuncOp>(
            loc,
            op.getSymName(),
            funcTy,
            ArrayAttr{},
            ArrayAttr{});
        auto funcLoc = funcOp.getLoc();
        Block* funcEntryBlock = rewriter.createBlock(&funcOp.getBody());
        IRMapping mapper;
        for (auto type : funcTy.getInputs())
            funcEntryBlock->addArgument(type, funcLoc);

        rewriter.setInsertionPointToStart(funcEntryBlock);
        rewriter.create<vitis::PragmaInlineOp>(loc, true);
        for (auto [arg, newArg] :
             llvm::zip(op.getBody().getArguments(), funcOp.getArguments())) {
            auto casted = converter->materializeSourceConversion(
                rewriter,
                funcLoc,
                arg.getType(),
                newArg);
            mapper.map(arg, casted);
        }
        for (auto &op : op.getBody().getOps()) {
            if (auto loopOp = dyn_cast<LoopOp>(op)) {
                if (!loopOp.getIterArgs().empty())
                    return rewriter.notifyMatchFailure(
                        loopOp.getLoc(),
                        "Don't support iteration argument in this conversion, "
                        "please use other ops to implement similar logic.");
                for (auto &loopBodyOp : loopOp.getBody().getOps())
                    rewriter.clone(loopBodyOp, mapper);
            } else {
                rewriter.clone(op, mapper);
            }
        }

        rewriter.replaceOp(op, funcOp);
        return success();
    }
};

struct ConvertPullToStreamRead : OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    ConvertPullToStreamRead(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto pullChan = op.getChan();
        auto pulledValue = op.getResult();

        auto vitisStream =
            pullChan.getDefiningOp<UnrealizedConversionCastOp>()->getOperand(0);

        if (!isa<MemRefType>(pulledValue.getType())) {
            auto readOp = rewriter.create<vitis::StreamReadOp>(
                loc,
                dyn_cast<vitis::StreamType>(vitisStream.getType())
                    .getStreamType(),
                vitisStream);
            rewriter.replaceOp(op, readOp.getResult());
            return success();
        } else {
            // Create an array to store the pulled data
            auto memrefTy = cast<MemRefType>(pulledValue.getType());
            auto shape = memrefTy.getShape();
            // auto size = memrefTy.getNumElements();
            auto elemTy = memrefTy.getElementType();
            auto array = rewriter.create<vitis::VariableOp>(
                loc,
                vitis::ArrayType::get(shape, elemTy));
            // Create a nested loop to pull from stream
            SmallVector<Value> indices;
            for (auto dim : shape) {
                auto forOp = rewriter.create<vitis::ForOp>(loc, 0, dim);
                rewriter.setInsertionPointToEnd(&forOp.getBody().front());
                indices.push_back(forOp.getInductionVar());
            }
            auto curLoc = rewriter.getUnknownLoc();
            rewriter.create<vitis::PragmaPipelineOp>(loc);
            auto streamReadOp = rewriter.create<vitis::StreamReadOp>(
                curLoc,
                dyn_cast<vitis::StreamType>(vitisStream.getType())
                    .getStreamType(),
                vitisStream);
            rewriter.create<vitis::ArrayWriteOp>(
                curLoc,
                streamReadOp.getResult(),
                array.getResult(),
                indices);

            rewriter.replaceOp(op, array.getResult());
            return success();
        }

        return rewriter.notifyMatchFailure(loc, "Unable to convert pull.");
    }
};

struct ConvertPushToStreamWrite : OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    ConvertPushToStreamWrite(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp op,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto pushChan = op.getChan();
        auto pushedValue = op.getInp();

        auto vitisStream =
            pushChan.getDefiningOp<UnrealizedConversionCastOp>()->getOperand(0);

        if (!isa<MemRefType>(pushedValue.getType())) {
            auto writeOp = rewriter.create<vitis::StreamWriteOp>(
                loc,
                pushedValue,
                vitisStream);
            rewriter.replaceOp(op, writeOp);
            return success();
        } else {
            // The array was casted back to memref using unrealized casts
            auto array = pushedValue.getDefiningOp()->getOperand(0);
            auto memrefTy = cast<MemRefType>(pushedValue.getType());
            // Create a nested loop to repeatedly read from the channel
            SmallVector<Value> indices;
            for (auto dim : memrefTy.getShape()) {
                auto forOp = rewriter.create<vitis::ForOp>(loc, 0, dim);
                rewriter.setInsertionPointToEnd(&forOp.getBody().front());
                indices.push_back(forOp.getInductionVar());
            }
            auto curLoc = rewriter.getUnknownLoc();
            rewriter.create<vitis::PragmaPipelineOp>(loc);
            auto arrayReadOp =
                rewriter.create<vitis::ArrayReadOp>(curLoc, array, indices);
            rewriter.create<vitis::StreamWriteOp>(
                curLoc,
                arrayReadOp.getResult(),
                vitisStream);

            rewriter.eraseOp(op);
            return success();
        }

        return rewriter.notifyMatchFailure(loc, "Unable to convert pull.");
    }
};

struct ConvertRegionToFunc : OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    ConvertRegionToFunc(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        RegionOp op,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op->getLoc();
        auto funcTy = op.getFunctionType();

        // The function type is different from process func, which is pointer
        // type, so here don't use type converter
        SmallVector<Type> args;
        SmallVector<int64_t> argBufferSizes;
        for (auto inTy : funcTy.getInputs()) {
            auto elemTy = cast<OutputType>(inTy).getElementType();
            if (auto shapedTy = dyn_cast<ShapedType>(elemTy)) {
                elemTy = shapedTy.getElementType();
                argBufferSizes.push_back(shapedTy.getNumElements());
            } else
                argBufferSizes.push_back(1);
            args.push_back(
                vitis::PointerType::get(rewriter.getContext(), elemTy));
        }
        for (auto outTy : funcTy.getResults()) {
            auto elemTy = cast<InputType>(outTy).getElementType();
            if (auto shapedTy = dyn_cast<ShapedType>(elemTy)) {
                elemTy = shapedTy.getElementType();
                argBufferSizes.push_back(shapedTy.getNumElements());
            } else
                argBufferSizes.push_back(1);
            args.push_back(
                vitis::PointerType::get(rewriter.getContext(), elemTy));
        }

        auto funcOp = rewriter.create<vitis::FuncOp>(
            loc,
            op.getSymName(),
            rewriter.getFunctionType(args, TypeRange{}),
            ArrayAttr{},
            ArrayAttr{});
        funcOp->setAttr(
            "argBufferSizes",
            rewriter.getDenseI64ArrayAttr(argBufferSizes));
        funcOp->setAttr(
            "num_inputs",
            rewriter.getI64IntegerAttr(funcTy.getNumInputs()));
        funcOp->setAttr(
            "num_outputs",
            rewriter.getI64IntegerAttr(funcTy.getNumResults()));
        auto funcLoc = funcOp.getLoc();
        Block* funcEntryBlock = rewriter.createBlock(&funcOp.getBody());
        IRMapping mapper;
        for (auto type : args) funcEntryBlock->addArgument(type, funcLoc);

        rewriter.setInsertionPointToStart(funcEntryBlock);
        // Insert interface pragmas for each argument
        for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
            rewriter.create<vitis::PragmaInterfaceOp>(
                funcLoc,
                vitis::PragmaInterfaceMode::m_axi,
                funcOp.getArgument(i),
                "gmem_arg" + std::to_string(i),
                vitis::PragmaInterfaceMasterAxiOffset::slave);
            rewriter.create<vitis::PragmaInterfaceOp>(
                funcLoc,
                vitis::PragmaInterfaceMode::s_axilite,
                funcOp.getArgument(i),
                "control");
        }
        // Insert interface pragma to genereate the slave control port
        rewriter.create<vitis::PragmaReturnInterfaceOp>(funcLoc);
        for (auto [arg, newArg] :
             llvm::zip(op.getBody().getArguments(), funcOp.getArguments())) {
            // Same as FunctionType, here we create unrealized cast manually
            auto casted = rewriter.create<UnrealizedConversionCastOp>(
                funcLoc,
                arg.getType(),
                newArg);
            mapper.map(arg, casted.getResult(0));
        }
        // Always put channel definition at top
        for (auto &op : op.getBody().getOps())
            if (isa<ChannelOp>(op)) rewriter.clone(op, mapper);
        // So that a dataflow region is created after
        rewriter.create<vitis::PragmaDataflowOp>(funcLoc, [&] {
            for (auto &op : op.getBody().getOps())
                if (!isa<ChannelOp>(op)) rewriter.clone(op, mapper);
        });

        rewriter.replaceOp(op, funcOp);
        return success();
    }
};

struct ConvertChannelToStream : OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ConvertChannelToStream(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto channelTy = op.getEncapsulatedType();
        if (auto shapedTy = dyn_cast<ShapedType>(channelTy))
            channelTy = shapedTy.getElementType();
        if (!op.getBufferSize().has_value())
            return rewriter.notifyMatchFailure(
                loc,
                "No support for boundless channel");

        auto streamVar = rewriter.create<vitis::VariableOp>(
            loc,
            vitis::StreamType::get(rewriter.getContext(), channelTy));
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            loc,
            TypeRange{op.getInChan().getType(), op.getOutChan().getType()},
            streamVar.getResult());
        op.getInChan().replaceAllUsesWith(cast.getResult(0));
        op.getOutChan().replaceAllUsesWith(cast.getResult(1));

        rewriter.create<vitis::PragmaStreamOp>(
            loc,
            streamVar.getResult(),
            op.getBufferSize().value());
        rewriter.create<vitis::PragmaBindStorageOp>(
            loc,
            streamVar.getResult(),
            vitis::PragmaStorageType::fifo,
            vitis::PragmaStorageImpl::srl);

        rewriter.replaceOp(op, cast.getResults());
        return success();
    }
};

template<typename ConnectDirection>
struct ConvertConnectToCall : OpConversionPattern<ConnectDirection> {
    using OpConversionPattern<ConnectDirection>::OpConversionPattern;

    ConvertConnectToCall(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ConnectDirection>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ConnectDirection op,
        ConnectDirection::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto dir = isa<ConnectInputOp>(op);
        auto regionPort = op.getOperand(0);
        auto channelPort = op.getOperand(1);

        Type elemTy;
        if (dir)
            elemTy = cast<OutputType>(regionPort.getType()).getElementType();
        else
            elemTy = cast<InputType>(regionPort.getType()).getElementType();
        unsigned size = 1;
        if (auto shapedTy = dyn_cast<ShapedType>(elemTy)) {
            elemTy = shapedTy.getElementType();
            size = shapedTy.getNumElements();
        }

        auto regionPortDefOp =
            dyn_cast<UnrealizedConversionCastOp>(regionPort.getDefiningOp());
        auto channelPortDefOp =
            dyn_cast<UnrealizedConversionCastOp>(channelPort.getDefiningOp());

        OpBuilder builder(rewriter.getContext());
        auto callFunc = getHelperFunc(builder, op.getOperation(), elemTy, size);
        if (!callFunc)
            return rewriter.notifyMatchFailure(
                loc,
                "Cannot create/find helper function due to unsupported "
                "type.");

        if (!dir) {
            auto dataflowRegion =
                op->template getParentOfType<vitis::PragmaDataflowOp>();
            rewriter.setInsertionPointToEnd(
                &dataflowRegion.getDataflowRegion().front());
        }
        auto callOp = rewriter.create<vitis::CallOp>(
            loc,
            callFunc,
            dir ? ValueRange{regionPortDefOp.getOperand(0), channelPortDefOp.getOperand(0)}
                : ValueRange{
                      channelPortDefOp.getOperand(0),
                      regionPortDefOp.getOperand(0)});

        rewriter.replaceOp(op, callOp);
        return success();
    }

private:
    std::optional<std::string> getTypeSuffix(Type type) const
    {
        auto bitwidth = type.getIntOrFloatBitWidth();
        if (auto intTy = dyn_cast<IntegerType>(type)) {
            if (intTy.isUnsigned())
                return "ui" + std::to_string(bitwidth);
            else
                return "i" + std::to_string(bitwidth);
        } else if (isa<FloatType>(type)) {
            if (bitwidth == 16) return "half";
            if (bitwidth == 32) return "float";
            if (bitwidth == 64)
                return "double";
            else
                return std::nullopt;
        } else {
            return std::nullopt;
        }
    }
    vitis::FuncOp getHelperFunc(
        OpBuilder &builder,
        Operation* direction,
        Type type,
        unsigned size = 1) const
    {
        bool dir = isa<ConnectInputOp>(direction);
        auto moduleOp = direction->getParentOfType<ModuleOp>();

        vitis::FuncOp funcOp;
        auto typeSuffix = getTypeSuffix(type);
        if (!typeSuffix.has_value()) return funcOp;
        auto name = (dir ? "mem2stream_" : "stream2mem_") + typeSuffix.value()
                    + ((size == 1) ? "" : "_" + std::to_string(size));

        moduleOp.walk([&](vitis::FuncOp op) {
            auto funcName = op.getSymName();
            if (funcName == name) funcOp = op;
        });

        // If the func op is found
        if (funcOp.getOperation() != nullptr) return funcOp;
        // Else, create it and return
        auto loc = moduleOp.getLoc();
        builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
        if (dir)
            funcOp = createMem2Stream(loc, builder, name, type, size);
        else
            funcOp = createStream2Mem(loc, builder, name, type, size);
        builder.setInsertionPoint(direction);
        return funcOp;
    }
    vitis::FuncOp createMem2Stream(
        Location loc,
        OpBuilder &builder,
        StringRef name,
        Type type,
        unsigned size) const
    {
        SmallVector<Type> args;
        args.push_back(vitis::PointerType::get(builder.getContext(), type));
        args.push_back(vitis::StreamType::get(builder.getContext(), type));

        auto funcOp = builder.create<vitis::FuncOp>(
            loc,
            name,
            builder.getFunctionType(args, TypeRange{}),
            ArrayAttr{},
            ArrayAttr{});
        auto funcLoc = funcOp.getLoc();
        Block* funcEntryBlock = builder.createBlock(&funcOp.getBody());
        for (auto type : args) funcEntryBlock->addArgument(type, funcLoc);

        builder.setInsertionPointToStart(funcEntryBlock);
        builder.create<vitis::PragmaInlineOp>(loc, true);
        if (size == 1) {
            auto idx =
                builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
            auto read = builder.create<vitis::ArrayPointerReadOp>(
                loc,
                funcOp.getArgument(0),
                idx.getResult());
            builder.create<vitis::StreamWriteOp>(
                loc,
                read.getResult(),
                funcOp.getArgument(1));
        } else {
            auto forOp = builder.create<vitis::ForOp>(loc, 0, size, 1);
            builder.setInsertionPointToStart(&forOp.getBody().front());
            builder.create<vitis::PragmaPipelineOp>(loc);
            auto read = builder.create<vitis::ArrayPointerReadOp>(
                loc,
                funcOp.getArgument(0),
                forOp.getInductionVar());
            builder.create<vitis::StreamWriteOp>(
                loc,
                read.getResult(),
                funcOp.getArgument(1));
        }
        return funcOp;
    }
    vitis::FuncOp createStream2Mem(
        Location loc,
        OpBuilder &builder,
        StringRef name,
        Type type,
        unsigned size) const
    {
        SmallVector<Type> args;
        args.push_back(vitis::StreamType::get(builder.getContext(), type));
        args.push_back(vitis::PointerType::get(builder.getContext(), type));

        auto funcOp = builder.create<vitis::FuncOp>(
            loc,
            name,
            builder.getFunctionType(args, TypeRange{}),
            ArrayAttr{},
            ArrayAttr{});
        auto funcLoc = funcOp.getLoc();
        Block* funcEntryBlock = builder.createBlock(&funcOp.getBody());
        for (auto type : args) funcEntryBlock->addArgument(type, funcLoc);

        builder.setInsertionPointToStart(funcEntryBlock);
        builder.create<vitis::PragmaInlineOp>(loc, true);
        if (size == 1) {
            auto idx =
                builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
            auto read = builder.create<vitis::StreamReadOp>(
                loc,
                type,
                funcOp.getArgument(0));
            builder.create<vitis::ArrayPointerWriteOp>(
                loc,
                read.getResult(),
                funcOp.getArgument(1),
                idx.getResult());
        } else {
            auto forOp = builder.create<vitis::ForOp>(loc, 0, size, 1);
            builder.setInsertionPointToStart(&forOp.getBody().front());
            builder.create<vitis::PragmaPipelineOp>(loc);
            auto read = builder.create<vitis::StreamReadOp>(
                loc,
                type,
                funcOp.getArgument(0));
            builder.create<vitis::ArrayPointerWriteOp>(
                loc,
                read.getResult(),
                funcOp.getArgument(1),
                forOp.getInductionVar());
        }
        return funcOp;
    }
};

struct ConvertInstantiateToCall : OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    ConvertInstantiateToCall(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();

        vitis::FuncOp funcOp;
        auto moduleOp = op.getOperation()->getParentOfType<ModuleOp>();
        moduleOp.walk([&](vitis::FuncOp calledOp) {
            if (calledOp.getSymName()
                == op.getCallee().getRootReference().str())
                funcOp = calledOp;
        });
        if (!funcOp)
            return rewriter.notifyMatchFailure(
                loc,
                "Cannot find called function");

        SmallVector<Value> args;
        for (auto input : op.getInputs()) {
            //
            auto cast =
                dyn_cast<UnrealizedConversionCastOp>(input.getDefiningOp());
            args.push_back(cast.getOperand(0));
        }
        for (auto output : op.getOutputs()) {
            //
            auto cast =
                dyn_cast<UnrealizedConversionCastOp>(output.getDefiningOp());
            args.push_back(cast.getOperand(0));
        }

        auto callOp = rewriter.create<vitis::CallOp>(loc, funcOp, args);

        rewriter.replaceOp(op, callOp);
        return success();
    }
};
} // namespace

void mlir::populateDfgToVitisConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertProcessToFunc>(typeConverter, patterns.getContext());
    patterns.add<ConvertPullToStreamRead>(typeConverter, patterns.getContext());
    patterns.add<ConvertPushToStreamWrite>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertRegionToFunc>(typeConverter, patterns.getContext());
    patterns.add<ConvertChannelToStream>(typeConverter, patterns.getContext());
    patterns.add<ConvertConnectToCall<ConnectInputOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertConnectToCall<ConnectOutputOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertInstantiateToCall>(
        typeConverter,
        patterns.getContext());
}

namespace {
struct ConvertDfgToVitisPass
        : public impl::ConvertDfgToVitisBase<ConvertDfgToVitisPass> {
    void runOnOperation() final;
};
} // namespace

namespace {
struct DfgTypeConverter : public TypeConverter {
public:
    DfgTypeConverter(MLIRContext* context) : context(context)
    {
        addConversion([](Type type) { return type; });
        addConversion([this](InputType type) -> Type {
            auto channelTy = type.getElementType();
            Type elemTy;
            if (auto memrefTy = dyn_cast<MemRefType>(channelTy))
                elemTy = memrefTy.getElementType();
            else
                elemTy = channelTy;
            return vitis::StreamType::get(this->context, elemTy);
        });
        addConversion([this](OutputType type) -> Type {
            auto channelTy = type.getElementType();
            Type elemTy;
            if (auto memrefTy = dyn_cast<MemRefType>(channelTy))
                elemTy = memrefTy.getElementType();
            else
                elemTy = channelTy;
            return vitis::StreamType::get(this->context, elemTy);
        });
        addConversion([this](FunctionType type) -> Type {
            SmallVector<Type> args;
            for (auto inTy : type.getInputs())
                args.push_back(this->convertType(inTy));
            for (auto outTy : type.getResults())
                args.push_back(this->convertType(outTy));
            return FunctionType::get(this->context, args, TypeRange{});
        });
        addSourceMaterialization(materializeAsUnrealizedCast);
        addArgumentMaterialization(materializeAsUnrealizedCast);
        addTargetMaterialization(materializeAsUnrealizedCast);
    }

private:
    MLIRContext* context;
    static Value materializeAsUnrealizedCast(
        OpBuilder &builder,
        Type resultType,
        ValueRange inputs,
        Location loc)
    {
        if (inputs.size() != 1) return Value();
        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
    }
};
} // namespace

void ConvertDfgToVitisPass::runOnOperation()
{
    auto &context = getContext();

    DfgTypeConverter converter(&context);
    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    populateDfgToVitisConversionPatterns(converter, patterns);

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<vitis::VitisDialect>();
    target.addIllegalDialect<DfgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToVitisPass()
{
    return std::make_unique<ConvertDfgToVitisPass>();
}

void mlir::dfg::addConvertToVitisPasses(OpPassManager &pm)
{
    pm.addPass(dfg::createDfgOperatorToProcessPass());
    pm.addPass(dfg::createDfgInlineRegionPass());
    pm.addPass(dfg::createDfgLowerInsideToLinalgPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertMemrefToVitisPass());
    pm.addPass(createConvertDfgToVitisPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertArithIndexToVitisPass());
    pm.addPass(createConvertMathToVitisPass());
    pm.addPass(createConvertScfToVitisPass());
    pm.addPass(vitis::createVitisMergeCastChainPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(vitis::createVitisInsertIncludesPass());
    pm.addPass(createCanonicalizerPass());
}

void mlir::dfg::registerConvertToVitisPipelines()
{
    PassPipelineRegistration<>(
        "convert-to-vitis",
        "Convert everything to vitis dialect",
        [](OpPassManager &pm) { addConvertToVitisPasses(pm); });
}
