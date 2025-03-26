/// BufferizableOpInterface implementation
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/BufferizableOpInterfaceImpl.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <string>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::dfg;

namespace mlir {
namespace dfg {
namespace {

template<typename OpT>
struct ProcessLikeOpInterface
        : public BufferizableOpInterface::
              ExternalModel<ProcessLikeOpInterface<OpT>, OpT> {
    bool hasTensorSemantics(Operation* op) const
    {
        auto thisOp = cast<OpT>(op);
        auto funcTy = thisOp.getFunctionType();
        bool hasTensorInputs = llvm::any_of(funcTy.getInputs(), [](Type type) {
            if (auto outTy = dyn_cast<OutputType>(type))
                return isa<TensorType>(outTy.getElementType());
            return false;
        });
        bool hasTensorOutputs =
            llvm::any_of(funcTy.getResults(), [](Type type) {
                if (auto inTy = dyn_cast<InputType>(type))
                    return isa<TensorType>(inTy.getElementType());
                return false;
            });
        return hasTensorInputs || hasTensorOutputs;
    }
    FailureOr<BaseMemRefType> getBufferType(
        Operation* op,
        Value value,
        const BufferizationOptions &options,
        SmallVector<Value> &invocationStack) const
    {
        // Push the current value onto the invocation stack.
        invocationStack.push_back(value);
        auto popFromStack =
            llvm::make_scope_exit([&]() { invocationStack.pop_back(); });

        Type type = value.getType();
        // If the type is a dfg::OutputType
        if (auto outputTy = dyn_cast<dfg::OutputType>(type)) {
            auto elemTy = outputTy.getElementType();
            if (auto tensorTy = dyn_cast<TensorType>(elemTy)) {
                ArrayRef<int64_t> shape = tensorTy.getShape();
                Type tensorElemTy = tensorTy.getElementType();
                // Construct a MemRefType from the shape and element type.
                auto memrefType = MemRefType::get(shape, tensorElemTy);
                return memrefType;
            }
        }
        // If the type is a dfg::InputType
        if (auto inputTy = dyn_cast<dfg::InputType>(type)) {
            auto elemTy = inputTy.getElementType();
            if (auto tensorTy = dyn_cast<TensorType>(elemTy)) {
                ArrayRef<int64_t> shape = tensorTy.getShape();
                Type tensorElemTy = tensorTy.getElementType();
                auto memrefType = MemRefType::get(shape, tensorElemTy);
                return memrefType;
            }
        }
        // In the FuntionType of ProcessOp, there are only OutputType and
        // InputType, any other Types are unexpected
        return op->emitError("unexpected type for bufferization");
    }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &options) const
    {
        auto thisOp = cast<OpT>(op);
        auto funcTy = thisOp.getFunctionType();

        // Inputs bufferization
        SmallVector<Type> inputTypes;
        for (const auto &it : llvm::enumerate(funcTy.getInputs())) {
            Type inTy = it.value();
            Type inElemTy = cast<OutputType>(inTy).getElementType();
            size_t inIdx = it.index();
            if (isa<TensorType>(inElemTy)) {
                SmallVector<Value> invocationStack;
                auto memrefTy = getBufferType(
                    op,
                    thisOp.getBody().getArgument(inIdx),
                    options,
                    invocationStack);
                if (failed(memrefTy))
                    return rewriter.notifyMatchFailure(
                        op->getLoc(),
                        "Failed to get bufferization type from argument "
                            + std::to_string(inIdx));
                auto newInTy =
                    OutputType::get(rewriter.getContext(), memrefTy.value());
                inputTypes.push_back(newInTy);
                continue;
            }
            inputTypes.push_back(inTy);
        }
        // Outputs bufferization
        SmallVector<Type> outputTypes;
        for (const auto &it : llvm::enumerate(funcTy.getResults())) {
            Type outTy = it.value();
            Type outElemTy = cast<InputType>(outTy).getElementType();
            size_t inIdx = it.index();
            if (isa<TensorType>(outElemTy)) {
                SmallVector<Value> invocationStack;
                auto memrefTy = getBufferType(
                    op,
                    thisOp.getBody().getArgument(inIdx),
                    options,
                    invocationStack);
                if (failed(memrefTy))
                    return rewriter.notifyMatchFailure(
                        op->getLoc(),
                        "Failed to get bufferization type from argument "
                            + std::to_string(inIdx));
                auto newOutTy =
                    InputType::get(rewriter.getContext(), memrefTy.value());
                outputTypes.push_back(newOutTy);
                continue;
            }
            outputTypes.push_back(outTy);
        }

        // Create new ProcessOp with new signature
        auto newFuncTy = rewriter.getFunctionType(inputTypes, outputTypes);
        auto newOp =
            rewriter.create<OpT>(op->getLoc(), thisOp.getSymName(), newFuncTy);
        Block* processBlock = &newOp.getBody().front();
        rewriter.setInsertionPointToEnd(processBlock);
        // Clone the contents into ProcessOp with new block arguments
        IRMapping mapper;
        for (auto [oldArg, newArg] : llvm::zip(
                 thisOp.getBody().getArguments(),
                 newOp.getBody().getArguments()))
            mapper.map(oldArg, newArg);
        for (auto &opi : thisOp.getBody().getOps()) rewriter.clone(opi, mapper);

        rewriter.replaceOp(op, newOp);
        return success();
    }
};

struct PullOpInterface : public BufferizableOpInterface::
                             ExternalModel<PullOpInterface, dfg::PullOp> {
    bool bufferizesToMemoryRead(
        Operation* op,
        OpOperand &opOperand,
        const AnalysisState &state) const
    {
        return false;
    }
    bool bufferizesToMemoryWrite(
        Operation* op,
        OpOperand &opOperand,
        const AnalysisState &state) const
    {
        return true;
    }
    AliasingValueList getAliasingValues(
        Operation* op,
        OpOperand &opOperand,
        const AnalysisState &state) const
    {
        return {};
    }
    bool hasTensorSemantics(Operation* op) const
    {
        auto pullOp = cast<PullOp>(op);
        return llvm::isa<TensorType>(pullOp.getOutp().getType());
    }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &options) const
    {
        auto pullOp = cast<PullOp>(op);
        auto pullChan = pullOp.getChan();
        auto loc = pullOp.getLoc();

        // Get the bufferized types of channel
        auto pullChanElemTy =
            cast<TensorType>(pullChan.getType().getElementType());
        auto tensorShape = pullChanElemTy.getShape();
        auto tensorElemTy = pullChanElemTy.getElementType();
        auto bufferizedChanElemTy = MemRefType::get(tensorShape, tensorElemTy);
        auto bufferizedChanTy =
            OutputType::get(rewriter.getContext(), bufferizedChanElemTy);
        // First, insert unrealized cast to get the bufferized channel
        auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
            loc,
            TypeRange{bufferizedChanTy},
            pullChan);
        // Create a new bufferized PullOp
        auto newPullOp =
            rewriter.create<PullOp>(loc, unrealizedCast.getResult(0));
        // Cast the bufferized pull result back to tensor
        auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
            loc,
            newPullOp.getResult());

        rewriter.replaceOp(op, toTensorOp.getResult());
        return success();
    }
};

struct PushOpInterface : public BufferizableOpInterface::
                             ExternalModel<PushOpInterface, dfg::PushOp> {
    bool bufferizesToMemoryRead(
        Operation* op,
        OpOperand &opOperand,
        const AnalysisState &state) const
    {
        return true;
    }
    bool bufferizesToMemoryWrite(
        Operation* op,
        OpOperand &opOperand,
        const AnalysisState &state) const
    {
        return false;
    }
    AliasingValueList getAliasingValues(
        Operation* op,
        OpOperand &opOperand,
        const AnalysisState &state) const
    {
        return {};
    }
    bool hasTensorSemantics(Operation* op) const
    {
        auto pushOp = cast<PushOp>(op);
        return llvm::isa<TensorType>(pushOp.getInp().getType());
    }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &options) const
    {
        auto pushOp = cast<PushOp>(op);
        auto pushChan = pushOp.getChan();
        auto pushValue = pushOp.getInp();
        auto loc = pushOp.getLoc();

        // Get the bufferized types of channel
        auto pushChanElemTy =
            cast<TensorType>(pushChan.getType().getElementType());
        auto pushChanTensorShape = pushChanElemTy.getShape();
        auto pushChanTensorElemTy = pushChanElemTy.getElementType();
        auto bufferizedChanElemTy =
            MemRefType::get(pushChanTensorShape, pushChanTensorElemTy);
        auto bufferizedChanTy =
            InputType::get(rewriter.getContext(), bufferizedChanElemTy);
        // and pushed value
        auto pushValueElemTy = cast<TensorType>(pushValue.getType());
        auto pushValueTensorShape = pushValueElemTy.getShape();
        auto pushValueTensorElemTy = pushValueElemTy.getElementType();
        auto bufferizedValueTy =
            MemRefType::get(pushValueTensorShape, pushValueTensorElemTy);

        // First, insert unrealized cast to get the bufferized channel
        // and to_memref to bufferize the input tensor to push
        auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
            loc,
            TypeRange{bufferizedChanTy},
            pushChan);
        auto toMemref = rewriter.create<bufferization::ToMemrefOp>(
            loc,
            bufferizedValueTy,
            pushValue);
        // Create a new bufferized PushOp
        auto newPushOp = rewriter.create<PushOp>(
            loc,
            toMemref.getMemref(),
            unrealizedCast.getResult(0));

        rewriter.replaceOp(op, newPushOp);
        return success();
    }
};

struct RegionOpInterface : public BufferizableOpInterface::
                               ExternalModel<RegionOpInterface, dfg::RegionOp> {
    bool hasTensorSemantics(Operation* op) const
    {
        auto regionOp = cast<RegionOp>(op);
        auto funcTy = regionOp.getFunctionType();
        bool hasTensorInputs = llvm::any_of(funcTy.getInputs(), [](Type type) {
            auto outTy = cast<OutputType>(type);
            return isa<TensorType>(outTy.getElementType());
        });
        bool hasTensorOutputs =
            llvm::any_of(funcTy.getResults(), [](Type type) {
                auto inTy = cast<InputType>(type);
                return isa<TensorType>(inTy.getElementType());
            });
        return hasTensorInputs || hasTensorOutputs;
    }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &options) const
    {
        // rewriter.eraseOp(op);
        return success();
    }
};

struct ChannelOpInterface
        : public BufferizableOpInterface::
              ExternalModel<ChannelOpInterface, dfg::ChannelOp> {
    bool hasTensorSemantics(Operation* op) const
    {
        auto channelOp = cast<ChannelOp>(op);
        return isa<TensorType>(channelOp.getEncapsulatedType());
    }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &options) const
    {
        auto channelOp = cast<ChannelOp>(op);
        auto channelTy = cast<TensorType>(channelOp.getEncapsulatedType());
        auto channelSize = channelOp.getBufferSize();

        auto tensorShape = channelTy.getShape();
        auto tensorElemTy = channelTy.getElementType();
        auto bufferizedChanTy = MemRefType::get(tensorShape, tensorElemTy);

        auto newChannelOp = rewriter.create<ChannelOp>(
            op->getLoc(),
            bufferizedChanTy,
            channelSize.has_value() ? channelSize.value() : 0);

        rewriter.replaceOp(op, newChannelOp);
        return success();
    }
};

} // namespace
} // namespace dfg
} // namespace mlir

void mlir::dfg::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext* ctx, dfg::DfgDialect* dialect) {
        ProcessOp::attachInterface<ProcessLikeOpInterface<ProcessOp>>(*ctx);
        RegionOp::attachInterface<ProcessLikeOpInterface<RegionOp>>(*ctx);
        PullOp::attachInterface<PullOpInterface>(*ctx);
        PushOp::attachInterface<PushOpInterface>(*ctx);
        ChannelOp::attachInterface<ChannelOpInterface>(*ctx);
    });
}
