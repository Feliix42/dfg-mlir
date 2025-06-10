/// Implementation of FuncToDfg pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/FuncToDfg/FuncToDfg.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>
#include <ranges>
#include <string>

#define DEBUG_TYPE "func-to-dfg"

namespace mlir {
#define GEN_PASS_DEF_CONVERTFUNCTODFG
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

func::FuncOp regionFunc;
std::map<std::string, unsigned> numResultUse;
std::map<std::string, OperatorOp> nodeMap;
namespace {
struct ConvertFuncToOperator : OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    ConvertFuncToOperator(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<func::FuncOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        func::FuncOp op,
        func::FuncOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto funcTy = op.getFunctionType();
        auto funcName = op.getSymName();

        // If it's a node
        if (op != regionFunc) {
            IRMapping mapper;
            SmallVector<Type> outputs;
            for (unsigned i = 0; i < numResultUse[funcName.str()]; ++i)
                outputs.push_back(funcTy.getResult(0));
            auto newFuncTy =
                rewriter.getFunctionType(funcTy.getInputs(), outputs);
            auto operatorOp =
                rewriter.create<OperatorOp>(loc, funcName, newFuncTy);
            LLVM_DEBUG(
                llvm::dbgs() << "//=== Creating operator " << funcName << "\n");
            rewriter.setInsertionPointToEnd(&operatorOp.getBody().front());
            for (auto [oldArg, newArg] : llvm::zip(
                     op.getArguments(),
                     operatorOp.getBody().getArguments())) {
                mapper.map(oldArg, newArg);
            }
            for (auto &opi : op.getBody().getOps()) {
                LLVM_DEBUG(
                    llvm::dbgs() << "//=== Cloning " << opi.getName() << "\n");
                rewriter.clone(opi, mapper);
            }
            nodeMap[funcName.str()] = operatorOp;
        }
        // Or it's a region
        else {
            DenseMap<Value, SmallVector<Value>> mapper;
            // Get the new functional type
            SmallVector<Type> inTypes, outTypes;
            for (auto inTy : funcTy.getInputs())
                inTypes.push_back(OutputType::get(rewriter.getContext(), inTy));
            for (auto outTy : funcTy.getResults())
                outTypes.push_back(
                    InputType::get(rewriter.getContext(), outTy));
            auto newFuncTy = rewriter.getFunctionType(inTypes, outTypes);
            // Create the region
            auto regionOp = rewriter.create<RegionOp>(loc, funcName, newFuncTy);
            LLVM_DEBUG(
                llvm::dbgs() << "//=== Creating region " << funcName << "\n");
            auto regionLoc = regionOp.getLoc();
            Block* regionBlock = &regionOp.getBody().front();
            rewriter.setInsertionPointToEnd(regionBlock);
            // Create channels
            SmallVector<ChannelOp> channels;
            for (auto [i, inTy] : llvm::enumerate(funcTy.getInputs())) {
                // (TODO) For now, set buffer size as 2
                auto channelOp = rewriter.create<ChannelOp>(regionLoc, inTy, 2);
                LLVM_DEBUG(
                    llvm::dbgs()
                    << "//=== Creating input channel " << i << "\n");
                channels.push_back(channelOp);
                mapper[op.getArgument(i)] = {channelOp.getOutChan()};
            }
            // Connect channel's IO to region's, now there are only input/output
            // channels in the vector
            for (unsigned i = 0; i < funcTy.getNumInputs(); ++i) {
                rewriter.create<ConnectInputOp>(
                    regionLoc,
                    regionBlock->getArgument(i),
                    channels[i].getInChan());
                LLVM_DEBUG(
                    llvm::dbgs()
                    << "//=== Connecting input channel " << i << "\n");
            }
            // Iteratively process the contents of the original func
            Operation* connectInsertionPoint;
            for (auto [i, opi] : llvm::enumerate(op.getBody().front())) {
                if (auto returnOp = dyn_cast<func::ReturnOp>(opi)) break;
                auto callOp = cast<func::CallOp>(opi);
                auto callResult = callOp.getResult(0);
                auto operatorOp = nodeMap[callOp.getCallee().str()];
                // Channels that provide inputs must be created already, create
                // the one to store the output now
                rewriter.setInsertionPointAfter(channels.back());
                SmallVector<Value> outputs, mapOutputs;
                for (auto [idx, type] :
                     llvm::enumerate(operatorOp.getOutputPortTypes())) {
                    auto channelOp =
                        rewriter.create<ChannelOp>(regionLoc, type, 2);
                    LLVM_DEBUG(
                        llvm::dbgs()
                        << "//=== Creating channel for output " << idx
                        << " of operator " << operatorOp.getNodeName() << "\n");
                    channels.push_back(channelOp);
                    outputs.push_back(channelOp.getInChan());
                    mapOutputs.push_back(channelOp.getOutChan());
                }
                // Now assume all call op only generate one result
                mapper[callResult] = mapOutputs;
                rewriter.setInsertionPointToEnd(regionBlock);
                SmallVector<Value> inputs;
                // Get the inputs to be used in instance
                for (auto callValue : callOp.getOperands()) {
                    auto &valueVec = mapper[callValue];
                    auto input = valueVec.front();
                    valueVec.erase(valueVec.begin());
                    inputs.push_back(input);
                }
                auto instance = rewriter.create<InstantiateOp>(
                    regionLoc,
                    operatorOp.getSymName().str(),
                    inputs,
                    outputs,
                    false);
                LLVM_DEBUG(
                    llvm::dbgs() << "//=== Createing instance of operator "
                                 << operatorOp.getSymName() << "\n");
                if (i == 0) connectInsertionPoint = instance;
                for (auto user : callResult.getUsers()) {
                    if (auto returnOp = dyn_cast<func::ReturnOp>(user)) {
                        rewriter.setInsertionPoint(connectInsertionPoint);
                        for (auto [idx, value] :
                             llvm::enumerate(returnOp.getOperands())) {
                            if (value == callResult) {
                                rewriter.create<ConnectOutputOp>(
                                    regionLoc,
                                    regionBlock->getArgument(
                                        regionOp.getNumInputPorts() + idx),
                                    mapper[value].front());
                                LLVM_DEBUG(
                                    llvm::dbgs()
                                    << "//=== Connecting output channel " << idx
                                    << "\n");
                            }
                        }
                    }
                }
            }
        }
        LLVM_DEBUG(llvm::dbgs() << "\n");

        rewriter.eraseOp(op);
        return success();
    }
};
struct ConvertReturnToOutput : OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    ConvertReturnToOutput(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<func::ReturnOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        func::ReturnOp op,
        func::ReturnOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto operatorOp = op->getParentOfType<OperatorOp>();

        auto result = op.getOperand(0);
        SmallVector<Value> results;
        for (unsigned i = 0; i < operatorOp.getNumOutputPorts(); ++i)
            results.push_back(result);
        auto outputOp = rewriter.create<OutputOp>(loc, results);

        rewriter.replaceOp(op, outputOp);
        return success();
    }
};
} // namespace

void mlir::populateFuncToDfgConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertFuncToOperator>(typeConverter, patterns.getContext());
    patterns.add<ConvertReturnToOutput>(typeConverter, patterns.getContext());
}

// TODO: To systolic array
namespace {
struct ConvertFuncToDfgPass
        : public impl::ConvertFuncToDfgBase<ConvertFuncToDfgPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertFuncToDfgPass::runOnOperation()
{
    // Assume the last func is the top function, within which there are calls to
    // nodes.
    auto module = dyn_cast<ModuleOp>(getOperation());
    Operation* lastOp;
    module.walk([&](func::FuncOp funcOp) { lastOp = funcOp; });
    if (!isa<func::FuncOp>(lastOp)) {
        lastOp->emitError("Expected func.func at the end of module.\n");
        signalPassFailure();
    }
    regionFunc = cast<func::FuncOp>(lastOp);
    LLVM_DEBUG(
        llvm::dbgs()
        << "//=== Found region functions:" << regionFunc.getSymName() << "\n");
    // Check how many times the result of a function is used
    for (auto &opi : regionFunc.getBody().front()) {
        if (isa<func::ReturnOp>(opi)) break;
        if (!isa<func::CallOp>(opi)) {
            opi.emitError("Expected only func.call in the top function");
            signalPassFailure();
        }
        auto callOp = cast<func::CallOp>(opi);
        auto numUse = std::ranges::distance(callOp.getResult(0).getUses());
        numResultUse[callOp.getCallee().str()] = numUse;
        LLVM_DEBUG(
            llvm::dbgs() << "//===" << callOp.getCallee() << " should have "
                         << numUse << " results\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Pass configuration
    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateFuncToDfgConversionPatterns(converter, patterns);

    target.addLegalDialect<DfgDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertFuncToDfgPass()
{
    return std::make_unique<ConvertFuncToDfgPass>();
}
