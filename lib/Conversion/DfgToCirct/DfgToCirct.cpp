/// Implementation of DfgToCirct pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToCirct/DfgToCirct.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOCIRCT
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

ChannelStyle channelStyle = ChannelStyle::Dfg;
struct EraseSetChannelStyle : OpConversionPattern<SetChannelStyleOp> {
public:
    using OpConversionPattern<SetChannelStyleOp>::OpConversionPattern;

    EraseSetChannelStyle(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<SetChannelStyleOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        SetChannelStyleOp op,
        SetChannelStyleOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        channelStyle = op.getChannelStyle();
        rewriter.eraseOp(op);
        return success();
    }
};

// Helper to determine which new argument to use
SmallVector<std::pair<int, Value>> oldArgsIndex;
SmallVector<std::pair<Value, Value>> newArguments;
SmallVector<std::pair<Operation*, StringAttr>> newProcesses;

fsm::MachineOp insertController(
    ModuleOp module,
    std::string name,
    int numPullChan,
    int numPushChan,
    size_t numPull,
    size_t numPush,
    Value lastMultiPulledChan,
    FunctionType funcTy,
    SmallVector<Operation*> ops,
    bool hasMultiOutputs,
    bool hasLoopOp = false,
    SmallVector<std::pair<bool, int>> loopChanArgIdx = {})
{
    auto context = module.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto loc = builder.getUnknownLoc();

    // Create a new fsm.machine
    auto machine = builder.create<fsm::MachineOp>(loc, name, "READ0", funcTy);
    builder.setInsertionPointToStart(&machine.getBody().front());

    // Create constants and variables
    auto i1Ty = builder.getI1Type();
    auto c_true = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto c_false = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    // size_t numPull = 0;
    // size_t numPush = 0;
    size_t numHWInstanceResults = 0;
    SmallVector<Value> hwInstanceValids;
    SmallVector<Value> pullVars;
    SmallVector<Value> closeRegs;
    SmallVector<Value> closeValues;
    SmallVector<Value> hasClosePull;
    SmallVector<std::pair<size_t, Value>> closeRegIdices;
    SmallVector<Value> calcResults;
    SmallVector<std::pair<Value, int>> zeroWidth;
    bool hasHwInstance =
        (funcTy.getNumResults() > size_t(numPullChan + 2 * numPushChan + 1));
    // int numCalculation = 0;
    // Now assume work pipeline: pull -> calc -> push
    for (size_t i = 0; i < ops.size(); i++) {
        auto op = ops[i];
        if (auto pullOp = dyn_cast<PullOp>(op)) {
            assert(i <= numPull && "must pull every data at beginning");
            if (hasHwInstance) {
                auto validVarOp = builder.create<fsm::VariableOp>(
                    loc,
                    i1Ty,
                    builder.getIntegerAttr(i1Ty, 0),
                    builder.getStringAttr("valid" + std::to_string(i)));
                hwInstanceValids.push_back(validVarOp.getResult());
            }
            auto valueTy = pullOp.getChan().getType().getElementType();
            auto varOp = builder.create<fsm::VariableOp>(
                loc,
                valueTy,
                builder.getIntegerAttr(valueTy, 0),
                builder.getStringAttr("data" + std::to_string(i)));
            pullVars.push_back(varOp.getResult());
            newArguments.push_back(
                std::make_pair(varOp.getResult(), pullOp.getOutp()));
            auto argIndex =
                getNewIndexOrArg<int, Value>(pullOp.getChan(), oldArgsIndex)
                    .value();
            auto isChanLooped = loopChanArgIdx[argIndex].first;
            auto loopedArgIdx = loopChanArgIdx[argIndex].second;
            if (isChanLooped) {
                if (!isInSmallVector<Value>(pullOp.getChan(), hasClosePull)) {
                    if ((!hasHwInstance && i == numPull - 1 && numPush == 1)
                        || pullOp.getChan() == lastMultiPulledChan) {
                        if (!isInSmallVector<Value>(
                                machine.getArgument(loopedArgIdx),
                                closeRegs))
                            closeRegs.push_back(
                                machine.getArgument(loopedArgIdx));
                        closeValues.push_back(
                            machine.getArgument(loopedArgIdx));
                        continue;
                    }
                    auto closeVarOp = builder.create<fsm::VariableOp>(
                        loc,
                        i1Ty,
                        builder.getIntegerAttr(i1Ty, 0),
                        builder.getStringAttr("close" + std::to_string(i)));
                    closeRegs.push_back(closeVarOp.getResult());
                    closeValues.push_back(closeVarOp.getResult());
                    hasClosePull.push_back(pullOp.getChan());
                    closeRegIdices.push_back(
                        std::make_pair(i, pullOp.getChan()));
                } else {
                    closeValues.push_back(
                        closeValues[getNewIndexOrArg<size_t, Value>(
                                        pullOp.getChan(),
                                        closeRegIdices)
                                        .value()]);
                }
            } else {
                closeValues.push_back(nullptr);
            }
            // numPull++;
        } else if (auto pushOp = dyn_cast<PushOp>(op)) {
            // numPush++;
            continue;
        } else if (auto hwInstanceOp = dyn_cast<HWInstanceOp>(op)) {
            auto types = hwInstanceOp.getResultTypes();
            numHWInstanceResults = types.size();
            for (size_t i = 0; i < numHWInstanceResults; i++) {
                auto type = types[i];
                auto varName = "result" + std::to_string(i);
                auto varOp = builder.create<fsm::VariableOp>(
                    loc,
                    type,
                    builder.getIntegerAttr(type, 0),
                    builder.getStringAttr(varName));
                calcResults.push_back(varOp.getResult());
                newArguments.push_back(std::make_pair(
                    varOp.getResult(),
                    hwInstanceOp.getResult(i)));
            }
        } else {
            return fsm::MachineOp{};
        }
    }
    SmallVector<Value> calcDataValids;
    if (hasMultiOutputs) {
        for (size_t i = 0; i < numHWInstanceResults; i++) {
            auto validVarOp = builder.create<fsm::VariableOp>(
                loc,
                i1Ty,
                builder.getIntegerAttr(i1Ty, 0),
                builder.getStringAttr("calc_valid" + std::to_string(i)));
            calcDataValids.push_back(validVarOp.getResult());
        }
    }

    // Caculate if the machine should be shut down
    Value shouldClose;
    comb::XorOp notClose;
    if (hasLoopOp) {
        for (size_t i = 0; i < closeRegs.size(); i++) {
            auto closeReg = closeRegs[i];
            // if (closeReg != nullptr) {
            if (i == 0)
                shouldClose = closeReg;
            else {
                auto closeResult =
                    builder.create<comb::AndOp>(loc, shouldClose, closeReg);
                shouldClose = closeResult.getResult();
            }
            // }
        }
        notClose =
            builder.create<comb::XorOp>(loc, shouldClose, c_true.getResult());
    }

    // All zero vector
    std::vector<Value> outputAllZero;
    for (const auto out : machine.getFunctionType().getResults()) {
        auto outWidth = dyn_cast<IntegerType>(out).getWidth();
        if (outWidth == 1)
            outputAllZero.push_back(c_false.getResult());
        else if (
            const auto zeroExist =
                getNewIndexOrArg<Value, int>(outWidth, zeroWidth)) {
            outputAllZero.push_back(zeroExist.value());
        } else {
            auto zero = builder.create<hw::ConstantOp>(
                loc,
                builder.getIntegerType(outWidth),
                0);
            zeroWidth.push_back(std::make_pair(zero.getResult(), outWidth));
            outputAllZero.push_back(zero.getResult());
        }
    }
    auto idxBias = numPullChan + 2 * numPushChan + 1;
    // Init output vector
    std::vector<Value> outputInit = outputAllZero;
    if (hasHwInstance) outputInit[idxBias++] = c_true.getResult();
    // Output vectors, need to be modified
    std::vector<Value> outputTempVec = outputAllZero;
    if (hasHwInstance)
        for (size_t i = 0; i < numPull; i++) {
            outputTempVec[idxBias + 2 * i] = hwInstanceValids[i];
            outputTempVec[idxBias + 2 * i + 1] = pullVars[i];
        }

    // All pulls are at beginning of an process
    // For every PullOp there is one READ state
    assert(ops.size() > numPull && "cannot only pull");
    auto isNextPush = isa<PushOp>(ops[numPull]);
    for (size_t i = 0; i < pullVars.size(); i++) {
        auto pullOp = dyn_cast<PullOp>(ops[i]);
        auto argIndex =
            getNewIndexOrArg<int, Value>(pullOp.getChan(), oldArgsIndex)
                .value();
        auto isChanLooped = loopChanArgIdx[argIndex].first;
        auto loopedArgIdx = loopChanArgIdx[argIndex].second;
        auto validIdx = isChanLooped ? loopedArgIdx - 2 : loopedArgIdx - 1;

        std::vector<Value> newOutputs =
            hasHwInstance ? outputInit : outputTempVec;
        newOutputs[argIndex] = machine.getArgument(validIdx);
        auto stateRead = builder.create<fsm::StateOp>(
            loc,
            "READ" + std::to_string(i),
            newOutputs);
        builder.setInsertionPointToEnd(&stateRead.getTransitions().back());

        if (i == numPull - 1) {
            if (!isNextPush || (isNextPush && numPush > 1)) {
                builder.create<fsm::TransitionOp>(
                    loc,
                    (i == pullVars.size() - 1)
                        ? (isNextPush ? "WRITE0" : "CALC")
                        : "READ" + std::to_string(i + 1),
                    /*guard*/
                    [&] {
                        builder.create<fsm::ReturnOp>(
                            loc,
                            machine.getArgument(validIdx));
                    },
                    /*action*/
                    [&] {
                        if (hasHwInstance)
                            builder.create<fsm::UpdateOp>(
                                loc,
                                hwInstanceValids[i],
                                c_true.getResult());
                        builder.create<fsm::UpdateOp>(
                            loc,
                            pullVars[i],
                            machine.getArgument(validIdx + 1));
                        if (isChanLooped
                            && pullOp.getChan() != lastMultiPulledChan)
                            builder.create<fsm::UpdateOp>(
                                loc,
                                closeValues[i],
                                machine.getArgument(loopedArgIdx));
                    });
            } else if (isChanLooped) {
                // If there is only one push and the process is looped, add two
                // transitions
                // 1. If there is no close signal, go to normal WRITE state,
                // which will go to init state to restart the computation
                builder.create<fsm::TransitionOp>(
                    loc,
                    "WRITE0",
                    /*guard*/
                    [&] {
                        auto guardWrite = builder.create<comb::AndOp>(
                            loc,
                            machine.getArgument(validIdx),
                            notClose.getResult());
                        builder.create<fsm::ReturnOp>(
                            loc,
                            guardWrite.getResult());
                    });
                builder.setInsertionPointToEnd(
                    &stateRead.getTransitions().back());
                // 2. If there is close signal, got to WRITE_CLOSE state, which
                // will write out and assert the done signal at same time
                builder.create<fsm::TransitionOp>(
                    loc,
                    "WRITE_CLOSE",
                    /*guard*/
                    [&] {
                        auto closeQueue = builder.create<comb::AndOp>(
                            loc,
                            machine.getArgument(validIdx),
                            shouldClose);
                        builder.create<fsm::ReturnOp>(
                            loc,
                            closeQueue.getResult());
                    });
            } else if (!isChanLooped && isNextPush && numPush == 1) {
                builder.create<fsm::TransitionOp>(
                    loc,
                    "WRITE_CLOSE",
                    /*guard*/ [&] {
                        builder.create<fsm::ReturnOp>(
                            loc,
                            machine.getArgument(validIdx));
                    });
            }
        } else {
            builder.create<fsm::TransitionOp>(
                loc,
                "READ" + std::to_string(i + 1),
                /*guard*/
                [&] {
                    builder.create<fsm::ReturnOp>(
                        loc,
                        machine.getArgument(validIdx));
                },
                /*action*/
                [&] {
                    if (hasHwInstance)
                        builder.create<fsm::UpdateOp>(
                            loc,
                            hwInstanceValids[i],
                            c_true.getResult());
                    builder.create<fsm::UpdateOp>(
                        loc,
                        pullVars[i],
                        machine.getArgument(validIdx + 1));
                    if (isChanLooped && pullOp.getChan() != lastMultiPulledChan)
                        builder.create<fsm::UpdateOp>(
                            loc,
                            closeValues[i],
                            machine.getArgument(loopedArgIdx));
                });
        }
        builder.setInsertionPointToEnd(&machine.getBody().back());
    }

    if (!isNextPush) {
        // Wait until HwInstance's op are done
        auto stateCalc =
            builder.create<fsm::StateOp>(loc, "CALC", outputTempVec);
        builder.setInsertionPointToEnd(&stateCalc.getTransitions().back());
        fsm::TransitionOp transWrite;
        if (hasLoopOp || (!hasLoopOp && numPush != 1))
            transWrite = builder.create<fsm::TransitionOp>(loc, "WRITE0");
        else if (!hasLoopOp && numPush == 1)
            transWrite = builder.create<fsm::TransitionOp>(loc, "WRITE_CLOSE");
        builder.setInsertionPointToEnd(transWrite.ensureGuard(builder));
        transWrite.getGuard().front().front().erase();
        idxBias = loopChanArgIdx.back().second + 1;
        auto numCalcDataValid = calcDataValids.size();
        // If wrapped calc func has more than one output, need to wait all
        // output ports are ready to transit to next state
        Value returnValue;
        if (hasMultiOutputs) {
            Value calcDone;
            for (size_t i = 0; i < numCalcDataValid; i++) {
                auto validSignal = calcDataValids[i];
                if (i == 0)
                    calcDone = validSignal;
                else {
                    auto validAndResult =
                        builder.create<comb::AndOp>(loc, calcDone, validSignal);
                    calcDone = validAndResult.getResult();
                }
            }
            if (numPush == 1 && hasLoopOp) {
                returnValue = builder
                                  .create<comb::AndOp>(
                                      loc,
                                      calcDone,
                                      notClose.getResult())
                                  .getResult();
            } else {
                returnValue = calcDone;
            }
        } else if (numPush == 1 && hasLoopOp) {
            returnValue = builder
                              .create<comb::AndOp>(
                                  loc,
                                  machine.getArgument(idxBias),
                                  notClose.getResult())
                              .getResult();
        } else {
            returnValue = machine.getArgument(idxBias);
        }
        builder.create<fsm::ReturnOp>(loc, returnValue);
        // If only one output, update the variable with calc result
        if (!hasMultiOutputs) {
            builder.setInsertionPointToEnd(transWrite.ensureAction(builder));
            builder.create<fsm::UpdateOp>(
                loc,
                calcResults[0],
                machine.getArgument(idxBias + 1));
            builder.setInsertionPointToEnd(&machine.getBody().back());
        }
        // If there is only one push as following, add one transition to
        // write_close state
        if (numPush == 1 && hasLoopOp) {
            builder.setInsertionPointToEnd(&stateCalc.getTransitions().back());
            auto transWriteClose =
                builder.create<fsm::TransitionOp>(loc, "WRITE_CLOSE");
            builder.setInsertionPointToEnd(
                transWriteClose.ensureGuard(builder));
            transWriteClose.getGuard().front().front().erase();
            if (hasMultiOutputs) {
                Value calcDone;
                for (size_t i = 0; i < numCalcDataValid; i++) {
                    auto validSignal = calcDataValids[i];
                    if (i == 0)
                        calcDone = validSignal;
                    else {
                        auto validAndResult = builder.create<comb::AndOp>(
                            loc,
                            calcDone,
                            validSignal);
                        calcDone = validAndResult.getResult();
                    }
                }
                returnValue =
                    builder.create<comb::AndOp>(loc, calcDone, shouldClose)
                        .getResult();
            } else {
                returnValue = builder
                                  .create<comb::AndOp>(
                                      loc,
                                      machine.getArgument(idxBias),
                                      shouldClose)
                                  .getResult();
            }
            builder.create<fsm::ReturnOp>(loc, returnValue);
            if (!hasMultiOutputs) {
                builder.setInsertionPointToEnd(
                    transWriteClose.ensureAction(builder));
                builder.create<fsm::UpdateOp>(
                    loc,
                    calcResults[0],
                    machine.getArgument(idxBias + 1));
            }
            builder.setInsertionPointToEnd(&machine.getBody().back());
        }
        // If more than one output, add one transition to calc state itself,
        // update the variables at the same time
        if (hasMultiOutputs) {
            builder.setInsertionPointToEnd(&stateCalc.getTransitions().back());
            builder.create<fsm::TransitionOp>(
                loc,
                "CALC",
                /*guard*/ nullptr,
                /*action*/ [&] {
                    for (size_t i = 0; i < numCalcDataValid; i++) {
                        auto calcValid = calcDataValids[i];
                        auto newResult = builder.create<comb::MuxOp>(
                            loc,
                            calcValid,
                            calcResults[i],
                            machine.getArgument(idxBias + 2 * i + 1));
                        builder.create<fsm::UpdateOp>(
                            loc,
                            calcResults[i],
                            newResult.getResult());
                        auto newValid = builder.create<comb::OrOp>(
                            loc,
                            calcValid,
                            machine.getArgument(idxBias + 2 * i));
                        builder.create<fsm::UpdateOp>(
                            loc,
                            calcValid,
                            newValid.getResult());
                    }
                });
            builder.setInsertionPointToEnd(&machine.getBody().back());
        }
    }

    // Here should be all push ops
    int idxStateWrite = 0;
    size_t startIdx = isNextPush ? numPull : numPull + 1;
    for (size_t i = startIdx; i < ops.size(); i++) {
        if (!hasLoopOp && (numPush == 1 || i == ops.size() - 1)) break;
        auto pushOp = dyn_cast<PushOp>(ops[i]);
        assert(pushOp && "here should be all PushOp");
        auto argIndex =
            getNewIndexOrArg<int, Value>(pushOp.getChan(), oldArgsIndex)
                .value();
        auto isChanLooped = loopChanArgIdx[argIndex].first;
        auto loopedArgIdx = loopChanArgIdx[argIndex].second;
        auto readyIdx = isChanLooped ? loopedArgIdx - 1 : loopedArgIdx;

        std::vector<Value> newOutputs = outputTempVec;
        newOutputs[2 * argIndex - numPullChan] = machine.getArgument(readyIdx);
        newOutputs[2 * argIndex - numPullChan + 1] =
            getNewIndexOrArg<Value, Value>(pushOp.getInp(), newArguments)
                .value();
        auto stateWrite = builder.create<fsm::StateOp>(
            loc,
            "WRITE" + std::to_string(idxStateWrite),
            newOutputs);
        builder.setInsertionPointToEnd(&stateWrite.getTransitions().back());
        if (numPush != 1 && i == ops.size() - 2) {
            if (hasLoopOp) {
                builder.create<fsm::TransitionOp>(
                    loc,
                    "WRITE" + std::to_string(idxStateWrite + 1),
                    /*guard*/ [&] {
                        auto canWriteOut = builder.create<comb::AndOp>(
                            loc,
                            machine.getArgument(readyIdx),
                            notClose.getResult());
                        builder.create<fsm::ReturnOp>(
                            loc,
                            canWriteOut.getResult());
                    });

                builder.setInsertionPointToEnd(
                    &stateWrite.getTransitions().back());
                builder.create<fsm::TransitionOp>(
                    loc,
                    "WRITE_CLOSE",
                    /*guard*/ [&] {
                        auto closeQueue = builder.create<comb::AndOp>(
                            loc,
                            machine.getArgument(readyIdx),
                            shouldClose);
                        builder.create<fsm::ReturnOp>(
                            loc,
                            closeQueue.getResult());
                    });
            } else {
                builder.create<fsm::TransitionOp>(
                    loc,
                    "WRITE_CLOSE",
                    /*guard*/ [&] {
                        builder.create<fsm::ReturnOp>(
                            loc,
                            machine.getArgument(readyIdx));
                    });
            }
        } else if (hasLoopOp && i == ops.size() - 1) {
            builder.create<fsm::TransitionOp>(
                loc,
                "READ0",
                /*guard*/
                [&] {
                    builder.create<fsm::ReturnOp>(
                        loc,
                        machine.getArgument(readyIdx));
                },
                /*action*/
                [&] {
                    for (size_t i = 0; i < numPull; i++) {
                        if (hasHwInstance)
                            builder.create<fsm::UpdateOp>(
                                loc,
                                hwInstanceValids[i],
                                c_false.getResult());
                        auto pullVarWidth =
                            pullVars[i].getType().getIntOrFloatBitWidth();
                        Value zeroValue;
                        if (pullVarWidth == 1)
                            zeroValue = c_false.getResult();
                        else
                            zeroValue = getNewIndexOrArg<Value, int>(
                                            pullVarWidth,
                                            zeroWidth)
                                            .value();
                        builder.create<fsm::UpdateOp>(
                            loc,
                            pullVars[i],
                            zeroValue);
                    }
                    if (hasMultiOutputs) {
                        for (size_t i = 0; i < calcDataValids.size(); i++) {
                            builder.create<fsm::UpdateOp>(
                                loc,
                                calcDataValids[i],
                                c_false.getResult());
                        }
                    }
                    for (auto closeReg : closeRegs) {
                        if (closeReg && !isa<BlockArgument>(closeReg)) {
                            builder.create<fsm::UpdateOp>(
                                loc,
                                closeReg,
                                c_false.getResult());
                        }
                    }
                });
        } else {
            builder.create<fsm::TransitionOp>(
                loc,
                "WRITE" + std::to_string(idxStateWrite + 1),
                /*guard*/ [&] {
                    builder.create<fsm::ReturnOp>(
                        loc,
                        machine.getArgument(readyIdx));
                });
        }
        builder.setInsertionPointToEnd(&machine.getBody().back());
        idxStateWrite++;
    }

    // WRITE_CLOSE state to output the last data along with the done signal,
    // this is similar to the tlast signal behaviour in Xilinx modules
    std::vector<Value> newOutputs = outputTempVec;
    auto lastPushOp = dyn_cast<PushOp>(ops.back());
    auto argIndex =
        getNewIndexOrArg<int, Value>(lastPushOp.getChan(), oldArgsIndex)
            .value();
    auto isChanLooped = loopChanArgIdx[argIndex].first;
    auto loopedArgIdx = loopChanArgIdx[argIndex].second;
    auto readyIdx = isChanLooped ? loopedArgIdx - 1 : loopedArgIdx;
    newOutputs[2 * argIndex - numPullChan] = machine.getArgument(readyIdx);
    newOutputs[2 * argIndex - numPullChan + 1] =
        getNewIndexOrArg<Value, Value>(lastPushOp.getInp(), newArguments)
            .value();
    newOutputs[numPullChan + 2 * numPushChan] = c_true.getResult();
    builder.create<fsm::StateOp>(loc, "WRITE_CLOSE", newOutputs);
    builder.setInsertionPointToEnd(&machine.getBody().back());

    return machine;
}

SmallVector<std::pair<bool, std::string>> processHasLoop;
SmallVector<std::pair<SmallVector<int>, std::string>> processPortIdx;
struct ConvertProcessToHWModule : OpConversionPattern<ProcessOp> {
public:
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ConvertProcessToHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto opName = op.getSymName().str();
        auto funcTy = op.getFunctionType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumResults();

        auto args = op.getBody().getArguments();
        auto types = op.getBody().getArgumentTypes();
        size_t size = types.size();

        // If there is a LoopOp, get ops inside and log the index of channel
        auto opsInBody = op.getBody().getOps();
        bool hasLoopOp = false;
        SmallVector<int> loopChanIdx;
        if (auto loopOp = dyn_cast<LoopOp>(*opsInBody.begin())) {
            if (!loopOp.getOutChans().empty())
                return rewriter.notifyMatchFailure(
                    loopOp.getLoc(),
                    "Not supported closing on output ports yet!");
            opsInBody = loopOp.getBody().getOps();
            if (!loopOp.getInChans().empty()) {
                hasLoopOp = true;
                for (auto inChan : loopOp.getInChans()) {
                    auto idxChan = cast<BlockArgument>(inChan).getArgNumber();
                    loopChanIdx.push_back(idxChan);
                }
            }
        }
        processHasLoop.push_back(std::make_pair(hasLoopOp, opName));
        processPortIdx.push_back(std::make_pair(loopChanIdx, opName));

        auto i1Ty = rewriter.getI1Type();
        auto inDir = hw::ModulePort::Direction::Input;
        auto outDir = hw::ModulePort::Direction::Output;
        SmallVector<hw::PortInfo> ports;
        int in_num = 1;
        int out_num = 1;

        // Add clock port.
        ports.push_back(hw::PortInfo{
            {rewriter.getStringAttr("clock"), i1Ty, inDir}
        });

        // Add reset port.
        ports.push_back(hw::PortInfo{
            {rewriter.getStringAttr("reset"), i1Ty, inDir}
        });

        for (size_t i = 0; i < size; i++) {
            const auto type = types[i];
            std::string name;

            bool isMonitored = false;
            if (hasLoopOp) {
                if (llvm::find(loopChanIdx, i) != loopChanIdx.end())
                    isMonitored = true;
            }

            if (const auto inTy = dyn_cast<OutputType>(type)) {
                auto elemTy = dyn_cast<IntegerType>(inTy.getElementType());
                assert(elemTy && "only integers are supported on hardware");
                auto namePrefix = "in" + std::to_string(in_num) + "_";
                // Add ready for input port
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "ready"),
                     i1Ty, outDir}
                });
                // Add valid for input port
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "valid"),
                     i1Ty, inDir}
                });
                // Add bits for input port
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "bits"),
                     rewriter.getIntegerType(elemTy.getWidth()),
                     inDir}
                });
                // Add close for input port if it's looped
                if (isMonitored) {
                    ports.push_back(hw::PortInfo{
                        {rewriter.getStringAttr(namePrefix + "close"),
                         i1Ty, inDir}
                    });
                }
                // Index increment
                in_num++;
            } else if (const auto outTy = dyn_cast<InputType>(type)) {
                auto elemTy = dyn_cast<IntegerType>(outTy.getElementType());
                assert(elemTy && "only integers are supported on hardware");
                auto namePrefix = "out" + std::to_string(out_num);
                // Add ready for output port
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "_ready"),
                     i1Ty, inDir}
                });
                // Add valid for output port
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "_valid"),
                     i1Ty, outDir}
                });
                // Add bits for output port
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "_bits"),
                     rewriter.getIntegerType(elemTy.getWidth()),
                     outDir}
                });
                out_num++;
            }
        }
        // Add done for output port
        ports.push_back(hw::PortInfo{
            {rewriter.getStringAttr("out_done"), i1Ty, outDir}
        });
        // Index increment
        out_num++;

        // Create new HWModule
        auto hwModule = rewriter.create<hw::HWModuleOp>(
            op.getLoc(),
            op.getSymNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        newProcesses.push_back(
            std::make_pair(hwModule.getOperation(), op.getSymNameAttr()));

        // Store the pair of old and new argument(s) in vector
        oldArgsIndex.clear();
        for (size_t i = 0; i < size; i++) {
            const auto arg = args[i];
            oldArgsIndex.push_back(std::make_pair(i, arg));
        }

        // Insert controller machine at top of module
        auto module = op->getParentOfType<ModuleOp>();
        SmallVector<Operation*> ops;
        std::string hwInstanceName;
        SmallVector<Type> hwInstanceInTypes;
        SmallVector<Type> hwInstanceOutTypes;
        for (auto &opi : opsInBody) {
            if (auto hwInstanceOp = dyn_cast<HWInstanceOp>(opi)) {
                hwInstanceName =
                    hwInstanceOp.getModuleAttr().getRootReference().str();
                auto inTypes = opi.getOperands().getTypes();
                hwInstanceInTypes.append(inTypes.begin(), inTypes.end());
                auto outTypes = opi.getResults().getTypes();
                hwInstanceOutTypes.append(outTypes.begin(), outTypes.end());
            }
            ops.push_back(&opi);
        }
        auto hwFuncTy = hwModule.getModuleType();
        auto hwInTypes = hwFuncTy.getInputTypes();
        auto hwOutTypes = hwFuncTy.getOutputTypes();
        // The machine inputs don't contain clock and reset
        rewriter.setInsertionPointToStart(&hwModule.getBody().front());
        auto loc = rewriter.getUnknownLoc();
        // auto i1Ty = rewriter.getI1Type();
        auto clock_seq = rewriter.create<seq::ToClockOp>(
            loc,
            hwModule.getBody().getArgument(0));
        SmallVector<Value> placeholderInstInputs;
        ArrayRef<Type> fsmInTypes(hwInTypes.data() + 2, hwInTypes.size() - 2);
        SmallVector<Type> fsmInTypesVec(fsmInTypes.begin(), fsmInTypes.end());
        for (auto type : hwInstanceOutTypes) {
            fsmInTypesVec.push_back(i1Ty);
            fsmInTypesVec.push_back(type);
        }
        bool hasHwInstance = hwInstanceInTypes.empty();
        SmallVector<Type> fsmOutTypesVec(hwOutTypes.begin(), hwOutTypes.end());
        if (!hasHwInstance) fsmOutTypesVec.push_back(i1Ty);
        for (auto type : hwInstanceInTypes) {
            if (!hasHwInstance) {
                auto placeholderInstanceData =
                    rewriter.create<hw::ConstantOp>(loc, type, 0);
                placeholderInstInputs.push_back(
                    placeholderInstanceData.getResult());
                auto placeholderInstanceValid =
                    rewriter.create<hw::ConstantOp>(loc, i1Ty, 0);
                placeholderInstInputs.push_back(
                    placeholderInstanceValid.getResult());
            }
            fsmOutTypesVec.push_back(i1Ty);
            fsmOutTypesVec.push_back(type);
        }
        auto placeholderCalcReset =
            rewriter.create<hw::ConstantOp>(loc, i1Ty, 0);

        // Update LoopChanIdx to indices in fsm machine
        SmallVector<std::pair<bool, int>> loopChanArgIdx;
        int idxArgBias = 2;
        for (size_t i = 0; i < numInputs; i++) {
            if (llvm::find(loopChanIdx, i) != loopChanIdx.end()) {
                loopChanArgIdx.push_back(std::make_pair(true, idxArgBias));
                idxArgBias += 3;
            } else {
                loopChanArgIdx.push_back(std::make_pair(false, idxArgBias - 1));
                idxArgBias += 2;
            }
        }
        idxArgBias = loopChanArgIdx.back().second + 1;
        for (size_t i = 0; i < numOutputs; i++) {
            if (llvm::find(loopChanIdx, i + numInputs) != loopChanIdx.end()) {
                loopChanArgIdx.push_back(std::make_pair(true, idxArgBias + 1));
                idxArgBias += 2;
            } else {
                loopChanArgIdx.push_back(std::make_pair(false, idxArgBias));
                idxArgBias++;
            }
        }
        // Insert fsm.machine at top
        size_t numPull = 0;
        size_t numPush = 0;
        SmallVector<Value> pulledChan;
        op.walk([&](PullOp op) {
            numPull++;
            pulledChan.push_back(op.getChan());
        });
        op.walk([&](PushOp op) { numPush++; });
        Value lastMultiPulledChan = nullptr;
        for (size_t i = 0; i < pulledChan.size() - 1; i++)
            if (pulledChan[i] == pulledChan[numPull - 1])
                lastMultiPulledChan = pulledChan[i];
        auto newMachine = insertController(
            module,
            op.getSymName().str() + "_controller",
            numInputs,
            numOutputs,
            numPull,
            numPush,
            lastMultiPulledChan,
            FunctionType::get(
                hwFuncTy.getContext(),
                fsmInTypesVec,
                fsmOutTypesVec),
            ops,
            hwInstanceOutTypes.size() > 1,
            hasLoopOp,
            loopChanArgIdx);

        auto hwInstanceOutSize = hwInstanceOutTypes.size();
        hw::InstanceOp instanceOp;
        if (!hasHwInstance) {
            // Create a hw.instance op and take the results to fsm
            auto c_true =
                rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
            hw::HWModuleExternOp opToInstance = nullptr;
            module.walk([&](hw::HWModuleExternOp externModuleOp) {
                auto name = externModuleOp.getSymNameAttr().str();
                if (name == hwInstanceName) opToInstance = externModuleOp;
            });
            assert(opToInstance && "cannot find the module to instantiate");
            SmallVector<Value> instanceInputs;
            instanceInputs.append(
                placeholderInstInputs.begin(),
                placeholderInstInputs.end());
            instanceInputs.push_back(c_true.getResult());
            instanceInputs.push_back(hwModule.getBody().getArgument(0));
            instanceInputs.push_back(placeholderCalcReset.getResult());
            for (size_t i = 0; i < hwInstanceOutSize; i++)
                instanceInputs.push_back(c_true.getResult());
            instanceOp = rewriter.create<hw::InstanceOp>(
                loc,
                opToInstance,
                "instance",
                instanceInputs);
        }
        // Create a fsm.hw_instance op and take the results to output
        SmallVector<Value> fsmInputs;
        for (size_t i = 2; i < hwInTypes.size(); i++)
            fsmInputs.push_back(hwModule.getBody().getArgument(i));
        auto idxBias = hwInstanceInTypes.size();
        if (!hasHwInstance)
            for (size_t i = 0; i < hwInstanceOutSize; i++) {
                fsmInputs.push_back(instanceOp.getResult(idxBias + 2 * i + 1));
                fsmInputs.push_back(instanceOp.getResult(idxBias + 2 * i));
            }
        auto outputs = rewriter.create<fsm::HWInstanceOp>(
            loc,
            newMachine.getFunctionType().getResults(),
            rewriter.getStringAttr("controller"),
            newMachine.getSymNameAttr(),
            fsmInputs,
            clock_seq.getResult(),
            hwModule.getBody().getArgument(1));

        size_t resultIdxBias = numInputs + numOutputs * 2 + 1;
        SmallVector<Value> outputVec;
        for (size_t i = 0; i < resultIdxBias; i++)
            outputVec.push_back(outputs.getResult(i));
        rewriter.create<hw::OutputOp>(loc, outputVec);

        if (!hasHwInstance) {
            placeholderCalcReset.replaceAllUsesWith(
                outputs.getResult(resultIdxBias));
            for (size_t i = 0; i < placeholderInstInputs.size(); i += 2) {
                auto newBias = resultIdxBias + 1 + i;
                placeholderInstInputs[i].replaceAllUsesWith(
                    outputs.getResult(newBias + 1));
                placeholderInstInputs[i + 1].replaceAllUsesWith(
                    outputs.getResult(newBias));
            }
        }

        // Clean up the placeholders
        hwModule.walk([&](hw::ConstantOp constOp) {
            if (constOp.getResult().use_empty()) rewriter.eraseOp(constOp);
        });

        rewriter.eraseOp(op);

        return success();
    }
};

hw::HWModuleExternOp insertXilinxQueue(OpBuilder &builder, Location loc)
{
    auto name = "xpm_fifo_axis";
    auto i1Ty = builder.getI1Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;

    SmallVector<Attribute> params;
    params.push_back(
        hw::ParamDeclAttr::get("FIFO_DEPTH", builder.getI32Type()));
    params.push_back(
        hw::ParamDeclAttr::get("TDATA_WIDTH", builder.getI32Type()));
    auto dataParamAttr = hw::ParamDeclRefAttr::get(
        builder.getStringAttr("TDATA_WIDTH"),
        builder.getI32Type());
    auto dataParamTy = hw::IntType::get(dataParamAttr);

    SmallVector<hw::PortInfo> ports;
    // Add clock port.
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("s_aclk"), i1Ty, inDir}
    });
    // Add reset port.
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("s_aresetn"), i1Ty, inDir}
    });
    // Add close signal
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("s_axis_tlast"), i1Ty, inDir}
    });
    // Add io_enq_valid
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("s_axis_tvalid"), i1Ty, inDir}
    });
    // Add io_enq_bits
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("s_axis_tdata"), dataParamTy, inDir}
    });
    // Add io_deq_ready
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("m_axis_tready"), i1Ty, inDir}
    });
    // Add io_enq_ready
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("s_axis_tready"), i1Ty, outDir}
    });
    // Add io_deq_valid
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("m_axis_tvalid"), i1Ty, outDir}
    });
    // Add io_deq_bits
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("m_axis_tdata"), dataParamTy, outDir}
    });
    // Add done signal
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("m_axis_tlast"), i1Ty, outDir}
    });
    // Create new HWModule
    auto hwModule = builder.create<hw::HWModuleExternOp>(
        loc,
        builder.getStringAttr(name),
        hw::ModulePortInfo(ports),
        StringRef{},
        ArrayAttr::get(builder.getContext(), params));

    return hwModule;
}

hw::HWModuleOp paramQueueModule = nullptr;
hw::HWModuleExternOp xilinxQueueModule = nullptr;
hw::HWModuleOp insertQueue(OpBuilder &builder, Location loc, bool isQueueXilinx)
{
    if (isQueueXilinx && xilinxQueueModule == nullptr)
        xilinxQueueModule = insertXilinxQueue(builder, loc);
    if (paramQueueModule != nullptr) return paramQueueModule;
    auto i1Ty = builder.getI1Type();
    auto i32Ty = builder.getI32Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;
    auto name = "queue";
    SmallVector<Attribute> params;
    params.push_back(hw::ParamDeclAttr::get("bitwidth", builder.getI32Type()));
    params.push_back(hw::ParamDeclAttr::get("size", builder.getI32Type()));
    auto dataParamAttr = hw::ParamDeclRefAttr::get(
        builder.getStringAttr("bitwidth"),
        builder.getI32Type());
    auto dataParamTy = hw::IntType::get(dataParamAttr);
    auto sizeParamAttr = hw::ParamDeclRefAttr::get(
        builder.getStringAttr("size"),
        builder.getI32Type());

    SmallVector<hw::PortInfo> ports;
    // Add clock port.
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("clock"), i1Ty, inDir}
    });
    // Add reset port.
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("reset"), i1Ty, inDir}
    });
    // Add close signal
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("close"), i1Ty, inDir}
    });
    // Add io_enq_valid
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("io_enq_valid"), i1Ty, inDir}
    });
    // Add io_enq_bits
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("io_enq_bits"), dataParamTy, inDir}
    });
    // Add io_deq_ready
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("io_deq_ready"), i1Ty, inDir}
    });
    // Add io_enq_ready
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("io_enq_ready"), i1Ty, outDir}
    });
    // Add io_deq_valid
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("io_deq_valid"), i1Ty, outDir}
    });
    // Add io_deq_bits
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("io_deq_bits"), dataParamTy, outDir}
    });
    // Add done signal
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("done"), i1Ty, outDir}
    });
    // Create new HWModule
    auto hwModule = builder.create<hw::HWModuleOp>(
        loc,
        builder.getStringAttr(name),
        hw::ModulePortInfo(ports),
        ArrayAttr::get(builder.getContext(), params),
        ArrayRef<NamedAttribute>{},
        StringAttr{},
        false);
    auto clk = hwModule.getBody().getArgument(0);
    auto rst = hwModule.getBody().getArgument(1);
    auto close_signal = hwModule.getBody().getArgument(2);
    auto in_valid = hwModule.getBody().getArgument(3);
    auto in_bits = hwModule.getBody().getArgument(4);
    auto out_ready = hwModule.getBody().getArgument(5);
    builder.setInsertionPointToStart(&hwModule.getBodyRegion().front());

    if (isQueueXilinx) {
        // Contants
        SmallVector<Value> inputs;
        auto c_true = builder.create<hw::ConstantOp>(loc, i1Ty, 1).getResult();
        auto areset = builder.create<comb::XorOp>(loc, rst, c_true).getResult();
        inputs.push_back(clk);
        inputs.push_back(areset);
        inputs.push_back(close_signal);
        inputs.push_back(in_valid);
        inputs.push_back(in_bits);
        inputs.push_back(out_ready);

        SmallVector<Attribute> params;
        // auto sizeInteger = dyn_cast<IntegerAttr>(sizeParamAttr).getInt();
        params.push_back(hw::ParamDeclAttr::get("FIFO_DEPTH", sizeParamAttr));
        params.push_back(hw::ParamDeclAttr::get("TDATA_WIDTH", dataParamAttr));

        auto callXilinxQueue = builder.create<hw::InstanceOp>(
            loc,
            xilinxQueueModule,
            "xpm_fifo_axis_inst",
            inputs,
            ArrayAttr::get(builder.getContext(), params));
        builder.create<hw::OutputOp>(loc, callXilinxQueue.getResults());
    } else {
        // Constants
        auto c_true = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
        auto c_false = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
        auto c_zero = builder.create<hw::ConstantOp>(loc, i32Ty, 0);
        auto c_one = builder.create<hw::ConstantOp>(loc, i32Ty, 1);
        auto sizeValue =
            builder.create<hw::ParamValueOp>(loc, i32Ty, sizeParamAttr);
        auto sizeMaxIdx = builder.create<comb::SubOp>(
            loc,
            sizeValue.getResult(),
            c_one.getResult());

        // Close behavior
        auto want_close = builder.create<sv::RegOp>(loc, i1Ty);
        auto want_close_value =
            builder.create<sv::ReadInOutOp>(loc, want_close.getResult());
        // RAM
        auto ram = builder.create<sv::RegOp>(
            loc,
            hw::UnpackedArrayType::get(
                builder.getContext(),
                dataParamTy,
                sizeParamAttr));
        auto placeholderReadIndex =
            builder.create<hw::ConstantOp>(loc, i32Ty, 0);
        auto ram_read = builder.create<sv::ArrayIndexInOutOp>(
            loc,
            ram.getResult(),
            placeholderReadIndex.getResult());
        auto ram_read_data =
            builder.create<sv::ReadInOutOp>(loc, ram_read.getResult());

        // Pointers
        auto ptr_write = builder.create<sv::RegOp>(loc, i32Ty);
        auto ptr_write_value =
            builder.create<sv::ReadInOutOp>(loc, ptr_write.getResult());
        auto ptr_read = builder.create<sv::RegOp>(loc, i32Ty);
        auto ptr_read_value =
            builder.create<sv::ReadInOutOp>(loc, ptr_read.getResult());
        placeholderReadIndex.replaceAllUsesWith(ptr_read_value.getResult());
        auto maybe_full = builder.create<sv::RegOp>(loc, i1Ty);
        auto maybe_full_value =
            builder.create<sv::ReadInOutOp>(loc, maybe_full.getResult());
        auto ptr_last = builder.create<sv::RegOp>(loc, i32Ty);
        auto ptr_last_value =
            builder.create<sv::ReadInOutOp>(loc, ptr_last.getResult());
        auto is_done = builder.create<sv::RegOp>(loc, i1Ty);
        auto is_done_value =
            builder.create<sv::ReadInOutOp>(loc, is_done.getResult());

        // Signals
        auto ptr_match = builder.create<comb::ICmpOp>(
            loc,
            comb::ICmpPredicate::eq,
            ptr_write_value.getResult(),
            ptr_read_value.getResult());
        auto _empty_T = builder.create<comb::XorOp>(
            loc,
            maybe_full_value.getResult(),
            c_true.getResult());
        auto empty = builder.create<comb::AndOp>(
            loc,
            ptr_match.getResult(),
            _empty_T.getResult());
        auto full = builder.create<comb::AndOp>(
            loc,
            ptr_match.getResult(),
            maybe_full_value.getResult());
        auto placeholderNotFull = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
        auto do_enq = builder.create<comb::AndOp>(
            loc,
            placeholderNotFull.getResult(),
            in_valid);
        auto placeholderNotEmpty = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
        auto do_deq = builder.create<comb::AndOp>(
            loc,
            out_ready,
            placeholderNotEmpty.getResult());
        auto next_write = builder.create<comb::AddOp>(
            loc,
            ptr_write_value.getResult(),
            c_one.getResult());
        auto next_read = builder.create<comb::AddOp>(
            loc,
            ptr_read_value.getResult(),
            c_one.getResult());
        auto notSameEnqDeq = builder.create<comb::ICmpOp>(
            loc,
            comb::ICmpPredicate::ne,
            do_enq.getResult(),
            do_deq.getResult());
        auto not_empty = builder.create<comb::XorOp>(
            loc,
            empty.getResult(),
            c_true.getResult());
        placeholderNotEmpty.replaceAllUsesWith(not_empty.getResult());
        auto not_full = builder.create<comb::XorOp>(
            loc,
            full.getResult(),
            c_true.getResult());
        auto not_want_close = builder.create<comb::XorOp>(
            loc,
            want_close_value.getResult(),
            c_true.getResult());
        auto io_enq_ready_value = builder.create<comb::AndOp>(
            loc,
            not_full.getResult(),
            not_want_close.getResult());
        placeholderNotFull.replaceAllUsesWith(io_enq_ready_value.getResult());
        // auto isDone = builder.create<comb::AndOp>(
        //     loc,
        //     want_close_value.getResult(),
        //     empty.getResult());
        auto last_read = builder.create<comb::ICmpOp>(
            loc,
            comb::ICmpPredicate::eq,
            ptr_last_value.getResult(),
            ptr_read_value.getResult());
        auto last_read_want_close = builder.create<comb::AndOp>(
            loc,
            want_close_value.getResult(),
            last_read.getResult());
        auto last_read_want_close_deq = builder.create<comb::AndOp>(
            loc,
            last_read_want_close.getResult(),
            do_deq.getResult());
        auto updateClose = builder.create<comb::AndOp>(
            loc,
            close_signal,
            not_want_close.getResult());
        auto isDone = builder.create<comb::OrOp>(
            loc,
            last_read_want_close_deq.getResult(),
            is_done_value.getResult());

        // Clocked logic
        builder
            .create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
                builder.create<sv::IfOp>(
                    loc,
                    rst,
                    [&] {
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_write.getResult(),
                            c_zero.getResult());
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_read.getResult(),
                            c_zero.getResult());
                        builder.create<sv::PAssignOp>(
                            loc,
                            maybe_full.getResult(),
                            c_false.getResult());
                        builder.create<sv::PAssignOp>(
                            loc,
                            want_close.getResult(),
                            c_false.getResult());
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_last.getResult(),
                            c_zero.getResult());
                        builder.create<sv::PAssignOp>(
                            loc,
                            is_done.getResult(),
                            c_false.getResult());
                    },
                    [&] {
                        builder.create<sv::IfOp>(loc, do_enq.getResult(), [&] {
                            auto ram_write =
                                builder.create<sv::ArrayIndexInOutOp>(
                                    loc,
                                    ram.getResult(),
                                    ptr_write_value.getResult());
                            builder.create<sv::PAssignOp>(
                                loc,
                                ram_write.getResult(),
                                in_bits);
                            auto isSizeMax = builder.create<comb::ICmpOp>(
                                loc,
                                comb::ICmpPredicate::eq,
                                ptr_write_value.getResult(),
                                sizeMaxIdx.getResult());
                            builder.create<sv::IfOp>(
                                loc,
                                isSizeMax.getResult(),
                                [&] {
                                    builder.create<sv::PAssignOp>(
                                        loc,
                                        ptr_write.getResult(),
                                        c_zero.getResult());
                                },
                                [&] {
                                    builder.create<sv::PAssignOp>(
                                        loc,
                                        ptr_write.getResult(),
                                        next_write.getResult());
                                });
                        });
                        builder.create<sv::IfOp>(loc, do_deq.getResult(), [&] {
                            auto isSizeMax = builder.create<comb::ICmpOp>(
                                loc,
                                comb::ICmpPredicate::eq,
                                ptr_read_value.getResult(),
                                sizeMaxIdx.getResult());
                            builder.create<sv::IfOp>(
                                loc,
                                isSizeMax.getResult(),
                                [&] {
                                    builder.create<sv::PAssignOp>(
                                        loc,
                                        ptr_read.getResult(),
                                        c_zero.getResult());
                                },
                                [&] {
                                    builder.create<sv::PAssignOp>(
                                        loc,
                                        ptr_read.getResult(),
                                        next_read.getResult());
                                });
                        });
                        builder.create<sv::IfOp>(
                            loc,
                            notSameEnqDeq.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    maybe_full.getResult(),
                                    do_enq.getResult());
                            });
                        builder.create<sv::IfOp>(
                            loc,
                            updateClose.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    want_close.getResult(),
                                    c_true.getResult());
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_last.getResult(),
                                    ptr_write_value.getResult());
                            });
                        builder.create<sv::IfOp>(
                            loc,
                            last_read_want_close_deq.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    is_done.getResult(),
                                    c_true.getResult());
                            });
                    });
            });

        // Clean up placeholders
        placeholderReadIndex.erase();
        placeholderNotEmpty.erase();
        placeholderNotFull.erase();

        // Output
        SmallVector<Value> outputs;
        outputs.push_back(io_enq_ready_value.getResult());
        outputs.push_back(not_empty.getResult());
        outputs.push_back(ram_read_data.getResult());
        outputs.push_back(isDone.getResult());
        builder.create<hw::OutputOp>(loc, outputs);
    }
    paramQueueModule = hwModule;
    return hwModule;
}

SmallVector<std::pair<Value, Value>> newConnections;
SmallVector<std::pair<int, Value>> oldOutputs;
SmallVector<Value> newOutputs;
SmallVector<std::pair<SmallVector<Value>, Value>> newOutputBundles;
SmallVector<Value> willBeConnected;
SmallVector<std::pair<Value, Value>> placeholderInputs;
SmallVector<std::pair<SmallVector<Value>, Value>> placeholderOutputs;
SmallVector<std::pair<SmallVector<Value>, Value>> instanceInputs;
SmallVector<std::pair<Value, Value>> instanceOutputs;
std::map<std::string, int> instantiatedProcesses;
int getNameOfInstance(std::string to_find)
{
    auto it = instantiatedProcesses.find(to_find);
    if (it != instantiatedProcesses.end()) {
        auto num = it->second;
        it->second++;
        return num;
    } else {
        instantiatedProcesses.emplace(to_find, 1);
        return 0;
    }
}

struct LegalizeHWModule : OpConversionPattern<hw::HWModuleOp> {
    using OpConversionPattern<hw::HWModuleOp>::OpConversionPattern;

    LegalizeHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<hw::HWModuleOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        hw::HWModuleOp op,
        hw::HWModuleOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto context = rewriter.getContext();
        // Get connection info
        op.walk([&](HWConnectOp connect) {
            newConnections.push_back(std::make_pair(
                connect.getPortArgument(),
                connect.getPortQueue()));
        });
        // Get output port index
        op.walk([&](hw::OutputOp outputs) {
            int i = 0;
            for (auto operand : outputs.getOperands())
                oldOutputs.push_back(std::make_pair(i, operand));
        });
        // Determine the looped ports are from arguments
        SmallVector<int> needClosePortIdx;
        op.walk([&](InstantiateOp instanceOp) {
            auto calleeName = instanceOp.getCallee().getRootReference().str();
            auto inputChans = instanceOp.getInputs();
            auto loopedProcess = getNewIndexOrArg(calleeName, processHasLoop);
            if (loopedProcess.value()) {
                auto indices =
                    getNewIndexOrArg(calleeName, processPortIdx).value();
                for (auto idx : indices) {
                    auto loopedQueue =
                        dyn_cast<ChannelOp>(inputChans[idx].getDefiningOp());
                    auto isConnected = getNewIndexOrArg<Value, Value>(
                        loopedQueue.getInChan(),
                        newConnections);
                    if (isConnected) {
                        needClosePortIdx.push_back(
                            cast<BlockArgument>(isConnected.value())
                                .getArgNumber());
                    }
                }
            }
        });

        auto funcTy = op.getModuleType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumOutputs();

        SmallVector<hw::PortInfo> ports;
        int in_num = 1;
        int out_num = 1;

        // Add clock port.
        hw::PortInfo clock;
        clock.name = rewriter.getStringAttr("clock");
        clock.dir = hw::ModulePort::Direction::Input;
        clock.type = rewriter.getI1Type();
        ports.push_back(clock);

        // Add reset port.
        hw::PortInfo reset;
        reset.name = rewriter.getStringAttr("reset");
        reset.dir = hw::ModulePort::Direction::Input;
        reset.type = rewriter.getI1Type();
        ports.push_back(reset);

        for (size_t i = 0; i < numInputs; i++) {
            auto type = funcTy.getInputType(i);
            std::string name;
            auto inTy = dyn_cast<InputType>(type);
            auto elemTy = dyn_cast<IntegerType>(inTy.getElementType());
            assert(elemTy && "only integers are supported on hardware");
            // Add ready for input port
            hw::PortInfo in_ready;
            name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                   + "_ready";
            in_ready.name = rewriter.getStringAttr(name);
            in_ready.dir = hw::ModulePort::Direction::Output;
            in_ready.type = rewriter.getI1Type();
            ports.push_back(in_ready);
            // Add valid for input port
            hw::PortInfo in_valid;
            name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                   + "_valid";
            in_valid.name = rewriter.getStringAttr(name);
            in_valid.dir = hw::ModulePort::Direction::Input;
            in_valid.type = rewriter.getI1Type();
            ports.push_back(in_valid);
            // Add bits for input port
            hw::PortInfo in_bits;
            name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                   + "_bits";
            in_bits.name = rewriter.getStringAttr(name);
            in_bits.dir = hw::ModulePort::Direction::Input;
            in_bits.type = rewriter.getIntegerType(elemTy.getWidth());
            ports.push_back(in_bits);
            // Add close for input port when needed
            if (llvm::find(needClosePortIdx, i) != needClosePortIdx.end()) {
                hw::PortInfo in_close;
                name = ((numInputs == 1) ? "in" : "in" + std::to_string(in_num))
                       + "_close";
                in_close.name = rewriter.getStringAttr(name);
                in_close.dir = hw::ModulePort::Direction::Input;
                in_close.type = rewriter.getI1Type();
                ports.push_back(in_close);
            }
            // Index increment
            in_num++;
        }
        for (size_t i = 0; i < numOutputs; i++) {
            auto type = funcTy.getOutputType(i);
            std::string name;
            auto outTy = dyn_cast<OutputType>(type);
            auto elemTy = dyn_cast<IntegerType>(outTy.getElementType());
            assert(elemTy && "only integers are supported on hardware");
            // Add ready for output port
            hw::PortInfo out_ready;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_ready";
            out_ready.name = rewriter.getStringAttr(name);
            out_ready.dir = hw::ModulePort::Direction::Input;
            out_ready.type = rewriter.getI1Type();
            ports.push_back(out_ready);
            // Add valid for output port
            hw::PortInfo out_valid;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_valid";
            out_valid.name = rewriter.getStringAttr(name);
            out_valid.dir = hw::ModulePort::Direction::Output;
            out_valid.type = rewriter.getI1Type();
            ports.push_back(out_valid);
            // Add bits for output port
            hw::PortInfo out_bits;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_bits";
            out_bits.name = rewriter.getStringAttr(name);
            out_bits.dir = hw::ModulePort::Direction::Output;
            out_bits.type = rewriter.getIntegerType(elemTy.getWidth());
            ports.push_back(out_bits);
            // Add done for output port
            hw::PortInfo out_done;
            name = ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num))
                   + "_done";
            out_done.name = rewriter.getStringAttr(name);
            out_done.dir = hw::ModulePort::Direction::Output;
            out_done.type = rewriter.getI1Type();
            ports.push_back(out_done);
            // Index increment
            out_num++;
        }

        // Compute the indices for later instantiation
        SmallVector<std::pair<bool, int>> hwModulePortIdx;
        int idxArgBias = 4;
        for (size_t i = 0; i < numInputs; i++) {
            if (llvm::find(needClosePortIdx, i) != needClosePortIdx.end()) {
                hwModulePortIdx.push_back(std::make_pair(true, idxArgBias));
                idxArgBias += 3;
            } else {
                hwModulePortIdx.push_back(
                    std::make_pair(false, idxArgBias - 1));
                idxArgBias += 2;
            }
        }
        idxArgBias = hwModulePortIdx.back().second + 1;
        for (size_t i = 0; i < numOutputs; i++) {
            if (llvm::find(needClosePortIdx, i + numInputs)
                != needClosePortIdx.end()) {
                hwModulePortIdx.push_back(std::make_pair(true, idxArgBias + 1));
                idxArgBias += 2;
            } else {
                hwModulePortIdx.push_back(std::make_pair(false, idxArgBias));
                idxArgBias++;
            }
        }

        // Create new HWModule
        auto hwModule = rewriter.create<hw::HWModuleOp>(
            loc,
            op.getNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        auto moduleOp = hwModule.getParentOp<ModuleOp>();
        rewriter.setInsertionPointToStart(&hwModule.getBody().front());
        // false signal for the queues are not gonna close
        Value noClose;
        for (size_t i = 0; i < numInputs; i++) {
            if (llvm::find(needClosePortIdx, i) != needClosePortIdx.end()) {
                continue;
            } else {
                noClose =
                    rewriter
                        .create<hw::ConstantOp>(loc, rewriter.getI1Type(), 0)
                        .getResult();
                break;
            }
        }
        // Create placeholders for the port that may connected later
        op.walk([&](InstantiateOp instances) {
            for (auto inport : instances.getInputs()) {
                auto placeholder = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getI1Type(),
                    0);
                placeholderInputs.push_back(
                    std::make_pair(placeholder, inport));
            }
            for (auto outport : instances.getOutputs()) {
                if (find(willBeConnected, outport) == willBeConnected.end())
                    willBeConnected.push_back(outport);
                auto placeholderValid = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getI1Type(),
                    0);
                auto placeholderBits = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getIntegerType(
                        dyn_cast<InputType>(outport.getType())
                            .getElementType()
                            .getIntOrFloatBitWidth()),
                    0);
                auto placeholderClose = rewriter.create<hw::ConstantOp>(
                    loc,
                    rewriter.getI1Type(),
                    0);
                SmallVector<Value> placeholders;
                placeholders.push_back(placeholderValid.getResult());
                placeholders.push_back(placeholderBits.getResult());
                placeholders.push_back(placeholderClose.getResult());
                placeholderOutputs.push_back(
                    std::make_pair(placeholders, outport));
            }
        });

        // Store the pair of old and new argument(s) in vector
        oldArgsIndex.clear();
        for (size_t i = 0; i < numInputs; i++)
            oldArgsIndex.push_back(
                std::make_pair(i, op.getBody().getArgument(i)));

        // Convert ops in the old HWModule into new one
        int queueSuffixNum = 0;
        for (auto &opInside :
             llvm::make_early_inc_range(op.getBody().getOps())) {
            if (auto channelOp = dyn_cast<ChannelOp>(opInside)) {
                auto portBit =
                    channelOp.getEncapsulatedType().getIntOrFloatBitWidth();
                auto size = channelOp.getBufferSize().value();

                OpBuilder builder(context);
                builder.setInsertionPointToStart(
                    &moduleOp.getBodyRegion().front());
                auto queueModule = insertQueue(
                    builder,
                    builder.getUnknownLoc(),
                    channelStyle == ChannelStyle::Xilinx);
                auto newModuleArg = getNewIndexOrArg<Value, Value>(
                    channelOp.getInChan(),
                    newConnections);
                auto isConnectedLater =
                    find(willBeConnected, channelOp.getInChan())
                    != willBeConnected.end();
                assert(
                    (newModuleArg || isConnectedLater)
                    && "all ports must be connected");
                SmallVector<Value> inputs;
                inputs.push_back(hwModule.getBody().getArgument(0));
                inputs.push_back(hwModule.getBody().getArgument(1));
                if (!isConnectedLater) { // If connected to arguments
                    auto newArgIndex = getNewIndexOrArg<int, Value>(
                        newModuleArg.value(),
                        oldArgsIndex);
                    auto idx_input = newArgIndex.value();
                    auto needClose = hwModulePortIdx[idx_input].first;
                    auto newIdxInput = hwModulePortIdx[idx_input].second;
                    if (needClose) {
                        inputs.push_back(
                            hwModule.getBody().getArgument(newIdxInput));
                        inputs.push_back(
                            hwModule.getBody().getArgument(newIdxInput - 2));
                        inputs.push_back(
                            hwModule.getBody().getArgument(newIdxInput - 1));
                    } else {
                        inputs.push_back(noClose);
                        inputs.push_back(
                            hwModule.getBody().getArgument(newIdxInput - 1));
                        inputs.push_back(
                            hwModule.getBody().getArgument(newIdxInput));
                    }
                } else { // If connected later
                    auto placeholders =
                        getNewIndexOrArg<SmallVector<Value>, Value>(
                            channelOp.getInChan(),
                            placeholderOutputs);
                    inputs.push_back(placeholders.value()[2]);
                    inputs.push_back(placeholders.value()[0]);
                    inputs.push_back(placeholders.value()[1]);
                }
                auto oldOutputArg = getNewIndexOrArg<int, Value>(
                    channelOp.getOutChan(),
                    oldOutputs);
                if (oldOutputArg) // If connected to output
                    inputs.push_back(hwModule.getBody().getArgument(
                        hwModulePortIdx.back().second + oldOutputArg.value()));
                else { // If connected later
                    auto placeholder = getNewIndexOrArg<Value, Value>(
                        channelOp.getOutChan(),
                        placeholderInputs);
                    if (!placeholder.has_value())
                        return rewriter.notifyMatchFailure(
                            channelOp.getLoc(),
                            "Channel " + std::to_string(queueSuffixNum)
                                + " is not connected.");
                    inputs.push_back(placeholder.value());
                }
                SmallVector<Attribute> params;
                params.push_back(hw::ParamDeclAttr::get(
                    "bitwidth",
                    rewriter.getI32IntegerAttr(portBit)));
                params.push_back(hw::ParamDeclAttr::get(
                    "size",
                    rewriter.getI32IntegerAttr(size)));
                auto queueInstance = rewriter.create<hw::InstanceOp>(
                    loc,
                    queueModule,
                    "queue" + std::to_string(queueSuffixNum++),
                    inputs,
                    ArrayAttr::get(rewriter.getContext(), params));
                auto in_ready = queueInstance.getResult(0);
                auto in_valid = queueInstance.getResult(1);
                auto in_bits = queueInstance.getResult(2);
                auto in_done = queueInstance.getResult(3);
                if (newModuleArg) newOutputs.push_back(in_ready);
                SmallVector<Value, 3> instancePorts;
                instancePorts.push_back(in_valid);
                instancePorts.push_back(in_bits);
                instancePorts.push_back(in_done);
                instanceInputs.push_back(
                    std::make_pair(instancePorts, channelOp.getOutChan()));
                if (oldOutputArg) {
                    SmallVector<Value, 3> newBundle;
                    newBundle.push_back(in_valid);
                    newBundle.push_back(in_bits);
                    newBundle.push_back(in_done);
                    newOutputBundles.push_back(
                        std::make_pair(newBundle, channelOp.getOutChan()));
                }
                instanceOutputs.push_back(
                    std::make_pair(in_ready, channelOp.getInChan()));
            } else if (auto instantiateOp = dyn_cast<InstantiateOp>(opInside)) {
                SmallVector<Value> inputs;
                inputs.push_back(hwModule.getBody().getArgument(0));
                inputs.push_back(hwModule.getBody().getArgument(1));
                auto calleeName = instantiateOp.getCallee().getRootReference();
                auto calleeStr = calleeName.str();
                auto needClosePortIdx =
                    getNewIndexOrArg(calleeStr, processPortIdx).value();
                SmallVector<Value> instantiateOpInChans(
                    instantiateOp.getInputs().begin(),
                    instantiateOp.getInputs().end());
                for (size_t i = 0; i < instantiateOpInChans.size(); i++) {
                    auto queuePorts = getNewIndexOrArg(
                        instantiateOpInChans[i],
                        instanceInputs);
                    if (llvm::find(needClosePortIdx, i)
                        != needClosePortIdx.end()) {
                        inputs.append(queuePorts.value());
                    } else {
                        inputs.push_back(queuePorts.value()[0]);
                        inputs.push_back(queuePorts.value()[1]);
                    }
                }
                // for (auto input : instantiateOp.getInputs()) {
                //     auto queuePorts = getNewIndexOrArg(input,
                //     instanceInputs); inputs.append(queuePorts.value());
                // }
                for (auto output : instantiateOp.getOutputs()) {
                    auto queuePort = getNewIndexOrArg(output, instanceOutputs);
                    inputs.push_back(queuePort.value());
                }
                auto processToCall = getNewIndexOrArg<Operation*, StringAttr>(
                    calleeName,
                    newProcesses);
                auto nameSuffix = getNameOfInstance(calleeStr);
                auto instanceName =
                    nameSuffix == 0 ? calleeStr
                                    : calleeStr + std::to_string(nameSuffix);
                auto instanceOp = rewriter.create<hw::InstanceOp>(
                    loc,
                    processToCall.value(),
                    instanceName,
                    inputs);
                int i = 0;
                for (auto input : instantiateOp.getInputs()) {
                    auto placeholder =
                        getNewIndexOrArg(input, placeholderInputs);
                    placeholder.value().replaceAllUsesWith(
                        instanceOp.getResult(i++));
                }
                int j = 0;
                for (auto output : instantiateOp.getOutputs()) {
                    auto placeholders =
                        getNewIndexOrArg(output, placeholderOutputs);
                    placeholders.value()[0].replaceAllUsesWith(
                        instanceOp.getResult(i + j++));
                    placeholders.value()[1].replaceAllUsesWith(
                        instanceOp.getResult(i + j++));
                    placeholders.value()[2].replaceAllUsesWith(
                        instanceOp.getResults().back());
                }
            } else if (auto connectOp = dyn_cast<HWConnectOp>(opInside)) {
                continue;
            } else if (auto outputOp = dyn_cast<hw::OutputOp>(opInside)) {
                for (auto operand : outputOp.getOperands())
                    if (auto bundle =
                            getNewIndexOrArg(operand, newOutputBundles))
                        newOutputs.append(bundle.value());
                rewriter.create<hw::OutputOp>(loc, newOutputs);
            } else {
                op.emitError() << "unsupported ops in top module";
            }
        }

        // Clean up the placeholders
        hwModule.walk([&](hw::ConstantOp constOp) {
            if (constOp.getResult().use_empty()) rewriter.eraseOp(constOp);
        });

        // rewriter.eraseOp(op);
        rewriter.replaceOp(op, hwModule);

        return success();
    }
};

hw::HWModuleOp getInstancedModule(ModuleOp moduleOp, StringAttr moduleName);

SmallVector<std::pair<Value, Value>> moduleToQueueConnection;
struct ConvertRegionToHWModule : OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    ConvertRegionToHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        RegionOp op,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto context = rewriter.getContext();

        op.walk([&](ConnectInputOp connectInputOp) {
            moduleToQueueConnection.push_back(std::make_pair(
                connectInputOp.getRegionPort(),
                connectInputOp.getChannelPort()));
        });

        bool processHasLoop = false;
        // op.walk([&](InstantiateOp instantiateOp) {
        //     auto calleeName =
        //         instantiateOp.getCallee().getRootReference().str();
        //     auto inputChans = instantiateOp.getInputs();
        //     auto loopedProcess = getNewIndexOrArg(calleeName,
        //     processHasLoop); if (loopedProcess.value()) {
        //         auto indices =
        //             getNewIndexOrArg(calleeName, processPortIdx).value();
        //         for (auto idx : indices) {
        //             auto loopedQueue =
        //                 dyn_cast<ChannelOp>(inputChans[idx].getDefiningOp());
        //             auto isConnected = getNewIndexOrArg<Value, Value>(
        //                 loopedQueue.getInChan(),
        //                 newConnections);
        //             if (isConnected) {
        //                 needClosePortIdx.push_back(
        //                     cast<BlockArgument>(isConnected.value())
        //                         .getArgNumber());
        //             }
        //         }
        //     }
        // });

        // SmallVector<hw::PortInfo> ports;
        // int in_num = 1;
        // int out_num = 1;

        // // Add clock port.
        // hw::PortInfo clock;
        // // clock.name = rewriter.getStringAttr("clock");
        // // clock.dir = hw::ModulePort::Direction::Input;
        // // clock.type = rewriter.getI1Type();
        // // ports.push_back(clock);
        // ports.push_back(hw::PortInfo{
        //     rewriter.getStringAttr("clock"),
        //     rewriter.getI1Type(),
        //     hw::ModulePort::Direction::Input});
        // rewriter.create<hw::HWModuleOp>(
        //     op.getLoc(),
        //     op.getSymNameAttr(),
        //     hw::ModulePortInfo(ports),
        //     ArrayAttr{},
        //     ArrayRef<NamedAttribute>{},
        //     StringAttr{},
        //     false);

        rewriter.eraseOp(op);
        return success();
    }
};

struct InsertQueueAndInstantiate : OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    InsertQueueAndInstantiate(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};

template<typename OpT>
struct ConvertCallToInstance : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    ConvertCallToInstance(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpT>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OpT op,
        typename OpT::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};

struct EarseConnectInput : OpConversionPattern<ConnectInputOp> {
    using OpConversionPattern<ConnectInputOp>::OpConversionPattern;

    EarseConnectInput(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ConnectInputOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ConnectInputOp op,
        ConnectInputOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};

struct InsertOutputToHWModule : OpConversionPattern<ConnectOutputOp> {
    using OpConversionPattern<ConnectOutputOp>::OpConversionPattern;

    InsertOutputToHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ConnectOutputOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ConnectOutputOp op,
        ConnectOutputOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void mlir::populateDfgToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // old patterns
    patterns.add<ConvertProcessToHWModule>(
        typeConverter,
        patterns.getContext());
    patterns.add<LegalizeHWModule>(typeConverter, patterns.getContext());

    // new patterns, wait for implementations
    patterns.add<EraseSetChannelStyle>(typeConverter, patterns.getContext());
    patterns.add<ConvertRegionToHWModule>(typeConverter, patterns.getContext());
    patterns.add<InsertQueueAndInstantiate>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCallToInstance<InstantiateOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCallToInstance<EmbedOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<EarseConnectInput>(typeConverter, patterns.getContext());
    patterns.add<InsertOutputToHWModule>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDfgToCirctPass
        : public impl::ConvertDfgToCirctBase<ConvertDfgToCirctPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToCirctPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });
    // converter.addConversion([&](InputType type) -> Type {
    //     return converter.convertType(type.getElementType());
    // });
    // converter.addConversion([&](OutputType type) -> Type {
    //     return converter.convertType(type.getElementType());
    // });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<
        comb::CombDialect,
        fsm::FSMDialect,
        hw::HWDialect,
        seq::SeqDialect,
        sv::SVDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();
    target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
        auto funcTy = op.getModuleType();
        for (const auto inTy : funcTy.getInputTypes())
            if (dyn_cast<InputType>(inTy)) return false;
        for (const auto outTy : funcTy.getOutputTypes())
            if (dyn_cast<OutputType>(outTy)) return false;
        return true;
    });
    target.addDynamicallyLegalOp<hw::OutputOp>([&](hw::OutputOp op) {
        for (auto resultTy : op.getOperandTypes())
            if (dyn_cast<OutputType>(resultTy)) return false;
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToCirctPass()
{
    return std::make_unique<ConvertDfgToCirctPass>();
}
