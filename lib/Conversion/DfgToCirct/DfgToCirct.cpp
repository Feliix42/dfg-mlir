/// Implementation of DfgToCirct pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

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

// Helper to determine which new argument to use
SmallVector<std::pair<int, Value>> oldArgsIndex;
SmallVector<std::pair<Value, Value>> newArguments;
SmallVector<std::pair<Operation*, StringAttr>> newOperators;

fsm::MachineOp insertController(
    ModuleOp module,
    std::string name,
    int numPullChan,
    int numPushChan,
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
    auto machine = builder.create<fsm::MachineOp>(loc, name, "INIT", funcTy);
    builder.setInsertionPointToStart(&machine.getBody().front());

    // Create constants and variables
    auto i1Ty = builder.getI1Type();
    auto c_true = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto c_false = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    size_t numPull = 0;
    size_t numHWInstanceResults = 0;
    SmallVector<Value> hwInstanceValids;
    SmallVector<Value> pullVars;
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
                    builder.getStringAttr("valid" + std::to_string(numPull)));
                hwInstanceValids.push_back(validVarOp.getResult());
            }
            auto valueTy = pullOp.getChan().getType().getElementType();
            auto varOp = builder.create<fsm::VariableOp>(
                loc,
                valueTy,
                builder.getIntegerAttr(valueTy, 0),
                builder.getStringAttr("data" + std::to_string(numPull++)));
            pullVars.push_back(varOp.getResult());
            newArguments.push_back(
                std::make_pair(varOp.getResult(), pullOp.getOutp()));
        } else if (auto pushOp = dyn_cast<PushOp>(op)) {
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

    // Create states and transitions
    // INIT
    auto stateInit = builder.create<fsm::StateOp>(loc, "INIT");
    builder.setInsertionPointToEnd(&stateInit.getOutput().back());
    stateInit.getOutput().front().front().erase();
    builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(outputInit));
    builder.setInsertionPointToEnd(&stateInit.getTransitions().back());
    auto transRead = builder.create<fsm::TransitionOp>(loc, "READ0");
    builder.setInsertionPointToEnd(transRead.ensureAction(builder));
    for (size_t i = 0; i < numPull; i++) {
        if (hasHwInstance)
            builder.create<fsm::UpdateOp>(
                loc,
                hwInstanceValids[i],
                c_false.getResult());
        auto pullVarWidth = pullVars[i].getType().getIntOrFloatBitWidth();
        Value zeroValue;
        if (pullVarWidth == 1)
            zeroValue = c_false.getResult();
        else
            zeroValue =
                getNewIndexOrArg<Value, int>(pullVarWidth, zeroWidth).value();
        builder.create<fsm::UpdateOp>(loc, pullVars[i], zeroValue);
    }
    if (hasMultiOutputs) {
        for (size_t i = 0; i < calcDataValids.size(); i++) {
            builder.create<fsm::UpdateOp>(
                loc,
                calcDataValids[i],
                c_false.getResult());
        }
    }
    builder.setInsertionPointToEnd(&machine.getBody().back());

    // All pulls are at beginning of an operator
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

        auto stateRead =
            builder.create<fsm::StateOp>(loc, "READ" + std::to_string(i));
        builder.setInsertionPointToEnd(&stateRead.getOutput().back());
        stateRead.getOutput().front().front().erase();
        std::vector<Value> newOutputs = outputTempVec;
        newOutputs[argIndex] = machine.getArgument(validIdx);
        builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(newOutputs));
        builder.setInsertionPointToEnd(&stateRead.getTransitions().back());
        auto transCalc = builder.create<fsm::TransitionOp>(
            loc,
            (i == pullVars.size() - 1) ? (isNextPush ? "WRITE0" : "CALC")
                                       : "READ" + std::to_string(i + 1));
        builder.setInsertionPointToEnd(transCalc.ensureGuard(builder));
        transCalc.getGuard().front().front().erase();
        builder.create<fsm::ReturnOp>(loc, machine.getArgument(validIdx));
        builder.setInsertionPointToEnd(transCalc.ensureAction(builder));
        if (hasHwInstance)
            builder.create<fsm::UpdateOp>(
                loc,
                hwInstanceValids[i],
                c_true.getResult());
        builder.create<fsm::UpdateOp>(
            loc,
            pullVars[i],
            machine.getArgument(validIdx + 1));
        if (isChanLooped) {
            builder.setInsertionPointToEnd(&stateRead.getTransitions().back());
            auto transClose = builder.create<fsm::TransitionOp>(loc, "CLOSE");
            builder.setInsertionPointToEnd(transClose.ensureGuard(builder));
            transClose.getGuard().front().front().erase();
            builder.create<fsm::ReturnOp>(
                loc,
                machine.getArgument(loopedArgIdx));
        }
        builder.setInsertionPointToEnd(&machine.getBody().back());
    }

    if (!isNextPush) {
        // Wait until HwInstance's op are done
        auto stateCalc = builder.create<fsm::StateOp>(loc, "CALC");
        builder.setInsertionPointToEnd(&stateCalc.getOutput().back());
        stateCalc.getOutput().front().front().erase();
        builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(outputTempVec));
        builder.setInsertionPointToEnd(&stateCalc.getTransitions().back());
        auto transWrite = builder.create<fsm::TransitionOp>(loc, "WRITE0");
        builder.setInsertionPointToEnd(transWrite.ensureGuard(builder));
        transWrite.getGuard().front().front().erase();
        idxBias = loopChanArgIdx.back().second + 1;
        auto numCalcDataValid = calcDataValids.size();
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
            builder.create<fsm::ReturnOp>(loc, calcDone);
        } else {
            builder.create<fsm::ReturnOp>(loc, machine.getArgument(idxBias));
        }
        if (!hasMultiOutputs) {
            builder.setInsertionPointToEnd(transWrite.ensureAction(builder));
            builder.create<fsm::UpdateOp>(
                loc,
                calcResults[0],
                machine.getArgument(idxBias + 1));
            builder.setInsertionPointToEnd(&machine.getBody().back());
        }
        if (hasMultiOutputs) {
            builder.setInsertionPointToEnd(&stateCalc.getTransitions().back());
            auto transCalcSelf = builder.create<fsm::TransitionOp>(loc, "CALC");
            builder.setInsertionPointToEnd(transCalcSelf.ensureAction(builder));
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
            builder.setInsertionPointToEnd(&machine.getBody().back());
        }
    }

    // Here should be all push ops
    int idxStateWrite = 0;
    size_t startIdx = isNextPush ? numPull : numPull + 1;
    for (size_t i = startIdx; i < ops.size(); i++) {
        auto pushOp = dyn_cast<PushOp>(ops[i]);
        assert(pushOp && "here shoule be all PushOp");
        auto argIndex =
            getNewIndexOrArg<int, Value>(pushOp.getChan(), oldArgsIndex)
                .value();
        auto isChanLooped = loopChanArgIdx[argIndex].first;
        auto loopedArgIdx = loopChanArgIdx[argIndex].second;
        auto readyIdx = isChanLooped ? loopedArgIdx - 1 : loopedArgIdx;

        auto stateWrite = builder.create<fsm::StateOp>(
            loc,
            "WRITE" + std::to_string(idxStateWrite));
        builder.setInsertionPointToEnd(&stateWrite.getOutput().back());
        stateWrite.getOutput().front().front().erase();
        std::vector<Value> newOutputs = outputTempVec;
        newOutputs[2 * argIndex - numPullChan] = machine.getArgument(readyIdx);
        newOutputs[2 * argIndex - numPullChan + 1] =
            getNewIndexOrArg<Value, Value>(pushOp.getInp(), newArguments)
                .value();
        builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(newOutputs));
        builder.setInsertionPointToEnd(&stateWrite.getTransitions().back());
        auto transInit = builder.create<fsm::TransitionOp>(
            loc,
            (i == ops.size() - 1)
                ? (hasLoopOp ? "INIT" : "CLOSE")
                : "WRITE" + std::to_string(idxStateWrite + 1));
        builder.setInsertionPointToEnd(transInit.ensureGuard(builder));
        transInit.getGuard().front().front().erase();
        if (hasLoopOp && i == ops.size() - 1) {
            Value closeQueue;
            for (size_t i = 0; i < loopChanArgIdx.size(); i++) {
                auto chanIdx = loopChanArgIdx[i];
                auto hasClose = chanIdx.first;
                auto closeSignalIdx = chanIdx.second;
                if (!hasClose) continue;
                auto closeSignal = machine.getArgument(closeSignalIdx);
                if (i == 0)
                    closeQueue = closeSignal;
                else {
                    auto closeOrResult = builder.create<comb::OrOp>(
                        loc,
                        closeQueue,
                        closeSignal);
                    closeQueue = closeOrResult.getResult();
                }
            }
            auto notCloseSignal = builder.create<comb::XorOp>(
                loc,
                closeQueue,
                c_true.getResult());
            auto canWriteOut = builder.create<comb::AndOp>(
                loc,
                machine.getArgument(readyIdx),
                notCloseSignal.getResult());
            builder.create<fsm::ReturnOp>(loc, canWriteOut.getResult());
        } else {
            builder.create<fsm::ReturnOp>(loc, machine.getArgument(readyIdx));
        }
        if (hasLoopOp && i == ops.size() - 1) {
            builder.setInsertionPointToEnd(&stateWrite.getTransitions().back());
            auto transClose = builder.create<fsm::TransitionOp>(loc, "CLOSE");
            builder.setInsertionPointToEnd(transClose.ensureGuard(builder));
            transClose.getGuard().front().front().erase();
            Value closeQueue;
            for (size_t i = 0; i < loopChanArgIdx.size(); i++) {
                auto chanIdx = loopChanArgIdx[i];
                auto hasClose = chanIdx.first;
                auto closeSignalIdx = chanIdx.second;
                if (!hasClose) continue;
                auto closeSignal = machine.getArgument(closeSignalIdx);
                if (i == 0)
                    closeQueue = closeSignal;
                else {
                    auto closeOrResult = builder.create<comb::OrOp>(
                        loc,
                        closeQueue,
                        closeSignal);
                    closeQueue = closeOrResult.getResult();
                }
            }
            builder.create<fsm::ReturnOp>(loc, closeQueue);
        }
        builder.setInsertionPointToEnd(&machine.getBody().back());
        idxStateWrite++;
    }

    // CLOSE state and its behavior
    auto stateClose = builder.create<fsm::StateOp>(loc, "CLOSE");
    builder.setInsertionPointToEnd(&stateClose.getOutput().back());
    stateClose.getOutput().front().front().erase();
    std::vector<Value> newOutputs = outputTempVec;
    newOutputs[numPullChan + 2 * numPushChan] = c_true.getResult();
    builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(newOutputs));
    builder.setInsertionPointToEnd(&machine.getBody().back());

    // int idxCalc = 0;
    // int idxPush = 0;
    // isNextPush = false;
    // for (size_t i = numPull; i < ops.size(); i++) {
    //     auto op = ops[i];
    //     auto isThisPush = dyn_cast<PushOp>(op);
    //     if (i < ops.size() - 1) isNextPush = isa<PushOp>(ops[i + 1]);

    //     // For every push create one STATE: WRITE
    //     if (isThisPush) {
    //         auto argIndex =
    //             getNewIndexOrArg<int, Value>(isThisPush.getChan(),
    //             oldArgsIndex)
    //                 .value();

    //         auto stateWrite = builder.create<fsm::StateOp>(
    //             loc,
    //             "WRITE" + std::to_string(idxPush));
    //         builder.setInsertionPointToEnd(&stateWrite.getOutput().back());
    //         stateWrite.getOutput().front().front().erase();
    //         std::vector<Value> newOutputs = outputAllZero;
    //         newOutputs[2 * argIndex - numPullChan] =
    //             machine.getArgument(argIndex + numPullChan);
    //         newOutputs[2 * argIndex - numPullChan + 1] =
    //             getNewIndexOrArg<Value, Value>(
    //                 isThisPush.getInp(),
    //                 newArguments)
    //                 .value();
    //         builder.create<fsm::OutputOp>(loc, ArrayRef<Value>(newOutputs));
    //         builder.setInsertionPointToEnd(&stateWrite.getTransitions().back());
    //         auto transWrite = builder.create<fsm::TransitionOp>(
    //             loc,
    //             (i == ops.size() - 1)
    //                 ? "INIT"
    //                 : (isNextPush ? "WRITE" + std::to_string(idxPush + 1)
    //                               : "CALC" + std::to_string(idxCalc)));
    //         builder.setInsertionPointToEnd(transWrite.ensureGuard(builder));
    //         transWrite.getGuard().front().front().erase();
    //         builder.create<fsm::ReturnOp>(
    //             loc,
    //             machine.getArgument(argIndex + numPullChan));
    //         builder.setInsertionPointToEnd(&machine.getBody().back());
    //         idxPush++;
    //     }
    //     // For every computation create one STATE: CALC
    //     else {
    //         auto stateCal = builder.create<fsm::StateOp>(
    //             loc,
    //             "CALC" + std::to_string(idxCalc));
    //         builder.setInsertionPointToEnd(&stateCal.getOutput().back());
    //         stateCal.getOutput().front().front().erase();
    //         builder.create<fsm::OutputOp>(loc,
    //         ArrayRef<Value>(outputAllZero));
    //         builder.setInsertionPointToEnd(&stateCal.getTransitions().back());
    //         auto transCal = builder.create<fsm::TransitionOp>(
    //             loc,
    //             (i == ops.size() - 1)
    //                 ? "INIT"
    //                 : (isNextPush ? "WRITE" + std::to_string(idxPush)
    //                               : "CALC" + std::to_string(idxCalc + 1)));
    //         builder.setInsertionPointToEnd(transCal.ensureAction(builder));
    //         auto newCalcOp = op->clone();
    //         for (size_t j = 0; j < newCalcOp->getNumOperands(); j++) {
    //             auto newOperand = getNewIndexOrArg<Value, Value>(
    //                                   newCalcOp->getOperand(j),
    //                                   newArguments)
    //                                   .value();
    //             newCalcOp->setOperand(j, newOperand);
    //         }
    //         builder.insert(newCalcOp);
    //         builder.create<fsm::UpdateOp>(
    //             loc,
    //             calcResults[idxCalc],
    //             newCalcOp->getResult(0));
    //         builder.setInsertionPointToEnd(&machine.getBody().back());
    //         idxCalc++;
    //     }
    // }

    return machine;
}

SmallVector<std::pair<bool, std::string>> operatorHasLoop;
SmallVector<std::pair<SmallVector<int>, std::string>> operatorPortIdx;
struct ConvertOperator : OpConversionPattern<OperatorOp> {
public:
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    ConvertOperator(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp op,
        OperatorOpAdaptor adaptor,
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
                    auto idxChan = inChan.cast<BlockArgument>().getArgNumber();
                    loopChanIdx.push_back(idxChan);
                }
            }
        }
        operatorHasLoop.push_back(std::make_pair(hasLoopOp, opName));
        operatorPortIdx.push_back(std::make_pair(loopChanIdx, opName));

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
                // Add close for input port if it's monitored
                if (isMonitored) {
                    hw::PortInfo in_close;
                    name = ((numInputs == 1) ? "in"
                                             : "in" + std::to_string(in_num))
                           + "_close";
                    in_close.name = rewriter.getStringAttr(name);
                    in_close.dir = hw::ModulePort::Direction::Input;
                    in_close.type = rewriter.getI1Type();
                    ports.push_back(in_close);
                }
                // Index increment
                in_num++;
            } else if (const auto outTy = dyn_cast<InputType>(type)) {
                auto elemTy = dyn_cast<IntegerType>(outTy.getElementType());
                assert(elemTy && "only integers are supported on hardware");
                // Add ready for output port
                hw::PortInfo out_ready;
                name = ((numOutputs == 1) ? "out"
                                          : "out" + std::to_string(out_num))
                       + "_ready";
                out_ready.name = rewriter.getStringAttr(name);
                out_ready.dir = hw::ModulePort::Direction::Input;
                out_ready.type = rewriter.getI1Type();
                ports.push_back(out_ready);
                // Add valid for output port
                hw::PortInfo out_valid;
                name = ((numOutputs == 1) ? "out"
                                          : "out" + std::to_string(out_num))
                       + "_valid";
                out_valid.name = rewriter.getStringAttr(name);
                out_valid.dir = hw::ModulePort::Direction::Output;
                out_valid.type = rewriter.getI1Type();
                ports.push_back(out_valid);
                // Add bits for output port
                hw::PortInfo out_bits;
                name = ((numOutputs == 1) ? "out"
                                          : "out" + std::to_string(out_num))
                       + "_bits";
                out_bits.name = rewriter.getStringAttr(name);
                out_bits.dir = hw::ModulePort::Direction::Output;
                out_bits.type = rewriter.getIntegerType(elemTy.getWidth());
                ports.push_back(out_bits);
                out_num++;
            }
        }
        // Add done for output port
        hw::PortInfo out_done;
        std::string name = "out_done";
        out_done.name = rewriter.getStringAttr(name);
        out_done.dir = hw::ModulePort::Direction::Output;
        out_done.type = rewriter.getI1Type();
        ports.push_back(out_done);
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
        newOperators.push_back(
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
        auto hwFuncTy = hwModule.getFunctionType();
        auto hwInTypes = hwFuncTy.getInputs();
        auto hwOutTypes = hwFuncTy.getResults();
        // The machine inputs don't contain clock and reset
        rewriter.setInsertionPointToStart(&hwModule.getBody().front());
        auto loc = rewriter.getUnknownLoc();
        auto i1Ty = rewriter.getI1Type();
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
        auto newMachine = insertController(
            module,
            op.getSymName().str() + "_controller",
            numInputs,
            numOutputs,
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
            hwModule.getBody().getArgument(0),
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

std::map<std::string, hw::HWModuleOp> queueList;
hw::HWModuleOp
insertQueue(OpBuilder &builder, Location loc, unsigned portBit, unsigned size)
{
    auto suffix = std::to_string(size) + "x" + "i" + std::to_string(portBit);
    auto name = "queue_" + suffix;
    auto isExist = queueList.find(name);
    if (isExist != queueList.end()) return isExist->second;
    auto i1Ty = builder.getI1Type();

    SmallVector<hw::PortInfo> ports;
    // Add clock port.
    hw::PortInfo clock;
    clock.name = builder.getStringAttr("clock");
    clock.dir = hw::ModulePort::Direction::Input;
    clock.type = i1Ty;
    ports.push_back(clock);
    // Add reset port.
    hw::PortInfo reset;
    reset.name = builder.getStringAttr("reset");
    reset.dir = hw::ModulePort::Direction::Input;
    reset.type = i1Ty;
    ports.push_back(reset);
    // Add close signal
    hw::PortInfo close;
    close.name = builder.getStringAttr("close");
    close.dir = hw::ModulePort::Direction::Input;
    close.type = i1Ty;
    ports.push_back(close);
    // Add io_enq_valid
    hw::PortInfo io_enq_valid;
    io_enq_valid.name = builder.getStringAttr("io_enq_valid");
    io_enq_valid.dir = hw::ModulePort::Direction::Input;
    io_enq_valid.type = i1Ty;
    ports.push_back(io_enq_valid);
    // Add io_enq_bits
    hw::PortInfo io_enq_bits;
    io_enq_bits.name = builder.getStringAttr("io_enq_bits");
    io_enq_bits.dir = hw::ModulePort::Direction::Input;
    io_enq_bits.type = builder.getIntegerType(portBit);
    ports.push_back(io_enq_bits);
    // Add io_deq_ready
    hw::PortInfo io_deq_ready;
    io_deq_ready.name = builder.getStringAttr("io_deq_ready");
    io_deq_ready.dir = hw::ModulePort::Direction::Input;
    io_deq_ready.type = i1Ty;
    ports.push_back(io_deq_ready);
    // Add io_enq_ready
    hw::PortInfo io_enq_ready;
    io_enq_ready.name = builder.getStringAttr("io_enq_ready");
    io_enq_ready.dir = hw::ModulePort::Direction::Output;
    io_enq_ready.type = i1Ty;
    ports.push_back(io_enq_ready);
    // Add io_deq_valid
    hw::PortInfo io_deq_valid;
    io_deq_valid.name = builder.getStringAttr("io_deq_valid");
    io_deq_valid.dir = hw::ModulePort::Direction::Output;
    io_deq_valid.type = i1Ty;
    ports.push_back(io_deq_valid);
    // Add io_deq_bits
    hw::PortInfo io_deq_bits;
    io_deq_bits.name = builder.getStringAttr("io_deq_bits");
    io_deq_bits.dir = hw::ModulePort::Direction::Output;
    io_deq_bits.type = builder.getIntegerType(portBit);
    ports.push_back(io_deq_bits);
    // Add done signal
    hw::PortInfo done;
    done.name = builder.getStringAttr("done");
    done.dir = hw::ModulePort::Direction::Output;
    done.type = i1Ty;
    ports.push_back(done);
    // Create new HWModule
    auto hwModule = builder.create<hw::HWModuleOp>(
        loc,
        builder.getStringAttr(name),
        hw::ModulePortInfo(ports),
        ArrayAttr{},
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

    // Constants
    auto c_true = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto c_false = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    unsigned bitDepth = std::ceil(std::log2(size));
    auto ptrTy = builder.getIntegerType(bitDepth);
    auto isTwoPower = size > 0 && (size & (size - 1)) == 0;
    auto c_widthZero = builder.create<hw::ConstantOp>(loc, ptrTy, 0);
    auto c_widthOne = builder.create<hw::ConstantOp>(loc, ptrTy, 1);
    hw::ConstantOp c_sizeMax;
    if (!isTwoPower)
        c_sizeMax = builder.create<hw::ConstantOp>(loc, ptrTy, size - 1);

    // RAM
    auto want_close = builder.create<sv::RegOp>(loc, i1Ty);
    auto want_close_value =
        builder.create<sv::ReadInOutOp>(loc, want_close.getResult());
    auto ram = builder.create<sv::RegOp>(
        loc,
        hw::UnpackedArrayType::get(builder.getIntegerType(portBit), size));
    auto placeholderReadIndex = builder.create<hw::ConstantOp>(loc, ptrTy, 0);
    auto ram_read = builder.create<sv::ArrayIndexInOutOp>(
        loc,
        ram.getResult(),
        placeholderReadIndex.getResult());
    auto ram_read_data =
        builder.create<sv::ReadInOutOp>(loc, ram_read.getResult());

    // Pointers
    auto ptr_write = builder.create<sv::RegOp>(loc, ptrTy);
    auto ptr_write_value =
        builder.create<sv::ReadInOutOp>(loc, ptr_write.getResult());
    auto ptr_read = builder.create<sv::RegOp>(loc, ptrTy);
    auto ptr_read_value =
        builder.create<sv::ReadInOutOp>(loc, ptr_read.getResult());
    placeholderReadIndex.replaceAllUsesWith(ptr_read_value.getResult());
    auto maybe_full = builder.create<sv::RegOp>(loc, i1Ty);
    auto maybe_full_value =
        builder.create<sv::ReadInOutOp>(loc, maybe_full.getResult());

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
        c_widthOne.getResult());
    auto next_read = builder.create<comb::AddOp>(
        loc,
        ptr_read_value.getResult(),
        c_widthOne.getResult());
    auto notSameEnqDeq = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::ne,
        do_enq.getResult(),
        do_deq.getResult());
    auto not_empty =
        builder.create<comb::XorOp>(loc, empty.getResult(), c_true.getResult());
    placeholderNotEmpty.replaceAllUsesWith(not_empty.getResult());
    auto not_full =
        builder.create<comb::XorOp>(loc, full.getResult(), c_true.getResult());
    auto not_want_close = builder.create<comb::XorOp>(
        loc,
        want_close_value.getResult(),
        c_true.getResult());
    auto io_enq_ready_value = builder.create<comb::AndOp>(
        loc,
        not_full.getResult(),
        not_want_close.getResult());
    placeholderNotFull.replaceAllUsesWith(io_enq_ready_value.getResult());
    auto isDone = builder.create<comb::AndOp>(
        loc,
        want_close_value.getResult(),
        empty.getResult());

    // Clocked logic
    builder.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
        builder.create<sv::IfOp>(
            loc,
            rst,
            [&] {
                builder.create<sv::PAssignOp>(
                    loc,
                    ptr_write.getResult(),
                    c_widthZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    ptr_read.getResult(),
                    c_widthZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    maybe_full.getResult(),
                    c_false.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    want_close.getResult(),
                    c_false.getResult());
            },
            [&] {
                builder.create<sv::IfOp>(loc, do_enq.getResult(), [&] {
                    auto ram_write = builder.create<sv::ArrayIndexInOutOp>(
                        loc,
                        ram.getResult(),
                        ptr_write_value.getResult());
                    builder.create<sv::PAssignOp>(
                        loc,
                        ram_write.getResult(),
                        in_bits);
                    if (isTwoPower) {
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_write.getResult(),
                            next_write.getResult());
                    } else {
                        auto isSizeMax = builder.create<comb::ICmpOp>(
                            loc,
                            comb::ICmpPredicate::eq,
                            ptr_write_value.getResult(),
                            c_sizeMax.getResult());
                        builder.create<sv::IfOp>(
                            loc,
                            isSizeMax.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_write.getResult(),
                                    c_widthZero.getResult());
                            },
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_write.getResult(),
                                    next_write.getResult());
                            });
                    }
                });
                builder.create<sv::IfOp>(loc, do_deq.getResult(), [&] {
                    if (isTwoPower) {
                        builder.create<sv::PAssignOp>(
                            loc,
                            ptr_read.getResult(),
                            next_read.getResult());
                    } else {
                        auto isSizeMax = builder.create<comb::ICmpOp>(
                            loc,
                            comb::ICmpPredicate::eq,
                            ptr_read_value.getResult(),
                            c_sizeMax.getResult());
                        builder.create<sv::IfOp>(
                            loc,
                            isSizeMax.getResult(),
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_read.getResult(),
                                    c_widthZero.getResult());
                            },
                            [&] {
                                builder.create<sv::PAssignOp>(
                                    loc,
                                    ptr_read.getResult(),
                                    next_read.getResult());
                            });
                    }
                });
                builder.create<sv::IfOp>(loc, notSameEnqDeq.getResult(), [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        maybe_full.getResult(),
                        do_enq.getResult());
                });
                builder.create<sv::IfOp>(loc, close_signal, [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        want_close.getResult(),
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
    queueList.emplace(name, hwModule);
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
std::map<std::string, int> instantiatedOperators;
int getNameOfInstance(std::string to_find)
{
    auto it = instantiatedOperators.find(to_find);
    if (it != instantiatedOperators.end()) {
        auto num = it->second;
        it->second++;
        return num;
    } else {
        instantiatedOperators.emplace(to_find, 1);
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
        // Determine the monitored ports are from arguments
        SmallVector<int> needClosePortIdx;
        op.walk([&](InstantiateOp instanceOp) {
            auto calleeName = instanceOp.getCallee().getRootReference().str();
            auto inputChans = instanceOp.getInputs();
            auto loopedOperator = getNewIndexOrArg(calleeName, operatorHasLoop);
            if (loopedOperator.value()) {
                auto indices =
                    getNewIndexOrArg(calleeName, operatorPortIdx).value();
                for (auto idx : indices) {
                    auto loopedQueue =
                        dyn_cast<ChannelOp>(inputChans[idx].getDefiningOp());
                    auto isConnected = getNewIndexOrArg<Value, Value>(
                        loopedQueue.getInChan(),
                        newConnections);
                    if (isConnected) {
                        needClosePortIdx.push_back(isConnected.value()
                                                       .cast<BlockArgument>()
                                                       .getArgNumber());
                    }
                }
            }
        });

        auto funcTy = op.getFunctionType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumResults();

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
            auto type = funcTy.getInput(i);
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
            auto type = funcTy.getResult(i);
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
                    portBit,
                    size);
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
                auto queueInstance = rewriter.create<hw::InstanceOp>(
                    loc,
                    queueModule,
                    "queue" + std::to_string(queueSuffixNum++),
                    inputs);
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
                    getNewIndexOrArg(calleeStr, operatorPortIdx).value();
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
                auto operatorToCall = getNewIndexOrArg<Operation*, StringAttr>(
                    calleeName,
                    newOperators);
                auto nameSuffix = getNameOfInstance(calleeStr);
                auto instanceName =
                    nameSuffix == 0 ? calleeStr
                                    : calleeStr + std::to_string(nameSuffix);
                auto instanceOp = rewriter.create<hw::InstanceOp>(
                    loc,
                    operatorToCall.value(),
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

        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

void mlir::populateDfgToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertOperator>(typeConverter, patterns.getContext());
    patterns.add<LegalizeHWModule>(typeConverter, patterns.getContext());
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
        sv::SVDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();
    target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
        auto funcTy = op.getFunctionType();
        for (const auto inTy : funcTy.getInputs())
            if (dyn_cast<InputType>(inTy)) return false;
        for (const auto outTy : funcTy.getResults())
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
