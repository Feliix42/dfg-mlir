/// Implementation of StdToCirct pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/StdToCirct/StdToCirct.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wrap-process-ops"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTDTOCIRCT
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

// func.func -> hw.module
struct FuncConversion : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    FuncConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<func::FuncOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        func::FuncOp op,
        func::FuncOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto context = rewriter.getContext();
        auto funcTy = op.getFunctionType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumResults();
        auto channelStyle =
            op.getOperation()->getAttrOfType<StringAttr>("channel_style");
        StringRef channelStyleStr;
        // assert(channelStyle != nullptr);
        if (channelStyle == nullptr)
            channelStyleStr = "dfg";
        else {
            channelStyleStr = channelStyle.getValue();
            if (channelStyleStr != "dfg" && channelStyleStr != "xilinx")
                return rewriter.notifyMatchFailure(
                    loc,
                    "Unsupported Channel Style Attribute");
        }
        if (channelStyleStr == "xilinx") {
            for (auto type : funcTy.getInputs()) {
                auto bitwidth = type.getIntOrFloatBitWidth();
                if (bitwidth < 8 || bitwidth > 2048)
                    return rewriter.notifyMatchFailure(
                        loc,
                        "Xilinx FIFO only support bitwidth from 8 to 2048");
            }
        }

        SmallVector<hw::PortInfo> ports;
        int in_num = 1;
        int out_num = 1;

        for (size_t i = 0; i < numInputs; i++) {
            auto inTy = funcTy.getInput(i);
            // Add input port
            hw::PortInfo inPort;
            std::string name =
                ((numInputs == 1) ? "in" : "in" + std::to_string(in_num));
            inPort.name = rewriter.getStringAttr(name);
            inPort.dir = hw::ModulePort::Direction::Input;
            inPort.type = InputType::get(context, inTy);
            inPort.argNum = in_num++ - 1;
            ports.push_back(inPort);
        }
        for (size_t i = 0; i < numOutputs; i++) {
            auto outTy = funcTy.getResult(i);
            // Add output port
            hw::PortInfo outPort;
            std::string name =
                ((numOutputs == 1) ? "out" : "out" + std::to_string(out_num));
            outPort.name = rewriter.getStringAttr(name);
            outPort.dir = hw::ModulePort::Direction::Output;
            outPort.type = OutputType::get(context, outTy);
            outPort.argNum = out_num++ - 1;
            ports.push_back(outPort);
        }

        auto hwModule = rewriter.create<hw::HWModuleOp>(
            loc,
            op.getSymNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{NamedAttribute(
                StringAttr::get(context, "channel_style"),
                StringAttr::get(context, channelStyleStr))},
            StringAttr{},
            false);
        SmallVector<std::pair<Value, Value>> newArguments;
        for (size_t i = 0; i < numInputs; i++)
            newArguments.push_back(std::make_pair(
                hwModule.getBody().getArgument(i),
                op.getArgument(i)));

        rewriter.setInsertionPointToStart(&hwModule.getBody().front());
        SmallVector<std::pair<Value, Value>> pullOutWhich;
        for (auto &opInside :
             llvm::make_early_inc_range(op.getBody().getOps())) {
            if (auto pushOp = dyn_cast<PushOp>(opInside)) {
                auto newArg = getNewIndexOrArg(pushOp.getInp(), newArguments);
                auto portQueue = pushOp.getChan();
                rewriter.create<HWConnectOp>(
                    rewriter.getUnknownLoc(),
                    newArg.value(),
                    portQueue);
            } else if (auto pullOp = dyn_cast<PullOp>(opInside)) {
                pullOutWhich.push_back(
                    std::make_pair(pullOp.getChan(), pullOp.getOutp()));
            } else if (auto returnOp = dyn_cast<func::ReturnOp>(opInside)) {
                SmallVector<Value> newOutputs;
                for (auto returnValue : returnOp.getOperands()) {
                    auto newOutput = getNewIndexOrArg<Value, Value>(
                        returnValue,
                        pullOutWhich);
                    newOutputs.push_back(newOutput.value());
                }
                rewriter.create<hw::OutputOp>(
                    rewriter.getUnknownLoc(),
                    newOutputs);
            } else if (auto channelOp = dyn_cast<ChannelOp>(opInside)) {
                if (channelStyleStr == "xilinx") {
                    auto bufSize = channelOp.getBufferSize();
                    if (!bufSize.has_value())
                        return rewriter.notifyMatchFailure(
                            channelOp.getLoc(),
                            "For hardware lowering, a channel must have a "
                            "size.");
                    auto sizeValue = bufSize.value();
                    auto size =
                        sizeValue < 16
                            ? 16
                            : (sizeValue > 4194304 ? 4194304 : sizeValue);
                    channelOp.setBufferSize(size);
                }
                opInside.moveBefore(
                    hwModule.getBodyBlock(),
                    hwModule.getBodyBlock()->end());
            } else {
                opInside.moveBefore(
                    hwModule.getBodyBlock(),
                    hwModule.getBodyBlock()->end());
            }
        }

        rewriter.eraseOp(op);

        return success();
    }
};

// Wrap calc ops in one handshake func
std::optional<int> getResultIdx(Value value, Operation* op)
{
    if (op == nullptr) return std::nullopt;
    for (size_t i = 0; i < op->getNumResults(); i++)
        if (op->getResult(i) == value) return (int)i;
    return std::nullopt;
}
void writeFuncToFile(func::FuncOp funcOp, StringRef funcName)
{
    std::string filename = funcName.str() + ".mlir";
    std::error_code error;
    llvm::raw_fd_ostream outputFile(filename, error);

    if (error) {
        llvm::errs()
            << "Could not open output file: " << error.message() << "\n";
        return;
    }

    funcOp.print(outputFile);
    outputFile.close();
}
void processNestedRegions(
    Operation* newCalcOp,
    SmallVector<Operation*> &newCalcOps,
    SmallVector<std::pair<int, Value>> &pulledValueIdx,
    SmallVector<std::pair<int, Operation*>> &calcOpIdx,
    SmallVector<std::pair<int, Value>> &iterArgsIdx,
    func::FuncOp &genFuncOp,
    size_t inputBias,
    size_t outputBias,
    PatternRewriter &rewriter)
{
    LLVM_DEBUG(
        llvm::dbgs() << "\nInserting op " << newCalcOp->getName() << " from "
                     << newCalcOp->getLoc() << "\n");
    int idxOperand = 0;

    for (auto operand : newCalcOp->getOperands()) {
        if (auto idxArg =
                getNewIndexOrArg<int, Value>(operand, pulledValueIdx)) {
            LLVM_DEBUG(
                llvm::dbgs() << "Found value: " << idxArg.value() << "\n");
            newCalcOp->setOperand(
                idxOperand++,
                genFuncOp.getBody().getArgument(idxArg.value()));
        } else {
            if (isa<BlockArgument>(operand)) {
                LLVM_DEBUG(
                    llvm::dbgs() << "Found number " << idxOperand
                                 << " operand, which is a " << operand << "\n");
                idxOperand++;
                continue;
            }
            auto definingOp = operand.getDefiningOp();
            if (auto idxIterArg =
                    getNewIndexOrArg<int, Value>(operand, iterArgsIdx)) {
                auto idx = idxIterArg.value();
                LLVM_DEBUG(
                    llvm::dbgs()
                    << "Found number " << idxOperand << " operand is number "
                    << idx << " iter_arg\n");
                newCalcOp->setOperand(
                    idxOperand++,
                    genFuncOp.getBody().getArgument(idx + inputBias));
                LLVM_DEBUG(
                    llvm::dbgs() << "Replaced with argment " << idx + inputBias
                                 << " of the function\n");
                continue;
            }
            LLVM_DEBUG(
                llvm::dbgs()
                << "Found number " << idxOperand << " operand's defining op "
                << definingOp->getName() << " at " << definingOp->getLoc()
                << "\n");
            auto idxCalcOp =
                getNewIndexOrArg<int, Operation*>(definingOp, calcOpIdx);
            auto idxResult = getResultIdx(operand, definingOp);
            if (idxResult) {
                if (idxCalcOp)
                    newCalcOp->setOperand(
                        idxOperand++,
                        newCalcOps[idxCalcOp.value()]->getResult(
                            idxResult.value()));
                else
                    newCalcOp->setOperand(
                        idxOperand++,
                        definingOp->getResult(idxResult.value()));
            }
            LLVM_DEBUG(
                llvm::dbgs() << "Replaced with result " << idxResult.value()
                             << " of the defining op\n");
        }
    }
    if (newCalcOp->getRegions().size() != 0) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "\nFound region at " << newCalcOp->getLoc() << "\n");
        for (auto &region : newCalcOp->getRegions()) {
            for (auto &opRegion : region.getOps()) {
                processNestedRegions(
                    &opRegion,
                    newCalcOps,
                    pulledValueIdx,
                    calcOpIdx,
                    iterArgsIdx,
                    genFuncOp,
                    inputBias,
                    outputBias,
                    rewriter);
            }
        }
    }
}
struct WrapProcessOps : public OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    WrapProcessOps(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto context = rewriter.getContext();
        auto operatorName = op.getSymName();
        auto funcTy = op.getFunctionType();
        auto opsProcess = op.getOps();
        auto moduleOp = op->getParentOfType<ModuleOp>();

        LoopOp loopOp = nullptr;
        bool hasLoopOp = false;
        bool hasIterArgs = false;
        SmallVector<int> loopInChanIdx, loopOutChanIdx;
        for (auto &opi : opsProcess)
            if (auto oldLoop = dyn_cast<LoopOp>(opi)) {
                // ops = oldLoop.getOps();
                loopOp = oldLoop;
                hasLoopOp = true;
                hasIterArgs = !oldLoop.getIterArgs().empty();
                for (auto inChan : oldLoop.getInChans()) {
                    auto idxChan = cast<BlockArgument>(inChan).getArgNumber();
                    loopInChanIdx.push_back(idxChan);
                }
                for (auto outChan : oldLoop.getOutChans()) {
                    auto idxChan = cast<BlockArgument>(outChan).getArgNumber();
                    loopOutChanIdx.push_back(idxChan);
                }
            }

        int idxPull = 0;
        int numPull = 0;
        int idxCalc = 0;
        int numCalc = 0;
        int numPush = 0;
        int numResult = 0;
        SmallVector<Value> pulledValue;
        SmallVector<Type> pulledTypes;
        SmallVector<int> pulledChanIdx;
        SmallVector<std::pair<int, Value>> pulledValueIdx;
        SmallVector<Value> pushedValueRepeat;
        SmallVector<Value> pushedValueNonRepeat;
        SmallVector<Type> pushedTypes;
        SmallVector<int> pushedChanIdx;
        std::vector<int> pushedValueIdx;
        SmallVector<std::pair<int, Value>> calcResultIdx;
        SmallVector<std::pair<int, Operation*>> calcOpIdx;
        SmallVector<Type> iterArgsTypes;
        SmallVector<Type> iterArgsReturnTypes;
        SmallVector<Value> iterArgsReturnValues;
        SmallVector<int> iterArgsReturnIdx;
        auto ops = loopOp.getBody().getOps();
        for (auto &opi : ops)
            if (auto pushOp = dyn_cast<PushOp>(opi))
                pushedValueRepeat.push_back(pushOp.getInp());
        for (auto &opi : ops) {
            if (auto pullOp = dyn_cast<PullOp>(opi)) {
                auto pullValue = pullOp.getOutp();
                pulledValue.push_back(pullValue);
                auto pullChan = pullOp.getChan();
                pulledTypes.push_back(pullChan.getType().getElementType());
                auto idxChan = cast<BlockArgument>(pullChan).getArgNumber();
                pulledChanIdx.push_back(idxChan);
                pulledValueIdx.push_back(std::make_pair(idxPull++, pullValue));
                numPull++;
            } else if (!isa<PushOp>(opi) && !isa<YieldOp>(opi)) {
                for (auto result : opi.getResults())
                    if (isInSmallVector<Value>(result, pushedValueRepeat))
                        calcResultIdx.push_back(
                            std::make_pair(idxCalc++, result));
                calcOpIdx.push_back(std::make_pair(numCalc++, &opi));
            } else if (auto pushOp = dyn_cast<PushOp>(opi)) {
                auto pushValue = pushOp.getInp();
                auto pushChan = pushOp.getChan();
                auto isPushPulledValue =
                    isInSmallVector<Value>(pushValue, pulledValue);
                if (!isInSmallVector<Value>(pushValue, pushedValueNonRepeat)
                    && !isPushPulledValue) {
                    pushedValueNonRepeat.push_back(pushValue);
                    pushedTypes.push_back(pushChan.getType().getElementType());
                    numResult++;
                }
                auto idxChan = cast<BlockArgument>(pushChan).getArgNumber();
                pushedChanIdx.push_back(idxChan);
                if (!isPushPulledValue) {
                    auto idx =
                        getNewIndexOrArg<int, Value>(pushValue, calcResultIdx);
                    pushedValueIdx.push_back(idx.value());
                } else {
                    pushedValueIdx.push_back(-1);
                }
                numPush++;
            } else if (auto yieldOp = dyn_cast<YieldOp>(opi)) {
                for (auto type : yieldOp.getOperandTypes())
                    iterArgsTypes.push_back(type);
                int bias = numResult;
                for (auto value : yieldOp.getOperands())
                    if (isInSmallVector<Value>(value, pushedValueNonRepeat)) {
                        // If yield value is in the pushed values, store the idx
                        // of the pushed value
                        auto idx =
                            getVectorIdx<Value>(value, pushedValueNonRepeat);
                        iterArgsReturnIdx.push_back(idx.value());
                    } else {
                        // If not, store the idx from the last one in return
                        // values of new generated func and save the type and
                        // value for later usage
                        iterArgsReturnTypes.push_back(value.getType());
                        iterArgsReturnValues.push_back(value);
                        iterArgsReturnIdx.push_back(bias++);
                    }
            }
        }

        auto newOperator =
            rewriter.create<ProcessOp>(op.getLoc(), op.getSymName(), funcTy);
        LLVM_DEBUG(
            llvm::dbgs() << "\nInserting " << newOperator << " at "
                         << newOperator.getLoc() << "\n");
        SmallVector<Value> newPulledValue;
        SmallVector<std::pair<int, Value>> iterArgsIdx;
        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPointToStart(&newOperator.getBody().front());
        SmallVector<Value> newIterArgs;
        if (hasIterArgs) {
            size_t i = 0;
            auto iterArgsSize = loopOp.getIterArgs().size();
            assert(iterArgsSize == 1);
            for (auto &opi : opsProcess) {
                if (i == iterArgsSize) break;
                if (auto constantOp = dyn_cast<arith::ConstantOp>(opi)) {
                    iterArgsIdx.push_back(
                        std::make_pair(i, constantOp.getResult()));
                    auto constant = constantOp.clone();
                    newIterArgs.push_back(constant.getResult());
                    rewriter.insert(constant);
                    i++;
                } else {
                    return rewriter.notifyMatchFailure(
                        opi.getLoc(),
                        "Wrong iter_args initialization here.");
                }
            }
        }
        if (hasLoopOp) {
            SmallVector<Value> loopInChans, loopOutChans;
            for (auto inChanIdx : loopInChanIdx) {
                loopInChans.push_back(
                    newOperator.getBody().getArgument(inChanIdx));
            }
            for (auto outChanIdx : loopOutChanIdx) {
                loopOutChans.push_back(
                    newOperator.getBody().getArgument(outChanIdx));
            }
            auto newLoop = rewriter.create<LoopOp>(
                loc,
                loopInChans,
                loopOutChans,
                newIterArgs);
            LLVM_DEBUG(
                llvm::dbgs()
                << "\nInserting " << newLoop << " into new process op.\n");
            rewriter.setInsertionPointToStart(&newLoop.getBody().front());
        }
        for (int i = 0; i < numPull; i++) {
            auto newPull = rewriter.create<PullOp>(
                loc,
                newOperator.getBody().getArgument(pulledChanIdx[i]));
            LLVM_DEBUG(
                llvm::dbgs()
                << "\nInserting " << newPull << " into new process op.\n");
            newPulledValue.push_back(newPull.getResult());
        }

        auto nameExtModule = "hls_" + operatorName.str() + "_calc";
        SmallVector<Value> hwInstanceInputs;
        hwInstanceInputs.append(newPulledValue.begin(), newPulledValue.end());
        hwInstanceInputs.append(newIterArgs.begin(), newIterArgs.end());
        SmallVector<Type> hwInstanceOutputTypes;
        hwInstanceOutputTypes.append(pushedTypes.begin(), pushedTypes.end());
        hwInstanceOutputTypes.append(
            iterArgsReturnTypes.begin(),
            iterArgsReturnTypes.end());
        auto instanceOp = rewriter.create<HWInstanceOp>(
            loc,
            hwInstanceOutputTypes,
            SymbolRefAttr::get(context, nameExtModule),
            hwInstanceInputs);

        for (int i = 0; i < numPush; i++) {
            auto pushValue = pushedValueRepeat[i];
            auto idxChan = pushedChanIdx[i];
            if (isInSmallVector<Value>(pushValue, pulledValue)) {
                auto idxPullValue =
                    getNewIndexOrArg<int, Value>(pushValue, pulledValueIdx);
                rewriter.create<PushOp>(
                    loc,
                    newPulledValue[idxPullValue.value()],
                    newOperator.getBody().getArgument(idxChan));
                continue;
            }
            rewriter.create<PushOp>(
                loc,
                instanceOp.getResult(pushedValueIdx[i]),
                newOperator.getBody().getArgument(idxChan));
        }

        if (hasIterArgs) {
            SmallVector<Value> newYieldValues;
            for (auto idx : iterArgsReturnIdx)
                newYieldValues.push_back(instanceOp.getResult(idx));
            rewriter.create<YieldOp>(loc, newYieldValues);
        }

        rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
        SmallVector<Type> funcInputTypes;
        funcInputTypes.append(pulledTypes.begin(), pulledTypes.end());
        funcInputTypes.append(iterArgsTypes.begin(), iterArgsTypes.end());
        auto genFuncOp = rewriter.create<func::FuncOp>(
            loc,
            nameExtModule,
            rewriter.getFunctionType(funcInputTypes, hwInstanceOutputTypes));
        Block* funcEntryBlock = rewriter.createBlock(&genFuncOp.getBody());
        for (auto type : funcInputTypes)
            funcEntryBlock->addArgument(type, genFuncOp.getLoc());
        LLVM_DEBUG(
            llvm::dbgs()
            << "\nInserting " << genFuncOp << " into new process op.\n");
        rewriter.setInsertionPointToStart(funcEntryBlock);

        SmallVector<Operation*> newCalcOps;
        LLVM_DEBUG(llvm::dbgs() << "\nInserting ops into the func.\n");
        for (int i = 0; i < numCalc; i++) {
            auto it = ops.begin();
            std::advance(it, numPull + i);
            auto oldCalcOp = &*it;
            auto newCalcOp = oldCalcOp->clone();
            processNestedRegions(
                newCalcOp,
                newCalcOps,
                pulledValueIdx,
                calcOpIdx,
                iterArgsIdx,
                genFuncOp,
                pulledTypes.size(),
                pushedTypes.size(),
                rewriter);
            newCalcOps.push_back(newCalcOp);
            rewriter.insert(newCalcOp);
        }

        // Resolve the return values
        SmallVector<Value> returnValues;
        for (int i = 0; i < numResult; i++) {
            auto result = pushedValueNonRepeat[i];
            auto definingOp = result.getDefiningOp();
            auto idxCalcOp =
                getNewIndexOrArg<int, Operation*>(definingOp, calcOpIdx);
            auto idxResult = getResultIdx(result, definingOp);
            returnValues.push_back(
                newCalcOps[idxCalcOp.value()]->getResult(idxResult.value()));
        }
        for (auto value : iterArgsReturnValues) {
            auto definingOp = value.getDefiningOp();
            auto idxCalcOp =
                getNewIndexOrArg<int, Operation*>(definingOp, calcOpIdx);
            auto idxResult = getResultIdx(value, definingOp);
            returnValues.push_back(
                newCalcOps[idxCalcOp.value()]->getResult(idxResult.value()));
        }

        rewriter.create<func::ReturnOp>(loc, returnValues);
        LLVM_DEBUG(llvm::dbgs() << "\nInserting return.\n");

        writeFuncToFile(genFuncOp, genFuncOp.getSymName());
        rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
        SmallVector<hw::PortInfo> ports;
        auto i1Ty = rewriter.getI1Type();
        auto inDir = hw::ModulePort::Direction::Input;
        auto outDir = hw::ModulePort::Direction::Output;
        for (size_t i = 0; i < funcInputTypes.size(); i++) {
            auto type = funcInputTypes[i];
            auto namePrefix = "in" + std::to_string(i);
            // data port
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr(namePrefix), type, inDir}
            });
            // valid port
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr(namePrefix + "_valid"), i1Ty, inDir}
            });
            // ready port
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr(namePrefix + "_ready"), i1Ty, outDir}
            });
        }
        // ctrl valid
        ports.push_back(hw::PortInfo{
            {rewriter.getStringAttr(
                 "in" + std::to_string(funcInputTypes.size()) + "_valid"),
             i1Ty, inDir}
        });
        // clock port
        ports.push_back(hw::PortInfo{
            {rewriter.getStringAttr("clock"), i1Ty, inDir}
        });
        // reset port
        ports.push_back(hw::PortInfo{
            {rewriter.getStringAttr("reset"), i1Ty, inDir}
        });
        for (size_t i = 0; i < hwInstanceOutputTypes.size(); i++) {
            auto type = hwInstanceOutputTypes[i];
            auto namePrefix = "out" + std::to_string(i);
            // ready port
            std::string name;
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr(namePrefix + "_ready"), i1Ty, inDir}
            });
            // data port
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr(namePrefix), type, outDir}
            });
            // valid port
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr(namePrefix + "_valid"), i1Ty, outDir}
            });
        }
        rewriter.create<hw::HWModuleExternOp>(
            loc,
            rewriter.getStringAttr(nameExtModule),
            ports);
        rewriter.eraseOp(genFuncOp);

        rewriter.eraseOp(op);

        return success();
    }
};

struct LegalizeChannelSize : public OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    LegalizeChannelSize(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto buffSize = op.getBufferSize();
        if (buffSize == std::nullopt)
            return rewriter.notifyMatchFailure(
                op.getLoc(),
                "For hardware lowering, a channel must have a "
                "size.");

        auto sizeValue = buffSize.value();
        int size =
            sizeValue < 16 ? 16 : (sizeValue > 4194304 ? 4194304 : sizeValue);
        rewriter.replaceOpWithNewOp<ChannelOp>(
            op,
            op.getEncapsulatedType(),
            size);

        return success();
    }
};

} // namespace

void mlir::populateStdToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{

    // func.func -> hw.module
    patterns.add<FuncConversion>(typeConverter, patterns.getContext());

    // operator calc ops -> handshake.func
    patterns.add<WrapProcessOps>(typeConverter, patterns.getContext());

    // channel op -> change size
    patterns.add<LegalizeChannelSize>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertStdToCirctPass
        : public impl::ConvertStdToCirctBase<ConvertStdToCirctPass> {
    void runOnOperation() override;
};
} // namespace

void ConvertStdToCirctPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) {
        if (isa<IntegerType>(type)) return type;
        return Type();
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateStdToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<
        arith::ArithDialect,
        scf::SCFDialect,

        comb::CombDialect,
        handshake::HandshakeDialect,
        hw::HWDialect,
        sv::SVDialect,
        dfg::DfgDialect>();

    func::FuncOp lastFunc;
    auto module = dyn_cast<ModuleOp>(getOperation());
    module.walk([&](func::FuncOp funcOp) { lastFunc = funcOp; });
    ChannelStyle channelStyle;
    module.walk([&](SetChannelStyleOp setChanStyleOp) {
        channelStyle = setChanStyleOp.getChannelStyle();
    });

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
        // auto name = funcOp.getSymName().str();
        // return name != "top";
        return funcOp != lastFunc;
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp returnOp) {
        auto funcOp = returnOp.getParentOp();
        // auto name = funcOp.getSymName().str();
        // return name != "top";
        return funcOp != lastFunc;
    });

    target.addDynamicallyLegalOp<ChannelOp>([&](ChannelOp channelOp) {
        auto buffSize = channelOp.getBufferSize();
        if (buffSize == std::nullopt) return false;
        if (channelStyle == ChannelStyle::Xilinx) {
            if (buffSize < 8 || buffSize > 4194304) return false;
        }
        return true;
    });

    target.addDynamicallyLegalOp<ProcessOp>([&](ProcessOp op) {
        auto ops = op.getBody().getOps();
        for (auto &opi : ops)
            if (isa<arith::ConstantOp>(opi))
                continue;
            else if (auto loopOp = dyn_cast<LoopOp>(opi)) {
                for (auto &opLoop : loopOp.getBody().getOps()) {
                    if (!isa<PullOp>(opLoop) && !isa<HWInstanceOp>(opLoop)
                        && !isa<PushOp>(opLoop) && !isa<YieldOp>(opLoop)) {
                        return false;
                    }
                }
            } else
                return false;
        // if (auto loopOp = dyn_cast<LoopOp>(*ops.begin()))
        //     ops = loopOp.getBody().getOps();
        // for (auto &opi : ops) {
        //     if (!isa<PullOp>(opi) && !isa<HWInstanceOp>(opi)
        //         && !isa<PushOp>(opi)) {
        //         return false;
        //     }
        // }
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertStdToCirctPass()
{
    return std::make_unique<ConvertStdToCirctPass>();
}

#undef DEBUG_TYPE
