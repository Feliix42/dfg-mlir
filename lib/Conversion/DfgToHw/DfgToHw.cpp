/// Implementation of DfgToHw pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToHw/DfgToHw.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOHW
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

// Helper, whcih stores the modules for arith with parameters
SmallVector<hw::HWModuleOp> arithModules;
// Helpers, which contain backedges for the lowrring of ChannelOp and
// InstantiateOp
SmallVector<std::pair<Value, SmallVector<Value>>> channelInputToInstanceOutput;
SmallVector<std::pair<Value, Value>> channelOutputToInstanceInput;
SmallVector<std::pair<Value, Value>> regionOutputToInstanceOutput;
SmallVector<Value> processIterArgs;

SmallVector<hw::PortInfo> getHWPorts(
    OpBuilder builder,
    FunctionType funcTy,
    bool hasClose = false,
    bool hasDone = false)
{
    auto funcTyNumInputs = funcTy.getNumInputs();
    auto funcTyNumOutputs = funcTy.getNumResults();
    auto i1Ty = builder.getI1Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;
    SmallVector<hw::PortInfo> ports, subModulePorts;
    // Add clock port.
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("clock"), i1Ty, inDir}
    });

    // Add reset port.
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("reset"), i1Ty, inDir}
    });
    for (size_t i = 0; i < funcTyNumInputs; i++) {
        auto inputTy = dyn_cast<IntegerType>(funcTy.getInput(i));
        auto namePrefix = "in" + std::to_string(i);
        // Add ready port
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr(namePrefix + "_ready"), i1Ty, outDir}
        });
        // Add valid port
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr(namePrefix + "_valid"), i1Ty, inDir}
        });
        // Add data port
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr(namePrefix), inputTy, inDir}
        });
        // Add close port (If the port is not looped, dangling the
        // port, the submodule won't take close port)
        if (hasClose)
            ports.push_back(hw::PortInfo{
                {builder.getStringAttr(namePrefix + "_close"), i1Ty, inDir}
            });
    }
    for (size_t i = 0; i < funcTyNumOutputs; i++) {
        auto outputTy = dyn_cast<IntegerType>(funcTy.getResult(i));
        auto namePrefix = "out" + std::to_string(i);
        // Add ready port
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr(namePrefix + "_ready"), i1Ty, inDir}
        });
        // Add valid port
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr(namePrefix + "_valid"), i1Ty, outDir}
        });
        // Add data port
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr(namePrefix), outputTy, outDir}
        });
    }
    if (hasDone)
        ports.push_back(hw::PortInfo{
            {builder.getStringAttr("done"), i1Ty, outDir}
        });

    return ports;
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
getReplaceValuesFromCasts(
    OpBuilder builder,
    hw::HWModuleOp hwModule,
    FunctionType funcTy,
    bool hasClose = false,
    MutableArrayRef<BlockArgument> originalBlkArgs = {})
{
    auto loc = hwModule.getLoc();
    auto i1Ty = builder.getI1Type();
    auto funcTyNumInputs = funcTy.getNumInputs();
    auto funcTyNumOutputs = funcTy.getNumResults();
    Region &hwModuleRegion = hwModule.getBody();
    builder.setInsertionPointToEnd(&hwModuleRegion.front());
    SmallVector<Value> inReadySignals, outValidDataSignals, replacePorts;
    for (size_t i = 0; i < funcTyNumInputs; i++) {
        auto inputTy = dyn_cast<IntegerType>(funcTy.getInput(i));
        SmallVector<Value> hwModuleBlkArgs;
        if (hasClose) {
            hwModuleBlkArgs.push_back(hwModuleRegion.getArgument(3 * i + 2));
            hwModuleBlkArgs.push_back(hwModuleRegion.getArgument(3 * i + 3));
            hwModuleBlkArgs.push_back(hwModuleRegion.getArgument(3 * i + 4));
        } else {
            hwModuleBlkArgs.push_back(hwModuleRegion.getArgument(2 * i + 2));
            hwModuleBlkArgs.push_back(hwModuleRegion.getArgument(2 * i + 3));
        }
        auto hwCastOp = builder.create<UnrealizedConversionCastOp>(
            loc,
            TypeRange{i1Ty, OutputType::get(builder.getContext(), inputTy)},
            hwModuleBlkArgs);
        if (!originalBlkArgs.empty()) {
            auto user = *originalBlkArgs[i].getUsers().begin();
            if (auto instantiateOp =
                    dyn_cast<InstantiateOp>(user) ?: dyn_cast<EmbedOp>(user)) {
                channelOutputToInstanceInput.push_back(std::make_pair(
                    hwCastOp.getResult(1),
                    hwCastOp.getResult(0)));
            }
        }
        inReadySignals.push_back(hwCastOp.getResult(0));
        replacePorts.push_back(hwCastOp.getResult(1));
    }
    for (size_t i = 0; i < funcTyNumOutputs; i++) {
        auto outputTy = dyn_cast<IntegerType>(funcTy.getResult(i));
        auto hwCastOp = builder.create<UnrealizedConversionCastOp>(
            loc,
            TypeRange{
                i1Ty,
                outputTy,
                InputType::get(builder.getContext(), outputTy)},
            hasClose ? hwModuleRegion.getArgument(i + 3 * funcTyNumInputs + 2)
                     : hwModuleRegion.getArgument(i + 2 * funcTyNumInputs + 2));
        SmallVector<Value> validData;
        validData.push_back(hwCastOp.getResult(0));
        validData.push_back(hwCastOp.getResult(1));
        if (!originalBlkArgs.empty()) {
            auto user =
                *originalBlkArgs[i + funcTyNumInputs].getUsers().begin();
            if (auto instantiateOp =
                    dyn_cast<InstantiateOp>(user) ?: dyn_cast<EmbedOp>(user)) {
                channelInputToInstanceOutput.push_back(
                    std::make_pair(hwCastOp.getResult(2), validData));
            }
        }
        outValidDataSignals.append(validData.begin(), validData.end());
        replacePorts.push_back(hwCastOp.getResult(2));
    }
    return std::make_tuple(inReadySignals, outValidDataSignals, replacePorts);
}

struct LowerProcessToHWModule : OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    LowerProcessToHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctx = rewriter.getContext();
        auto args = op.getBody().getArguments();
        auto funcTy = op.getFunctionType();
        auto funcTyNumInputs = funcTy.getNumInputs();
        auto funcTyNumOutputs = funcTy.getNumResults();
        auto i1Ty = rewriter.getI1Type();
        SmallVector<Type> inElemTy, outElemty;
        for (auto inTy : funcTy.getInputs()) {
            auto elemTy = dyn_cast<OutputType>(inTy).getElementType();
            assert(
                isa<IntegerType>(elemTy)
                && "only integers are supported on hardware");
            inElemTy.push_back(elemTy);
        }
        for (auto outTy : funcTy.getResults()) {
            auto elemTy = dyn_cast<InputType>(outTy).getElementType();
            assert(
                isa<IntegerType>(elemTy)
                && "only integers are supported on hardware");
            outElemty.push_back(elemTy);
        }
        auto intergerFuncTy = FunctionType::get(ctx, inElemTy, outElemty);

        // Create new HWModule
        auto ports = getHWPorts(
            OpBuilder(ctx),
            intergerFuncTy,
            /*hasClose*/ true,
            /*hasDone*/ true);
        auto hwModule = rewriter.create<hw::HWModuleOp>(
            op.getLoc(),
            op.getSymNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        auto hwModuleLoc = hwModule.getLoc();
        Region &hwModuleRegion = hwModule.getBody();
        rewriter.setInsertionPointToStart(&hwModuleRegion.front());
        auto [inReadySignals, outValidDataSignals, replacePorts] =
            getReplaceValuesFromCasts(
                OpBuilder(ctx),
                hwModule,
                intergerFuncTy,
                true);

        // Insert HWLoopOp to generate the done signal, which will later be
        // converted to the corresponding logic
        SmallVector<Operation*> kernelOps;
        bool hasLoop = false;
        llvm::DenseMap<Value, Value> argumentMap;
        for (size_t i = 0; i < args.size(); i++)
            argumentMap[args[i]] = replacePorts[i];
        Value loopOpResult;
        processIterArgs.clear();
        for (auto &opi : op.getBody().getOps()) {
            if (auto loopOp = dyn_cast<LoopOp>(opi)) {
                SmallVector<Value> loopedPorts;
                auto loopOpInChans = loopOp.getInChans();
                auto loopOpIterArgs = loopOp.getIterArgs();
                processIterArgs.append(
                    loopOpIterArgs.begin(),
                    loopOpIterArgs.end());
                if (loopOpInChans.size() == 0) {
                    for (auto map : argumentMap) {
                        auto inChan = map.second;
                        if (dyn_cast<OutputType>(inChan.getType()))
                            loopedPorts.push_back(inChan);
                    }
                } else {
                    for (auto inChan : loopOpInChans)
                        if (argumentMap.count(inChan))
                            loopedPorts.push_back(argumentMap[inChan]);
                }
                auto hwLoopOp =
                    rewriter.create<HWLoopOp>(hwModuleLoc, i1Ty, loopedPorts);
                loopOpResult = hwLoopOp.getResult();
                hasLoop = true;
                for (auto &opiLoop : loopOp.getBody().getOps())
                    kernelOps.push_back(&opiLoop);
            } else {
                if (!isa<arith::ConstantOp>(opi))
                    return rewriter.notifyMatchFailure(
                        opi.getLoc(),
                        "Unsupported op outside of the loop.");
                kernelOps.push_back(&opi);
            }
        }

        // Wrap every operations into a new module
        rewriter.setInsertionPoint(hwModule);
        auto kernelPorts = getHWPorts(
            OpBuilder(ctx),
            intergerFuncTy,
            /*hasClose*/ false,
            /*hasDone*/ true);
        auto hwKernelModule = rewriter.create<hw::HWModuleOp>(
            op.getLoc(),
            StringAttr::get(ctx, op.getSymNameAttr().str() + "_kernel"),
            hw::ModulePortInfo(kernelPorts),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        auto hwKernelModuleLoc = hwKernelModule.getLoc();
        Region &hwKernelModuleRegion = hwKernelModule.getBody();
        rewriter.setInsertionPointToStart(&hwKernelModuleRegion.front());
        Value cstTrue;
        for (auto &opi : kernelOps) {
            if (isa<arith::ConstantOp>(opi)) {
                cstTrue =
                    rewriter.create<hw::ConstantOp>(hwKernelModuleLoc, i1Ty, 1)
                        .getResult();
                break;
            }
        }
        auto
            [kernelInReadySignals,
             kernelOutValidDataSignals,
             kernelReplacePorts] =
                getReplaceValuesFromCasts(
                    OpBuilder(ctx),
                    hwKernelModule,
                    intergerFuncTy,
                    false);
        llvm::DenseMap<Value, Value> kernelArgumentMap, kernelResultMap;
        for (size_t i = 0; i < args.size(); i++)
            kernelArgumentMap[args[i]] = kernelReplacePorts[i];
        for (auto &opi : kernelOps) {
            Operation* newOp = rewriter.clone(*opi);
            for (size_t i = 0; i < newOp->getResults().size(); i++)
                kernelResultMap[opi->getResult(i)] = newOp->getResult(i);
            for (size_t i = 0; i < newOp->getOperands().size(); i++) {
                auto operand = newOp->getOperand(i);
                if (kernelArgumentMap.count(operand))
                    newOp->setOperand(i, kernelArgumentMap[operand]);
                if (kernelResultMap.count(operand))
                    newOp->setOperand(i, kernelResultMap[operand]);
            }
            // If the copying op is constant, create an unrealized cast op to
            // combine the true signal and the value, which will be inputs for
            // the module isntance. If constant is actually an iter_arg, replace
            // the storage in the vector with it.
            if (isa<arith::ConstantOp>(newOp)) {
                for (auto &v : processIterArgs)
                    if (v == opi->getResult(0)) v = newOp->getResult(0);
                auto constant = newOp->getResult(0);
                auto castOp = rewriter.create<UnrealizedConversionCastOp>(
                    hwKernelModuleLoc,
                    constant.getType(),
                    ValueRange{cstTrue, constant});
                // Update the result mapping as well
                kernelResultMap[opi->getResult(0)] = castOp.getResult(0);
            }
        }
        // By default, the last push to each channel should be waited to
        // generate the done signal for the kernel module
        SmallVector<PushOp> lastPushes;
        for (auto it = kernelOps.rbegin(); it != kernelOps.rend(); ++it) {
            if (auto pushOp = dyn_cast<PushOp>(*it)) {
                if (!isInSmallVector(pushOp, lastPushes))
                    lastPushes.push_back(pushOp);
            }
        }
        SmallVector<Value> waitedValues;
        for (auto pushOp : lastPushes) {
            auto waitedValue = kernelResultMap[pushOp.getInp()];
            if (!isInSmallVector(waitedValue, waitedValues))
                waitedValues.push_back(waitedValue);
        }
        auto waitOp =
            rewriter.create<HWWaitOp>(hwKernelModuleLoc, i1Ty, waitedValues);
        SmallVector<Value> kernelOutReadys;
        kernelOutReadys.push_back(waitOp.getResult());
        for (auto pushOp : lastPushes) {
            auto pushChan = kernelArgumentMap[pushOp.getChan()];
            kernelOutReadys.push_back(
                pushChan.getDefiningOp<UnrealizedConversionCastOp>().getOperand(
                    0));
        }
        auto kernelDoneOutput = rewriter.create<comb::AndOp>(
            hwKernelModuleLoc,
            i1Ty,
            kernelOutReadys);
        SmallVector<Value> kernelOutputs;
        kernelOutputs.append(
            kernelInReadySignals.begin(),
            kernelInReadySignals.end());
        kernelOutputs.append(
            kernelOutValidDataSignals.begin(),
            kernelOutValidDataSignals.end());
        kernelOutputs.push_back(kernelDoneOutput.getResult());
        rewriter.create<hw::OutputOp>(hwKernelModuleLoc, kernelOutputs);

        // Insert an instance to the kernel module
        rewriter.setInsertionPointToEnd(&hwModuleRegion.front());
        SmallVector<Value> filteredArgs;
        filteredArgs.push_back(hwModuleRegion.getArgument(0));
        filteredArgs.push_back(hwModuleRegion.getArgument(1));
        for (size_t i = 0; i < funcTyNumInputs; i++) {
            filteredArgs.push_back(hwModuleRegion.getArgument(3 * i + 2));
            filteredArgs.push_back(hwModuleRegion.getArgument(3 * i + 3));
        }
        for (size_t i = 0; i < funcTyNumOutputs; i++) {
            filteredArgs.push_back(
                hwModuleRegion.getArgument(i + 3 * funcTyNumInputs + 2));
        }
        auto instanceKernel = rewriter.create<hw::InstanceOp>(
            hwModuleLoc,
            hwKernelModule,
            hwKernelModule.getSymNameAttr(),
            filteredArgs);

        // OutputOp
        SmallVector<Value> outputs;
        outputs.append(
            instanceKernel.getResults().begin(),
            std::prev(instanceKernel.getResults().end()));
        Value doneSignal;
        if (hasLoop)
            doneSignal = rewriter
                             .create<comb::AndOp>(
                                 hwModuleLoc,
                                 instanceKernel.getResults().back(),
                                 loopOpResult)
                             .getResult();
        else
            doneSignal = instanceKernel.getResults().back();
        outputs.push_back(doneSignal);
        rewriter.create<hw::OutputOp>(hwModuleLoc, outputs);

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerRegionToHWModule : OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    LowerRegionToHWModule(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        RegionOp op,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctx = rewriter.getContext();
        auto args = op.getBody().getArguments();
        auto funcTy = op.getFunctionType();
        auto i1Ty = rewriter.getI1Type();
        SmallVector<Type> inElemTy, outElemty;
        for (auto inTy : funcTy.getInputs()) {
            auto elemTy = dyn_cast<OutputType>(inTy).getElementType();
            assert(
                isa<IntegerType>(elemTy)
                && "only integers are supported on hardware");
            inElemTy.push_back(elemTy);
        }
        for (auto outTy : funcTy.getResults()) {
            auto elemTy = dyn_cast<InputType>(outTy).getElementType();
            assert(
                isa<IntegerType>(elemTy)
                && "only integers are supported on hardware");
            outElemty.push_back(elemTy);
        }
        auto intergerFuncTy = FunctionType::get(ctx, inElemTy, outElemty);

        // Create new HWModule
        auto ports = getHWPorts(
            OpBuilder(ctx),
            intergerFuncTy,
            /*hasClose*/ true,
            /*hasDone*/ true);
        auto hwModule = rewriter.create<hw::HWModuleOp>(
            op.getLoc(),
            op.getSymNameAttr(),
            hw::ModulePortInfo(ports),
            ArrayAttr{},
            ArrayRef<NamedAttribute>{},
            StringAttr{},
            false);
        auto hwModuleLoc = hwModule.getLoc();
        Region &hwModuleRegion = hwModule.getBody();
        rewriter.setInsertionPointToStart(&hwModuleRegion.front());
        auto [inReadySignals, outValidDataSignals, replacePorts] =
            getReplaceValuesFromCasts(
                OpBuilder(ctx),
                hwModule,
                intergerFuncTy,
                true,
                args);

        // Insert the ops here and replace values
        llvm::DenseMap<Value, Value> argumentMap, resultMap;
        for (size_t i = 0; i < args.size(); i++)
            argumentMap[args[i]] = replacePorts[i];
        for (auto &opi : op.getBody().getOps()) {
            Operation* newOp = rewriter.clone(opi);
            for (size_t i = 0; i < newOp->getResults().size(); i++)
                resultMap[opi.getResult(i)] = newOp->getResult(i);
            for (size_t i = 0; i < newOp->getOperands().size(); i++) {
                auto operand = newOp->getOperand(i);
                if (argumentMap.count(operand))
                    newOp->setOperand(i, argumentMap[operand]);
                if (resultMap.count(operand))
                    newOp->setOperand(i, resultMap[operand]);
            }
        }

        // Insert wait output channels here
        SmallVector<Value> waitOutputs;
        for (auto map : argumentMap) {
            auto outChan = map.second;
            if (dyn_cast<InputType>(outChan.getType())) {
                auto user = *outChan.getUsers().begin();
                if (auto instantiateOp = dyn_cast<InstantiateOp>(user)
                                             ?: dyn_cast<EmbedOp>(user)) {
                    auto placeholder =
                        rewriter.create<hw::ConstantOp>(hwModuleLoc, i1Ty, 0)
                            .getResult();
                    waitOutputs.push_back(placeholder);
                    regionOutputToInstanceOutput.push_back(
                        std::make_pair(outChan, placeholder));
                } else {
                    waitOutputs.push_back(outChan);
                }
            }
        }
        auto waitOp = rewriter.create<HWWaitOp>(hwModuleLoc, i1Ty, waitOutputs);
        // OutputOp
        SmallVector<Value> outputs;
        outputs.append(inReadySignals.begin(), inReadySignals.end());
        outputs.append(outValidDataSignals.begin(), outValidDataSignals.end());
        outputs.push_back(waitOp.getResult());
        rewriter.create<hw::OutputOp>(hwModuleLoc, outputs);

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerLoopToLogic : OpConversionPattern<HWLoopOp> {
    using OpConversionPattern<HWLoopOp>::OpConversionPattern;

    LowerLoopToLogic(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<HWLoopOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        HWLoopOp op,
        HWLoopOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto operation = op.getOperation();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto i1Ty = rewriter.getI1Type();
        BackedgeBuilder bb(rewriter, loc);

        // Get the clock and reset signal from parent module
        auto hwModule = operation->getParentOfType<hw::HWModuleOp>();
        auto clock = hwModule.getBody().getArgument(0);
        auto reset = hwModule.getBody().getArgument(1);
        // Replace reset signal usage with a later defined restart value in this
        // module. The reset signal should only be used once in the
        // instantiation of the kernel module
        Backedge placeholderRestart = bb.get(i1Ty);
        reset.replaceAllUsesWith(placeholderRestart);

        // Get close signals
        SmallVector<Value> close;
        for (auto channel : op.getOperands()) {
            auto castOp = channel.getDefiningOp<UnrealizedConversionCastOp>();
            if (!castOp)
                return rewriter.notifyMatchFailure(
                    loc,
                    "Cannot find the defining operation of this channel in the "
                    "loop.");
            close.push_back(castOp.getOperands().back());
        }
        // Get the done signal of the instantiated kernel module, which is the
        // only comb::AndOp in this module
        auto resultUser = *op.getResult().getUsers().begin();
        auto done = cast<comb::AndOp>(resultUser).getOperand(0);

        // true and false constants
        auto cstTrue = rewriter.create<hw::ConstantOp>(loc, i1Ty, 1);
        auto cstFalse = rewriter.create<hw::ConstantOp>(loc, i1Ty, 0);
        // register for restart signal
        auto restart = rewriter.create<sv::RegOp>(
            loc,
            i1Ty,
            StringAttr::get(ctx, "restart"));
        auto restartValue =
            rewriter.create<sv::ReadInOutOp>(loc, restart.getResult());
        auto notRestart = rewriter.create<comb::XorOp>(
            loc,
            restartValue.getResult(),
            cstTrue.getResult());
        auto shouldRestart =
            rewriter.create<comb::AndOp>(loc, done, notRestart.getResult());
        // Merge the close signals to one
        Value wantClose;
        for (size_t i = 0; i < close.size(); i++) {
            if (i == 0)
                wantClose = close[i];
            else {
                auto closeResult =
                    rewriter.create<comb::OrOp>(loc, wantClose, close[i]);
                wantClose = closeResult.getResult();
            }
        }
        op.getResult().replaceAllUsesWith(wantClose);
        auto notWantClose =
            rewriter.create<comb::XorOp>(loc, wantClose, cstTrue.getResult());
        auto reboot =
            rewriter.create<comb::OrOp>(loc, reset, restartValue.getResult());
        auto stopReboot = rewriter.create<comb::AndOp>(
            loc,
            restartValue.getResult(),
            notWantClose.getResult());
        placeholderRestart.setValue(reboot.getResult());
        // Create the always block to update restart reg
        rewriter
            .create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clock, [&] {
                rewriter.create<sv::IfOp>(loc, reset, [&] {
                    rewriter.create<sv::PAssignOp>(
                        loc,
                        restart.getResult(),
                        cstFalse.getResult());
                });
                rewriter.create<sv::IfOp>(
                    loc,
                    shouldRestart.getResult(),
                    [&] {
                        rewriter.create<sv::PAssignOp>(
                            loc,
                            restart.getResult(),
                            cstTrue.getResult());
                    },
                    [&] {
                        rewriter.create<sv::IfOp>(
                            loc,
                            stopReboot.getResult(),
                            [&] {
                                rewriter.create<sv::PAssignOp>(
                                    loc,
                                    restart.getResult(),
                                    cstFalse.getResult());
                            });
                    });
            });

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerPullToInstance : OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    LowerPullToInstance(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto operation = op.getOperation();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // Get the pull_channel module
        auto module = operation->getParentOfType<ModuleOp>();
        std::optional<hw::HWModuleOp> pullModule = std::nullopt;
        module.walk([&](hw::HWModuleOp hwModuleOp) {
            if (hwModuleOp.getSymNameAttr().str() == "pull_channel")
                pullModule = hwModuleOp;
        });
        if (!pullModule.has_value())
            return rewriter.notifyMatchFailure(
                loc,
                "Cannot find pull channel module.");

        // Inputs of the instance
        SmallVector<Value> inputs;
        // Get the clock and reset signal from parent module
        auto hwModule = operation->getParentOfType<hw::HWModuleOp>();
        inputs.push_back(hwModule.getBody().getArgument(0));
        inputs.push_back(hwModule.getBody().getArgument(1));

        // Get the input channel valid and data port
        auto castOp = op.getChan().getDefiningOp();
        inputs.push_back(castOp->getOperand(0));
        auto inData = castOp->getOperand(1);
        inputs.push_back(inData);
        auto dataBitwidth = inData.getType().getIntOrFloatBitWidth();

        // Create an instance to the pull_channel module
        SmallVector<Attribute> params;
        params.push_back(hw::ParamDeclAttr::get(
            "bitwidth",
            rewriter.getI32IntegerAttr(dataBitwidth)));
        auto pullInstance = rewriter.create<hw::InstanceOp>(
            loc,
            pullModule.value().getOperation(),
            StringAttr::get(ctx, "pull_channel"),
            inputs,
            ArrayAttr::get(ctx, params));
        op.getResult().replaceAllUsesWith(pullInstance.getResult(2));
        castOp->getResult(0).replaceAllUsesWith(pullInstance.getResult(1));

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerIterArgsConstants : OpConversionPattern<arith::ConstantOp> {
    using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

    LowerIterArgsConstants(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::ConstantOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        arith::ConstantOp op,
        arith::ConstantOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto constant = op.getResult();
        auto constantAttr = dyn_cast<IntegerAttr>(op.getValue());

        if (!constantAttr)
            return rewriter.notifyMatchFailure(
                loc,
                "Only integer constants are supported for hardware.");

        Value newConstant;
        auto originalConstant =
            rewriter.create<hw::ConstantOp>(loc, constantAttr.getValue());
        if (isInSmallVector(constant, processIterArgs)) {
            // If the constant is an iter_arg, replace it with a register, later
            // will be updated by a yield
            auto idx = getVectorIdx(constant, processIterArgs).value();
            auto newConstantReg = rewriter.create<sv::RegOp>(
                loc,
                constantAttr.getType(),
                rewriter.getStringAttr("iter_arg" + std::to_string(idx)),
                hw::InnerSymAttr{},
                originalConstant.getResult());
            auto newConstantRegValue = rewriter.create<sv::ReadInOutOp>(
                loc,
                newConstantReg.getResult());
            newConstant = newConstantRegValue.getResult();
            // Update the vector that store all iter_args
            for (auto &v : processIterArgs)
                if (v == constant) v = newConstantReg.getResult();
        } else {
            // Else, it's just an integer to be used somewhere, replace it with
            // a hw.constant
            newConstant = originalConstant.getResult();
        }
        // Replace uses of old constant with the new one
        constant.replaceAllUsesWith(newConstant);

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerIterArgsYield : OpConversionPattern<YieldOp> {
    using OpConversionPattern<YieldOp>::OpConversionPattern;

    LowerIterArgsYield(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<YieldOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        YieldOp op,
        YieldOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto operation = op.getOperation();

        // Get the clock and reset signal from parent module
        auto hwModule = operation->getParentOfType<hw::HWModuleOp>();
        auto clock = hwModule.getBody().getArgument(0);
        auto reset = hwModule.getBody().getArgument(1);
        auto cstTrue =
            rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
        auto notReset =
            rewriter.create<comb::XorOp>(loc, reset, cstTrue.getResult());

        // Create the updating always block
        rewriter.create<
            sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clock, [&] {
            // For each iter_arg, create an update
            for (auto [iter_arg, update_value] :
                 llvm::zip(processIterArgs, op.getOperands())) {
                auto valid = update_value.getDefiningOp()->getResult(0);
                auto allowUpdate = rewriter.create<comb::AndOp>(
                    loc,
                    valid,
                    notReset.getResult());
                rewriter.create<sv::IfOp>(loc, allowUpdate.getResult(), [&] {
                    rewriter.create<sv::PAssignOp>(loc, iter_arg, update_value);
                });
            }
        });

        rewriter.eraseOp(op);
        return success();
    }
};

template<typename OpFrom, typename OpTo>
struct CreateAndInstanceArithOps : OpConversionPattern<OpFrom> {
    using OpConversionPattern<OpFrom>::OpConversionPattern;

    CreateAndInstanceArithOps(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpFrom op,
        OpFrom::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto i1Ty = rewriter.getI1Type();
        auto i32Ty = rewriter.getI32Type();
        auto opName = op.getOperationName().str();
        std::replace(opName.begin(), opName.end(), '.', '_');
        auto opOperandType = op.getOperands().back().getType();
        auto opResult = op.getResult();
        bool isSelectOp = isa<arith::SelectOp>(op);
        bool isCompareOp = isa<arith::CmpIOp>(op);
        comb::ICmpPredicate predicate;
        auto hwParentModule =
            op.getOperation()->template getParentOfType<hw::HWModuleOp>();
        if (isCompareOp) {
            auto cmpOp = dyn_cast<arith::CmpIOp>(op.getOperation());
            switch (cmpOp.getPredicate()) {
            case arith::CmpIPredicate::eq:
                predicate = comb::ICmpPredicate::eq;
                opName += "_eq";
                break;
            case arith::CmpIPredicate::ne:
                predicate = comb::ICmpPredicate::ne;
                opName += "_ne";
                break;
            case arith::CmpIPredicate::slt:
                predicate = comb::ICmpPredicate::slt;
                opName += "_slt";
                break;
            case arith::CmpIPredicate::ult:
                predicate = comb::ICmpPredicate::ult;
                opName += "_ult";
                break;
            case arith::CmpIPredicate::sle:
                predicate = comb::ICmpPredicate::sle;
                opName += "_sle";
                break;
            case arith::CmpIPredicate::ule:
                predicate = comb::ICmpPredicate::ule;
                opName += "_ule";
                break;
            case arith::CmpIPredicate::sgt:
                predicate = comb::ICmpPredicate::sgt;
                opName += "_sgt";
                break;
            case arith::CmpIPredicate::ugt:
                predicate = comb::ICmpPredicate::ugt;
                opName += "_ugt";
                break;
            case arith::CmpIPredicate::sge:
                predicate = comb::ICmpPredicate::sge;
                opName += "_sge";
                break;
            case arith::CmpIPredicate::uge:
                predicate = comb::ICmpPredicate::uge;
                opName += "_uge";
                break;
            }
        }

        // Check if the module is already created
        hw::HWModuleOp hwModule;
        for (auto arithModule : arithModules) {
            auto name = arithModule.getSymName().str();
            if (name == opName) hwModule = arithModule;
        }
        // If the hw.module is not created yet, create and instance it
        if (hwModule == nullptr) {
            auto inDir = hw::ModulePort::Direction::Input;
            auto outDir = hw::ModulePort::Direction::Output;
            SmallVector<Attribute> params;
            params.push_back(hw::ParamDeclAttr::get("bitwidth", i32Ty));
            auto dataParamAttr = hw::ParamDeclRefAttr::get(
                rewriter.getStringAttr("bitwidth"),
                i32Ty);
            auto dataParamTy = hw::IntType::get(dataParamAttr);
            SmallVector<hw::PortInfo> ports;
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("reset"), i1Ty, inDir}
            });
            for (size_t i = 0; i < op.getOperands().size(); i++) {
                auto selectSignal = (i == 0) && isSelectOp;
                auto namePrefix = "in" + std::to_string(i);
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix + "_valid"),
                     i1Ty, inDir}
                });
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr(namePrefix),
                     selectSignal ? i1Ty : dataParamTy,
                     inDir}
                });
            }
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("out_valid"), i1Ty, outDir}
            });
            if (isCompareOp) {
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr("out"), i1Ty, outDir}
                });
            } else {
                ports.push_back(hw::PortInfo{
                    {rewriter.getStringAttr("out"), dataParamTy, outDir}
                });
            }
            rewriter.setInsertionPoint(hwParentModule);
            hwModule = rewriter.create<hw::HWModuleOp>(
                hwParentModule.getLoc(),
                rewriter.getStringAttr(opName),
                hw::ModulePortInfo(ports),
                rewriter.getArrayAttr(params),
                ArrayRef<NamedAttribute>{},
                StringAttr{},
                false);
            auto hwModuleLoc = hwModule.getLoc();
            Region &hwModuleRegion = hwModule.getBody();
            SmallVector<Value> newValids, newOperands;
            for (size_t i = 1; i < hwModuleRegion.getArguments().size();
                 i += 2) {
                newValids.push_back(hwModuleRegion.getArgument(i));
                newOperands.push_back(hwModuleRegion.getArgument(i + 1));
            }
            rewriter.setInsertionPointToStart(&hwModuleRegion.front());
            auto cstFalse =
                rewriter.create<hw::ConstantOp>(hwModuleLoc, i1Ty, 0);
            auto cstTrue =
                rewriter.create<hw::ConstantOp>(hwModuleLoc, i1Ty, 1);
            auto validInput =
                rewriter.create<comb::AndOp>(hwModuleLoc, i1Ty, newValids);
            Value dataInput;
            if (isCompareOp) {
                dataInput = rewriter
                                .create<comb::ICmpOp>(
                                    hwModuleLoc,
                                    i1Ty,
                                    predicate,
                                    newOperands[0],
                                    newOperands[1])
                                .getResult();
            } else {
                dataInput =
                    rewriter.create<OpTo>(hwModuleLoc, dataParamTy, newOperands)
                        .getResult();
            }
            auto regValid = rewriter.create<sv::RegOp>(
                hwModuleLoc,
                i1Ty,
                rewriter.getStringAttr("valid_reg"));
            auto regValidValue = rewriter.create<sv::ReadInOutOp>(
                hwModuleLoc,
                regValid.getResult());
            sv::RegOp regData;
            if (isCompareOp) {
                regData = rewriter.create<sv::RegOp>(
                    hwModuleLoc,
                    i1Ty,
                    rewriter.getStringAttr("data_reg"));
            } else {
                regData = rewriter.create<sv::RegOp>(
                    hwModuleLoc,
                    dataParamTy,
                    rewriter.getStringAttr("data_reg"));
            }
            auto regDataValue = rewriter.create<sv::ReadInOutOp>(
                hwModuleLoc,
                regData.getResult());
            auto notValidOnce = rewriter.create<comb::XorOp>(
                hwModuleLoc,
                regValidValue.getResult(),
                cstTrue.getResult());
            auto updateReg = rewriter.create<comb::AndOp>(
                hwModuleLoc,
                notValidOnce.getResult(),
                validInput.getResult());
            rewriter.create<sv::AlwaysCombOp>(hwModuleLoc, [&] {
                rewriter.create<sv::IfOp>(
                    hwModuleLoc,
                    hwModuleRegion.getArgument(0),
                    [&] {
                        rewriter.create<sv::PAssignOp>(
                            hwModuleLoc,
                            regValid.getResult(),
                            cstFalse.getResult());
                    },
                    [&] {
                        rewriter.create<sv::IfOp>(
                            hwModuleLoc,
                            updateReg.getResult(),
                            [&] {
                                rewriter.create<sv::PAssignOp>(
                                    hwModuleLoc,
                                    regValid.getResult(),
                                    cstTrue.getResult());
                                rewriter.create<sv::PAssignOp>(
                                    hwModuleLoc,
                                    regData.getResult(),
                                    dataInput);
                            });
                    });
            });
            rewriter.create<hw::OutputOp>(
                hwModuleLoc,
                ValueRange{
                    regValidValue.getResult(),
                    regDataValue.getResult()});
            arithModules.push_back(hwModule);
        }

        // The inputs of the instance to the module should be the value and the
        // valid signal, which is defined by the same operaiton that defines the
        // value
        SmallVector<Value> inputs;
        inputs.push_back(hwParentModule.getRegion().getArgument(1));
        for (auto operand : op.getOperands()) {
            if (auto definition =
                    operand.template getDefiningOp<hw::InstanceOp>()) {
                inputs.push_back(definition.getResult(0));
                inputs.push_back(operand);
            } else if (
                auto definition =
                    operand
                        .template getDefiningOp<UnrealizedConversionCastOp>()) {
                inputs.push_back(definition.getOperand(0));
                inputs.push_back(definition.getOperand(1));
            }
        }
        // Create the instance to the hw.module
        rewriter.setInsertionPoint(op);
        auto bitwidth = opOperandType.getIntOrFloatBitWidth();
        SmallVector<Attribute> instanceParams;
        instanceParams.push_back(hw::ParamDeclAttr::get(
            "bitwidth",
            rewriter.getI32IntegerAttr(bitwidth)));
        auto instance = rewriter.create<hw::InstanceOp>(
            loc,
            hwModule.getOperation(),
            rewriter.getStringAttr(opName + "_i" + std::to_string(bitwidth)),
            inputs,
            rewriter.getArrayAttr(instanceParams));

        // Replace the original result of arith op with the valid and data
        // generated from the instance
        for (auto user : opResult.getUsers())
            // If it's used in wait operation, replace it with the valid signal
            if (isa<HWWaitOp>(user))
                user->replaceUsesOfWith(opResult, instance.getResult(0));
        // Else such as used by push or any other arith ops, replace it with
        // the data
        opResult.replaceAllUsesWith(instance.getResult(1));

        rewriter.eraseOp(op);
        return success();
    }
};

template<typename OpT, bool isSigned, bool isTruncation>
struct CreateAndInstanceArithExtTrunc : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    CreateAndInstanceArithExtTrunc(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<OpT>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpT op,
        OpT::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto i1Ty = rewriter.getI1Type();
        auto opOperand = op.getOperand();
        auto opResult = op.getResult();
        auto bitwidthIn = opOperand.getType().getIntOrFloatBitWidth();
        auto bitwidthOut = opResult.getType().getIntOrFloatBitWidth();
        auto dataTyIn = opOperand.getType();
        auto dataTyOut = opResult.getType();
        auto opName = op.getOperationName().str();
        std::replace(opName.begin(), opName.end(), '.', '_');
        opName += "_i" + std::to_string(bitwidthIn) + "_to_i"
                  + std::to_string(bitwidthOut);
        auto hwParentModule =
            op.getOperation()->template getParentOfType<hw::HWModuleOp>();

        // Check if the module is already created
        hw::HWModuleOp hwModule;
        for (auto arithModule : arithModules) {
            auto name = arithModule.getSymName().str();
            if (name == opName) hwModule = arithModule;
        }
        // If the hw.module is not created yet, create and instance it
        if (hwModule == nullptr) {
            auto inDir = hw::ModulePort::Direction::Input;
            auto outDir = hw::ModulePort::Direction::Output;
            SmallVector<hw::PortInfo> ports;
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("reset"), i1Ty, inDir}
            });
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("in_valid"), i1Ty, inDir}
            });
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("in"), dataTyIn, inDir}
            });
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("out_valid"), i1Ty, outDir}
            });
            ports.push_back(hw::PortInfo{
                {rewriter.getStringAttr("out"), dataTyOut, outDir}
            });
            rewriter.setInsertionPoint(hwParentModule);
            hwModule = rewriter.create<hw::HWModuleOp>(
                hwParentModule.getLoc(),
                rewriter.getStringAttr(opName),
                hw::ModulePortInfo(ports),
                ArrayAttr{},
                ArrayRef<NamedAttribute>{},
                StringAttr{},
                false);
            auto hwModuleLoc = hwModule.getLoc();
            Region &hwModuleRegion = hwModule.getBody();
            auto valid = hwModuleRegion.getArgument(1);
            auto data = hwModuleRegion.getArgument(2);
            rewriter.setInsertionPointToStart(&hwModuleRegion.front());
            auto cstFalse =
                rewriter.create<hw::ConstantOp>(hwModuleLoc, i1Ty, 0);
            auto cstTrue =
                rewriter.create<hw::ConstantOp>(hwModuleLoc, i1Ty, 1);
            Value outputValue;
            if (isTruncation) {
                // If it's arith.trunc, replace it with comb.extract
                auto extractOp = rewriter.create<comb::ExtractOp>(
                    hwModuleLoc,
                    dataTyOut,
                    data,
                    0);
                outputValue = extractOp.getResult();
            } else if (isSigned) {
                // If it's signed, extract the sign bit and replicate it
                // (bitwidth_out - bitwidth_in) times then concat
                auto signBit = rewriter.create<comb::ExtractOp>(
                    hwModuleLoc,
                    i1Ty,
                    data,
                    bitwidthIn - 1);
                auto replicateOp = rewriter.create<comb::ReplicateOp>(
                    hwModuleLoc,
                    signBit.getResult(),
                    bitwidthOut - bitwidthIn);
                auto concatOp = rewriter.create<comb::ConcatOp>(
                    hwModuleLoc,
                    replicateOp.getResult(),
                    data);
                outputValue = concatOp.getResult();
            } else {
                // If it's unsigned, use comb.concat directly
                auto dataTyDiff =
                    rewriter.getIntegerType(bitwidthOut - bitwidthIn);
                auto cstDiff =
                    rewriter.create<hw::ConstantOp>(hwModuleLoc, dataTyDiff, 0);
                auto concatOp = rewriter.create<comb::ConcatOp>(
                    hwModuleLoc,
                    cstDiff.getResult(),
                    data);
                outputValue = concatOp.getResult();
            }
            auto regValid = rewriter.create<sv::RegOp>(
                hwModuleLoc,
                i1Ty,
                rewriter.getStringAttr("valid_reg"));
            auto regValidValue = rewriter.create<sv::ReadInOutOp>(
                hwModuleLoc,
                regValid.getResult());
            auto regData = rewriter.create<sv::RegOp>(
                hwModuleLoc,
                dataTyOut,
                rewriter.getStringAttr("data_reg"));
            auto regDataValue = rewriter.create<sv::ReadInOutOp>(
                hwModuleLoc,
                regData.getResult());
            auto notValidOnce = rewriter.create<comb::XorOp>(
                hwModuleLoc,
                regValidValue.getResult(),
                cstTrue.getResult());
            auto updateReg = rewriter.create<comb::AndOp>(
                hwModuleLoc,
                notValidOnce.getResult(),
                valid);
            rewriter.create<sv::AlwaysCombOp>(hwModuleLoc, [&] {
                rewriter.create<sv::IfOp>(
                    hwModuleLoc,
                    hwModuleRegion.getArgument(0),
                    [&] {
                        rewriter.create<sv::PAssignOp>(
                            hwModuleLoc,
                            regValid.getResult(),
                            cstFalse.getResult());
                    },
                    [&] {
                        rewriter.create<sv::IfOp>(
                            hwModuleLoc,
                            updateReg.getResult(),
                            [&] {
                                rewriter.create<sv::PAssignOp>(
                                    hwModuleLoc,
                                    regValid.getResult(),
                                    cstTrue.getResult());
                                rewriter.create<sv::PAssignOp>(
                                    hwModuleLoc,
                                    regData.getResult(),
                                    outputValue);
                            });
                    });
            });
            rewriter.create<hw::OutputOp>(
                hwModuleLoc,
                ValueRange{
                    regValidValue.getResult(),
                    regDataValue.getResult()});
            arithModules.push_back(hwModule);
        }

        // The inputs of the instance to the module should be the value and the
        // valid signal, which is defined by the same operaiton that defines the
        // value
        SmallVector<Value> inputs;
        inputs.push_back(hwParentModule.getRegion().getArgument(1));
        if (auto definition =
                opOperand.template getDefiningOp<hw::InstanceOp>()) {
            inputs.push_back(definition.getResult(0));
            inputs.push_back(opOperand);
        } else if (
            auto definition =
                opOperand
                    .template getDefiningOp<UnrealizedConversionCastOp>()) {
            inputs.push_back(definition.getOperand(0));
            inputs.push_back(definition.getOperand(1));
        }
        // Create the instance to the hw.module
        rewriter.setInsertionPoint(op);
        auto instance = rewriter.create<hw::InstanceOp>(
            loc,
            hwModule.getOperation(),
            rewriter.getStringAttr(opName),
            inputs);

        // Replace the original result of arith op with the valid and data
        // generated from the instance
        for (auto user : opResult.getUsers())
            // If it's used in wait operation, replace it with the valid signal
            if (isa<HWWaitOp>(user))
                user->replaceUsesOfWith(opResult, instance.getResult(0));
        // Else such as used by push or any other arith ops, replace it with
        // the data
        opResult.replaceAllUsesWith(instance.getResult(1));

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerPushToInstance : OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    LowerPushToInstance(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        PushOp op,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto operation = op.getOperation();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // Get the pull_channel module
        auto module = operation->getParentOfType<ModuleOp>();
        std::optional<hw::HWModuleOp> pushModule = std::nullopt;
        module.walk([&](hw::HWModuleOp hwModuleOp) {
            if (hwModuleOp.getSymNameAttr().str() == "push_channel")
                pushModule = hwModuleOp;
        });
        if (!pushModule.has_value())
            return rewriter.notifyMatchFailure(
                loc,
                "Cannot find push channel module.");

        // Inputs of the instance
        SmallVector<Value> inputs;
        // Get the clock and reset signal from parent module
        auto hwModule = operation->getParentOfType<hw::HWModuleOp>();
        inputs.push_back(hwModule.getBody().getArgument(0));
        inputs.push_back(hwModule.getBody().getArgument(1));

        // Get the input channel ready port
        auto data = op.getInp();
        auto castOp = op.getChan().getDefiningOp();
        auto ready = castOp->getOperand(0);
        auto valid = data.getDefiningOp()->getResult(0);
        auto validInput = rewriter.create<comb::AndOp>(loc, valid, ready);
        inputs.push_back(validInput.getResult());
        auto dataBitwidth = data.getType().getIntOrFloatBitWidth();
        inputs.push_back(data);

        // Create an instance to the push_channel module
        SmallVector<Attribute> params;
        params.push_back(hw::ParamDeclAttr::get(
            "bitwidth",
            rewriter.getI32IntegerAttr(dataBitwidth)));
        auto pullInstance = rewriter.create<hw::InstanceOp>(
            loc,
            pushModule.value().getOperation(),
            StringAttr::get(ctx, "push_channel"),
            inputs,
            ArrayAttr::get(ctx, params));
        castOp->getResult(0).replaceAllUsesWith(pullInstance.getResult(1));
        castOp->getResult(1).replaceAllUsesWith(pullInstance.getResult(2));

        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerChannelToInstance : OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    LowerChannelToInstance(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptop,
        ConversionPatternRewriter &rewriter) const override
    {
        auto operation = op.getOperation();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto channelInputPort = op.getInChan();
        auto channelOutputPort = op.getOutChan();
        auto i1Ty = rewriter.getI1Type();

        // Get the queue module
        auto module = operation->getParentOfType<ModuleOp>();
        std::optional<hw::HWModuleOp> queueModule = std::nullopt;
        module.walk([&](hw::HWModuleOp hwModuleOp) {
            if (hwModuleOp.getSymNameAttr().str() == "queue")
                queueModule = hwModuleOp;
        });
        if (!queueModule.has_value())
            return rewriter.notifyMatchFailure(
                loc,
                "Cannot find queue module.");

        SmallVector<Attribute> params;
        auto channelType = op.getEncapsulatedType();
        params.push_back(hw::ParamDeclAttr::get(
            "bitwidth",
            rewriter.getI32IntegerAttr(channelType.getIntOrFloatBitWidth())));
        auto size = op.getBufferSize();
        if (!size.has_value() || size.value() == 0)
            return rewriter.notifyMatchFailure(
                loc,
                "Size-0 channel will be supported later.");
        params.push_back(hw::ParamDeclAttr::get(
            "size",
            rewriter.getI32IntegerAttr(size.value())));

        // Inputs of the instance
        SmallVector<Value> inputs;
        // Get the clock and reset signal from parent module
        auto hwModule = operation->getParentOfType<hw::HWModuleOp>();
        inputs.push_back(hwModule.getBody().getArgument(0));
        inputs.push_back(hwModule.getBody().getArgument(1));

        auto channelInputUser = *channelInputPort.getUsers().begin();
        // If the channel's input port is connected to the region, connect
        // to the valid, data and close signals
        if (auto connectOp = dyn_cast<ConnectInputOp>(channelInputUser)) {
            auto castOp = dyn_cast<UnrealizedConversionCastOp>(
                connectOp.getRegionPort().getDefiningOp());
            inputs.push_back(castOp.getOperand(0));
            inputs.push_back(castOp.getOperand(1));
            inputs.push_back(castOp.getOperand(2));
            rewriter.eraseOp(connectOp);
        }
        // If the channel's input port is connected to the instance, create the
        // backedge for the done/valid/data signal and store it in the helper
        else if (
            auto instantiateOp = dyn_cast<InstantiateOp>(channelInputUser)
                                     ?: dyn_cast<EmbedOp>(channelInputUser)) {
            SmallVector<Value> instanceOutputs;
            auto placeholderInstanceValid =
                rewriter.create<hw::ConstantOp>(loc, i1Ty, 0).getResult();
            inputs.push_back(placeholderInstanceValid);
            instanceOutputs.push_back(placeholderInstanceValid);
            auto placeholderInstanceData =
                rewriter.create<hw::ConstantOp>(loc, channelType, 0)
                    .getResult();
            inputs.push_back(placeholderInstanceData);
            instanceOutputs.push_back(placeholderInstanceData);
            auto placeholderInstanceDone =
                rewriter.create<hw::ConstantOp>(loc, i1Ty, 0).getResult();
            inputs.push_back(placeholderInstanceDone);
            instanceOutputs.push_back(placeholderInstanceDone);
            channelInputToInstanceOutput.push_back(
                std::make_pair(channelInputPort, instanceOutputs));
        }

        auto channelOutputUser = *channelOutputPort.getUsers().begin();
        // If the channel's output port is connected to the region, connect
        // to the valid, data and close signals
        if (auto connectOp = dyn_cast<ConnectOutputOp>(channelOutputUser)) {
            auto castOp = dyn_cast<UnrealizedConversionCastOp>(
                connectOp.getRegionPort().getDefiningOp());
            inputs.push_back(castOp.getOperand(0));
            rewriter.eraseOp(connectOp);
        }
        // If the channel's output port is connected to the instance, create the
        // backedge for the close signal and store it in the helper
        else if (
            auto instantiateOp = dyn_cast<InstantiateOp>(channelOutputUser)
                                     ?: dyn_cast<EmbedOp>(channelOutputUser)) {
            auto placeholderInstanceReady =
                rewriter.create<hw::ConstantOp>(loc, i1Ty, 0).getResult();
            inputs.push_back(placeholderInstanceReady);
            channelOutputToInstanceInput.push_back(
                std::make_pair(channelOutputPort, placeholderInstanceReady));
        }

        // Create the instance to the queue module
        auto queueInstance = rewriter.create<hw::InstanceOp>(
            loc,
            queueModule.value().getOperation(),
            StringAttr::get(ctx, "queue"),
            inputs,
            ArrayAttr::get(ctx, params));

        // If the channel's input port is connected to the region, replace the
        // output ready signal with the generated one from this instance
        if (auto connectOp = dyn_cast<ConnectInputOp>(channelInputUser)) {
            auto castOp = dyn_cast<UnrealizedConversionCastOp>(
                connectOp.getRegionPort().getDefiningOp());
            castOp.getResult(0).replaceAllUsesWith(queueInstance.getResult(0));
        }
        // If the channel's input port is connected to the isntance, create an
        // unrealized cast and update the usage of input port value
        else if (
            auto instantiateOp = dyn_cast<InstantiateOp>(channelInputUser)
                                     ?: dyn_cast<EmbedOp>(channelInputUser)) {
            auto castOp = rewriter.create<UnrealizedConversionCastOp>(
                loc,
                TypeRange{InputType::get(ctx, channelType)},
                ValueRange{queueInstance.getResult(0)});
            auto newChannelInputPort = castOp.getResult(0);
            // Update the content of helper
            for (auto &pair : channelInputToInstanceOutput)
                if (pair.first == channelInputPort)
                    pair.first = newChannelInputPort;
            channelInputPort.replaceAllUsesWith(newChannelInputPort);
        }
        // If the channel's output port is connected to the region, replace the
        // output valid and data signals with the genereated one from this
        // instance
        if (auto connectOp = dyn_cast<ConnectOutputOp>(channelOutputUser)) {
            auto castOp = dyn_cast<UnrealizedConversionCastOp>(
                connectOp.getRegionPort().getDefiningOp());
            castOp.getResult(0).replaceAllUsesWith(queueInstance.getResult(1));
            castOp.getResult(1).replaceAllUsesWith(queueInstance.getResult(2));
            // If the output port is connected to the region, it must be also
            // waited in HWWaitOp, so replace the waited channel value to the
            // done signal from this instance
            castOp.getResult(2).replaceAllUsesWith(queueInstance.getResult(3));
        }
        // If the channel's output port is connected to the instance
        else if (
            auto instantiateOp = dyn_cast<InstantiateOp>(channelOutputUser)
                                     ?: dyn_cast<EmbedOp>(channelOutputUser)) {
            auto castOp = rewriter.create<UnrealizedConversionCastOp>(
                loc,
                TypeRange{OutputType::get(ctx, channelType)},
                ValueRange{
                    queueInstance.getResult(1),
                    queueInstance.getResult(2),
                    queueInstance.getResult(3)});
            auto newChannelOutputPort = castOp.getResult(0);
            // Update the content of helper
            for (auto &pair : channelOutputToInstanceInput)
                if (pair.first == channelOutputPort)
                    pair.first = newChannelOutputPort;
            channelOutputPort.replaceAllUsesWith(newChannelOutputPort);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

template<typename OpT>
struct LowerInstantiateOrEmbed : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    LowerInstantiateOrEmbed(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpT>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpT op,
        OpT::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto operation = op.getOperation();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto calleeName = op.getCallee().getRootReference().str();
        auto numInputs =
            std::distance(op.getInputs().begin(), op.getInputs().end());
        auto numOutputs =
            std::distance(op.getOutputs().begin(), op.getOutputs().end());

        // Get the instantiated module
        auto module = operation->template getParentOfType<ModuleOp>();
        std::optional<hw::HWModuleOp> instaceModule = std::nullopt;
        module.walk([&](hw::HWModuleOp hwModuleOp) {
            if (hwModuleOp.getSymNameAttr().str() == calleeName)
                instaceModule = hwModuleOp;
        });
        if (!instaceModule.has_value())
            return rewriter.notifyMatchFailure(
                loc,
                "Cannot find module to be instantiated.");

        SmallVector<Value> inputs, replaceValue, replaceWait;
        // Get the clock and reset signal from parent module
        auto hwModule = operation->template getParentOfType<hw::HWModuleOp>();
        inputs.push_back(hwModule.getBody().getArgument(0));
        inputs.push_back(hwModule.getBody().getArgument(1));
        // Get valid, data, close inputs from input channels
        for (auto inChan : op.getInputs()) {
            auto castOp =
                dyn_cast<UnrealizedConversionCastOp>(inChan.getDefiningOp());
            inputs.append(
                castOp.getOperands().begin(),
                castOp.getOperands().end());
            for (auto &pair : channelOutputToInstanceInput)
                if (pair.first == inChan) replaceValue.push_back(pair.second);
        }
        // Get ready inputs from output channels
        for (auto outChan : op.getOutputs()) {
            auto castOp =
                dyn_cast<UnrealizedConversionCastOp>(outChan.getDefiningOp());
            inputs.append(
                castOp.getOperands().begin(),
                castOp.getOperands().end());
            for (auto &pair : channelInputToInstanceOutput)
                if (pair.first == outChan)
                    replaceValue.append(pair.second.begin(), pair.second.end());
            for (auto &pair : regionOutputToInstanceOutput)
                if (pair.first == outChan) replaceWait.push_back(pair.second);
        }

        // Create the instance
        auto instance = rewriter.create<hw::InstanceOp>(
            loc,
            instaceModule.value().getOperation(),
            StringAttr::get(ctx, calleeName),
            inputs);

        // Replace the value with the actual connected value
        SmallVector<Value> instanceOutputs;
        for (auto i = 0; i < numInputs; i++)
            instanceOutputs.push_back(instance.getResult(i));
        for (auto i = 0; i < numOutputs; i++) {
            instanceOutputs.push_back(instance.getResult(2 * i + numInputs));
            instanceOutputs.push_back(
                instance.getResult(2 * i + numInputs + 1));
            instanceOutputs.push_back(instance.getResults().back());
        }
        for (auto [placeholder, instanceOutput] :
             llvm::zip(replaceValue, instanceOutputs)) {
            placeholder.replaceAllUsesWith(instanceOutput);
            rewriter.eraseOp(placeholder.getDefiningOp());
        }
        for (auto placeholder : replaceWait) {
            placeholder.replaceAllUsesWith(instance.getResults().back());
            rewriter.eraseOp(placeholder.getDefiningOp());
        }
        rewriter.eraseOp(op);
        return success();
    }
};

struct CombineWaitedSignal : OpConversionPattern<HWWaitOp> {
    using OpConversionPattern<HWWaitOp>::OpConversionPattern;

    CombineWaitedSignal(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<HWWaitOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        HWWaitOp op,
        HWWaitOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto operands = op.getOperands();

        // Check if all operands is i1
        for (auto operand : operands)
            if (operand.getType().getIntOrFloatBitWidth() != 1)
                return rewriter.notifyMatchFailure(
                    loc,
                    "Something is wrong and ask yourself.");

        // Replace the usage of the result with the i1 value
        Value replaceValue;
        if (operands.size() == 1)
            replaceValue = op.getOperand(0);
        else {
            auto andOp = rewriter.create<comb::AndOp>(
                loc,
                rewriter.getI1Type(),
                operands);
            replaceValue = andOp.getResult();
        }
        op.getResult().replaceAllUsesWith(replaceValue);

        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

void mlir::populateDfgToHWConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<LowerProcessToHWModule>(typeConverter, patterns.getContext());
    patterns.add<LowerRegionToHWModule>(typeConverter, patterns.getContext());
    patterns.add<LowerLoopToLogic>(typeConverter, patterns.getContext());
    patterns.add<LowerPullToInstance>(typeConverter, patterns.getContext());
    patterns.add<LowerPushToInstance>(typeConverter, patterns.getContext());
    patterns.add<LowerChannelToInstance>(typeConverter, patterns.getContext());
    patterns.add<LowerIterArgsConstants>(typeConverter, patterns.getContext());
    patterns.add<LowerIterArgsYield>(typeConverter, patterns.getContext());
    patterns.add<LowerInstantiateOrEmbed<InstantiateOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<LowerInstantiateOrEmbed<EmbedOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::AddIOp, comb::AddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::SubIOp, comb::SubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::MulIOp, comb::MulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::DivUIOp, comb::DivSOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::DivSIOp, comb::DivUOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::RemUIOp, comb::ModUOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::RemSIOp, comb::ModSOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::AndIOp, comb::AndOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::OrIOp, comb::OrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::XOrIOp, comb::XorOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::ShLIOp, comb::ShlOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::ShRUIOp, comb::ShrUOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::ShRSIOp, comb::ShrSOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::SelectOp, comb::MuxOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithOps<arith::CmpIOp, comb::ICmpOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithExtTrunc<arith::ExtUIOp, false, false>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithExtTrunc<arith::ExtSIOp, true, false>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CreateAndInstanceArithExtTrunc<arith::TruncIOp, false, true>>(
        typeConverter,
        patterns.getContext());
    patterns.add<CombineWaitedSignal>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDfgToHWPass
        : public impl::ConvertDfgToHWBase<ConvertDfgToHWPass> {
    void insertPullPushChannelModule(OpBuilder builder, bool pullOrPush);
    void insertQueueModule(OpBuilder builder);
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToHWPass::insertPullPushChannelModule(
    OpBuilder builder,
    bool pullOrPush)
{
    auto name = pullOrPush ? "push_channel" : "pull_channel";
    auto ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    auto i1Ty = builder.getI1Type();
    auto i32Ty = builder.getI32Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;
    SmallVector<Attribute> params;
    params.push_back(hw::ParamDeclAttr::get("bitwidth", i32Ty));
    auto dataParamAttr =
        hw::ParamDeclRefAttr::get(builder.getStringAttr("bitwidth"), i32Ty);
    auto dataParamTy = hw::IntType::get(dataParamAttr);

    SmallVector<hw::PortInfo> ports;
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("clock"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("reset"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("valid"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in"), dataParamTy, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("finish"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("ready"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out"), dataParamTy, outDir}
    });
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
    auto inValid = hwModule.getBody().getArgument(2);
    auto inData = hwModule.getBody().getArgument(3);
    builder.setInsertionPointToStart(&hwModule.getBodyRegion().front());

    auto cstFalse = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto cstTrue = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto validReg =
        builder.create<sv::RegOp>(loc, i1Ty, StringAttr::get(ctx, "valid_reg"));
    auto validRegValue =
        builder.create<sv::ReadInOutOp>(loc, validReg.getResult());
    auto dataReg = builder.create<sv::RegOp>(
        loc,
        dataParamTy,
        StringAttr::get(ctx, "data_reg"));
    auto dataRegValue =
        builder.create<sv::ReadInOutOp>(loc, dataReg.getResult());
    auto validRegComp = builder.create<comb::XorOp>(
        loc,
        validRegValue.getResult(),
        cstTrue.getResult());
    auto readyOutput = builder.create<comb::MuxOp>(
        loc,
        validRegValue.getResult(),
        cstFalse.getResult(),
        inValid);
    Value dataOutput = dataRegValue.getResult();
    if (pullOrPush)
        dataOutput = builder
                         .create<comb::MuxOp>(
                             loc,
                             validRegValue.getResult(),
                             dataRegValue.getResult(),
                             inData)
                         .getResult();

    builder.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
        builder.create<sv::IfOp>(
            loc,
            rst,
            [&] {
                builder.create<sv::PAssignOp>(
                    loc,
                    validReg.getResult(),
                    cstFalse.getResult());
            },
            [&] {
                builder.create<sv::IfOp>(loc, inValid, [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        validReg.getResult(),
                        cstTrue.getResult());
                    builder.create<sv::IfOp>(
                        loc,
                        validRegComp.getResult(),
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                dataReg.getResult(),
                                inData);
                        });
                });
            });
    });

    builder.create<hw::OutputOp>(
        loc,
        ValueRange{
            validRegValue.getResult(),
            readyOutput.getResult(),
            dataOutput});
}

void ConvertDfgToHWPass::insertQueueModule(OpBuilder builder)
{
    auto loc = builder.getUnknownLoc();
    BackedgeBuilder bb(builder, loc);
    auto ctx = builder.getContext();
    auto i1Ty = builder.getI1Type();
    auto i32Ty = builder.getI32Type();
    auto inDir = hw::ModulePort::Direction::Input;
    auto outDir = hw::ModulePort::Direction::Output;
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
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("clock"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("reset"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in_valid"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in"), dataParamTy, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("close"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out_ready"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in_ready"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out_valid"), i1Ty, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("out"), dataParamTy, outDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("done"), i1Ty, outDir}
    });

    auto hwModule = builder.create<hw::HWModuleOp>(
        loc,
        builder.getStringAttr("queue"),
        hw::ModulePortInfo(ports),
        ArrayAttr::get(builder.getContext(), params),
        ArrayRef<NamedAttribute>{},
        StringAttr{},
        false);
    auto clk = hwModule.getBody().getArgument(0);
    auto rst = hwModule.getBody().getArgument(1);
    auto inValid = hwModule.getBody().getArgument(2);
    auto inData = hwModule.getBody().getArgument(3);
    auto close = hwModule.getBody().getArgument(4);
    auto outReady = hwModule.getBody().getArgument(5);
    builder.setInsertionPointToStart(&hwModule.getBodyRegion().front());

    auto cstTrue = builder.create<hw::ConstantOp>(loc, i1Ty, 1);
    auto cstFalse = builder.create<hw::ConstantOp>(loc, i1Ty, 0);
    auto cstZero = builder.create<hw::ConstantOp>(loc, i32Ty, 0);
    auto cstOne = builder.create<hw::ConstantOp>(loc, i32Ty, 1);
    auto sizeValue =
        builder.create<hw::ParamValueOp>(loc, i32Ty, sizeParamAttr);
    auto sizeMaxIdx = builder.create<comb::SubOp>(
        loc,
        sizeValue.getResult(),
        cstOne.getResult());

    // Close behavior
    auto wantClose = builder.create<sv::RegOp>(
        loc,
        i1Ty,
        StringAttr::get(ctx, "want_close"));
    auto wantCloseValue =
        builder.create<sv::ReadInOutOp>(loc, wantClose.getResult());
    // Memory
    auto ram = builder.create<sv::RegOp>(
        loc,
        hw::UnpackedArrayType::get(
            builder.getContext(),
            dataParamTy,
            sizeParamAttr),
        StringAttr::get(ctx, "ram"));
    Backedge placeholderReadIndex = bb.get(i32Ty);
    auto ramRead = builder.create<sv::ArrayIndexInOutOp>(
        loc,
        ram.getResult(),
        placeholderReadIndex);
    auto ramReadValue =
        builder.create<sv::ReadInOutOp>(loc, ramRead.getResult());

    // Pointers
    auto ptrWrite = builder.create<sv::RegOp>(
        loc,
        i32Ty,
        StringAttr::get(ctx, "ptr_write"));
    auto ptrWriteValue =
        builder.create<sv::ReadInOutOp>(loc, ptrWrite.getResult());
    auto ptrRead =
        builder.create<sv::RegOp>(loc, i32Ty, StringAttr::get(ctx, "ptr_read"));
    auto ptrReadValue =
        builder.create<sv::ReadInOutOp>(loc, ptrRead.getResult());
    placeholderReadIndex.setValue(ptrReadValue.getResult());
    auto maybeFull = builder.create<sv::RegOp>(
        loc,
        i1Ty,
        StringAttr::get(ctx, "maybe_full"));
    auto maybeFullValue =
        builder.create<sv::ReadInOutOp>(loc, maybeFull.getResult());
    auto ptrLast =
        builder.create<sv::RegOp>(loc, i32Ty, StringAttr::get(ctx, "ptr_last"));
    auto ptrLastValue =
        builder.create<sv::ReadInOutOp>(loc, ptrLast.getResult());
    auto isDone =
        builder.create<sv::RegOp>(loc, i1Ty, StringAttr::get(ctx, "is_done"));
    auto isDoneValue = builder.create<sv::ReadInOutOp>(loc, isDone.getResult());

    // Signals
    auto ptrMatch = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::eq,
        ptrWriteValue.getResult(),
        ptrReadValue.getResult());
    auto emptyT = builder.create<comb::XorOp>(
        loc,
        maybeFullValue.getResult(),
        cstTrue.getResult());
    auto empty = builder.create<comb::AndOp>(
        loc,
        ptrMatch.getResult(),
        emptyT.getResult());
    auto full = builder.create<comb::AndOp>(
        loc,
        ptrMatch.getResult(),
        maybeFullValue.getResult());
    Backedge placeholderNotFull = bb.get(i1Ty);
    auto doEnq = builder.create<comb::AndOp>(loc, placeholderNotFull, inValid);
    Backedge placeholderNotEmpty = bb.get(i1Ty);
    auto doDeq =
        builder.create<comb::AndOp>(loc, outReady, placeholderNotEmpty);
    auto nextWrite = builder.create<comb::AddOp>(
        loc,
        ptrWriteValue.getResult(),
        cstOne.getResult());
    auto nextRead = builder.create<comb::AddOp>(
        loc,
        ptrReadValue.getResult(),
        cstOne.getResult());
    auto notSameEnqDeq = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::ne,
        doEnq.getResult(),
        doDeq.getResult());
    auto notEmpty = builder.create<comb::XorOp>(
        loc,
        empty.getResult(),
        cstTrue.getResult());
    placeholderNotEmpty.setValue(notEmpty.getResult());
    auto not_full =
        builder.create<comb::XorOp>(loc, full.getResult(), cstTrue.getResult());
    auto notWantClose = builder.create<comb::XorOp>(
        loc,
        wantCloseValue.getResult(),
        cstTrue.getResult());
    auto inReadyOutput = builder.create<comb::AndOp>(
        loc,
        not_full.getResult(),
        notWantClose.getResult());
    placeholderNotFull.setValue(inReadyOutput.getResult());
    auto last_read = builder.create<comb::ICmpOp>(
        loc,
        comb::ICmpPredicate::eq,
        ptrLastValue.getResult(),
        ptrReadValue.getResult());
    auto lastReadWantClose = builder.create<comb::AndOp>(
        loc,
        wantCloseValue.getResult(),
        last_read.getResult());
    auto lastReadWantCloseDeq = builder.create<comb::AndOp>(
        loc,
        lastReadWantClose.getResult(),
        doDeq.getResult());
    auto updateClose =
        builder.create<comb::AndOp>(loc, close, notWantClose.getResult());
    auto doneOutput = builder.create<comb::OrOp>(
        loc,
        lastReadWantCloseDeq.getResult(),
        isDoneValue.getResult());

    // Clocked logic
    builder.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk, [&] {
        builder.create<sv::IfOp>(
            loc,
            rst,
            [&] {
                builder.create<sv::PAssignOp>(
                    loc,
                    ptrWrite.getResult(),
                    cstZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    ptrRead.getResult(),
                    cstZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    maybeFull.getResult(),
                    cstFalse.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    wantClose.getResult(),
                    cstFalse.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    ptrLast.getResult(),
                    cstZero.getResult());
                builder.create<sv::PAssignOp>(
                    loc,
                    isDone.getResult(),
                    cstFalse.getResult());
            },
            [&] {
                builder.create<sv::IfOp>(loc, doEnq.getResult(), [&] {
                    auto ramWrite = builder.create<sv::ArrayIndexInOutOp>(
                        loc,
                        ram.getResult(),
                        ptrWriteValue.getResult());
                    builder.create<sv::PAssignOp>(
                        loc,
                        ramWrite.getResult(),
                        inData);
                    auto isSizeMax = builder.create<comb::ICmpOp>(
                        loc,
                        comb::ICmpPredicate::eq,
                        ptrWriteValue.getResult(),
                        sizeMaxIdx.getResult());
                    builder.create<sv::IfOp>(
                        loc,
                        isSizeMax.getResult(),
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrWrite.getResult(),
                                cstZero.getResult());
                        },
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrWrite.getResult(),
                                nextWrite.getResult());
                        });
                });
                builder.create<sv::IfOp>(loc, doDeq.getResult(), [&] {
                    auto isSizeMax = builder.create<comb::ICmpOp>(
                        loc,
                        comb::ICmpPredicate::eq,
                        ptrReadValue.getResult(),
                        sizeMaxIdx.getResult());
                    builder.create<sv::IfOp>(
                        loc,
                        isSizeMax.getResult(),
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrRead.getResult(),
                                cstZero.getResult());
                        },
                        [&] {
                            builder.create<sv::PAssignOp>(
                                loc,
                                ptrRead.getResult(),
                                nextRead.getResult());
                        });
                });
                builder.create<sv::IfOp>(loc, notSameEnqDeq.getResult(), [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        maybeFull.getResult(),
                        doEnq.getResult());
                });
                builder.create<sv::IfOp>(loc, updateClose.getResult(), [&] {
                    builder.create<sv::PAssignOp>(
                        loc,
                        wantClose.getResult(),
                        cstTrue.getResult());
                    builder.create<sv::PAssignOp>(
                        loc,
                        ptrLast.getResult(),
                        ptrWriteValue.getResult());
                });
                builder.create<sv::IfOp>(
                    loc,
                    lastReadWantCloseDeq.getResult(),
                    [&] {
                        builder.create<sv::PAssignOp>(
                            loc,
                            isDone.getResult(),
                            cstTrue.getResult());
                    });
            });
    });

    // Outputs
    builder.create<hw::OutputOp>(
        loc,
        ValueRange{
            inReadyOutput.getResult(),
            notEmpty.getResult(),
            ramReadValue.getResult(),
            doneOutput.getResult()});
}

void ConvertDfgToHWPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });

    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    auto insertionPoint = &module.getBodyRegion().front();
    module.walk([&](ProcessOp processOp) -> WalkResult {
        builder.setInsertionPointToStart(insertionPoint);
        insertPullPushChannelModule(builder, 1);
        builder.setInsertionPointToStart(insertionPoint);
        insertPullPushChannelModule(builder, 0);
        return WalkResult::interrupt();
    });
    module.walk([&](RegionOp regionOp) {
        regionOp.walk([&](ChannelOp channelOp) -> WalkResult {
            builder.setInsertionPointToStart(insertionPoint);
            insertQueueModule(builder);
            return WalkResult::interrupt();
        });
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToHWConversionPatterns(converter, patterns);

    target.addLegalDialect<
        comb::CombDialect,
        hw::HWDialect,
        seq::SeqDialect,
        sv::SVDialect>();
    target.addIllegalDialect<dfg::DfgDialect, arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToHWPass()
{
    return std::make_unique<ConvertDfgToHWPass>();
}
