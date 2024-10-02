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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOHW
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

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
    bool hasClose = false)
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
        auto hwCastOp = builder.create<HWCastOp>(
            loc,
            TypeRange{i1Ty, OutputType::get(builder.getContext(), inputTy)},
            hwModuleBlkArgs);
        inReadySignals.push_back(hwCastOp.getResult(0));
        replacePorts.push_back(hwCastOp.getResult(1));
    }
    for (size_t i = 0; i < funcTyNumOutputs; i++) {
        auto outputTy = dyn_cast<IntegerType>(funcTy.getResult(i));
        auto hwCastOp = builder.create<HWCastOp>(
            loc,
            TypeRange{
                i1Ty,
                outputTy,
                InputType::get(builder.getContext(), outputTy)},
            hasClose ? hwModuleRegion.getArgument(i + 3 * funcTyNumInputs + 2)
                     : hwModuleRegion.getArgument(i + 2 * funcTyNumInputs + 2));
        outValidDataSignals.push_back(hwCastOp.getResult(0));
        outValidDataSignals.push_back(hwCastOp.getResult(1));
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
        for (auto &opi : op.getBody().getOps()) {
            if (auto loopOp = dyn_cast<LoopOp>(opi)) {
                SmallVector<Value> loopedPorts;
                auto loopOpInChans = loopOp.getInChans();
                if (loopOpInChans.size() == 0)
                    for (auto map : argumentMap)
                        loopedPorts.push_back(map.second);
                else {
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
                kernelOps.push_back(&opi);
            }
        }
        // If there is no LoopOp in the ProcessOp, insert an empty HWLoopOp
        if (!hasLoop) {
            auto hwLoopOp =
                rewriter.create<HWLoopOp>(hwModuleLoc, i1Ty, ValueRange{});
            loopOpResult = hwLoopOp.getResult();
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
        }
        auto waitOp =
            rewriter.create<HWWaitOp>(hwKernelModuleLoc, i1Ty, ValueRange{});
        SmallVector<Value> kernelOutputs;
        kernelOutputs.append(
            kernelInReadySignals.begin(),
            kernelInReadySignals.end());
        kernelOutputs.append(
            kernelOutValidDataSignals.begin(),
            kernelOutValidDataSignals.end());
        kernelOutputs.push_back(waitOp.getResult());
        rewriter.create<hw::OutputOp>(hwKernelModuleLoc, kernelOutputs);

        // Insert an instance to the kernel
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
        auto doneSignal = rewriter.create<comb::AndOp>(
            hwModuleLoc,
            instanceKernel.getResults().back(),
            loopOpResult);
        // outputs.append(inReadySignals.begin(), inReadySignals.end());
        // outputs.append(outValidDataSignals.begin(),
        // outValidDataSignals.end());
        outputs.push_back(doneSignal.getResult());
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

        // OutputOp
        auto waitOp =
            rewriter.create<HWWaitOp>(hwModuleLoc, i1Ty, ValueRange{});
        SmallVector<Value> outputs;
        outputs.append(inReadySignals.begin(), inReadySignals.end());
        outputs.append(outValidDataSignals.begin(), outValidDataSignals.end());
        outputs.push_back(waitOp.getResult());
        rewriter.create<hw::OutputOp>(hwModuleLoc, outputs);

        rewriter.eraseOp(op);
        return success();
    }
};

template<typename OpT>
struct PruneUnusedHWOps : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    PruneUnusedHWOps(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpT>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpT op,
        OpT::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
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
    patterns.add<PruneUnusedHWOps<HWCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<PruneUnusedHWOps<HWLoopOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<PruneUnusedHWOps<HWWaitOp>>(
        typeConverter,
        patterns.getContext());
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
        {builder.getStringAttr("close"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in_valid"), i1Ty, inDir}
    });
    ports.push_back(hw::PortInfo{
        {builder.getStringAttr("in"), dataParamTy, inDir}
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
    auto close = hwModule.getBody().getArgument(2);
    auto inValid = hwModule.getBody().getArgument(3);
    auto inData = hwModule.getBody().getArgument(4);
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
    // Implementing RegionOp lowering, all other ops are legal
    target.addLegalOp<
        LoopOp,
        PullOp,
        PushOp,
        ChannelOp,
        ConnectInputOp,
        ConnectOutputOp,
        InstantiateOp,
        arith::AddIOp>();
    target.addDynamicallyLegalOp<HWCastOp, HWLoopOp, HWWaitOp>(
        [&](Operation* op) {
            for (auto result : op->getResults()) {
                auto resultUses = result.getUses();
                if (std::distance(resultUses.begin(), resultUses.end()) == 0)
                    return false;
                else
                    return true;
            }
        });

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
