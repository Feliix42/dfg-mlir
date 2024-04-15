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

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTDTOCIRCT
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

// arith -> comb
template<typename From, typename To>
struct OneToOneConversion : public OpConversionPattern<From> {
    using OpConversionPattern<From>::OpConversionPattern;

    OneToOneConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<From>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        From op,
        typename From::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<To>(
            op,
            adaptor.getOperands(),
            op->getAttrs());

        return success();
    }
};

struct ExtSConversion : public OpConversionPattern<arith::ExtSIOp> {
    using OpConversionPattern<arith::ExtSIOp>::OpConversionPattern;

    ExtSConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::ExtSIOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        arith::ExtSIOp op,
        arith::ExtSIOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto width = op.getType().getIntOrFloatBitWidth();
        rewriter.replaceOp(
            op,
            comb::createOrFoldSExt(
                op.getLoc(),
                op.getOperand(),
                rewriter.getIntegerType(width),
                rewriter));

        return success();
    }
};

struct ExtZConversion : public OpConversionPattern<arith::ExtUIOp> {
    using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;

    ExtZConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::ExtUIOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        arith::ExtUIOp op,
        arith::ExtUIOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto outWidth = op.getOut().getType().getIntOrFloatBitWidth();
        auto inWidth = adaptor.getIn().getType().getIntOrFloatBitWidth();

        rewriter.replaceOp(
            op,
            rewriter.create<comb::ConcatOp>(
                loc,
                rewriter.create<hw::ConstantOp>(
                    loc,
                    APInt(outWidth - inWidth, 0)),
                adaptor.getIn()));

        return success();
    }
};

struct TruncConversion : public OpConversionPattern<arith::TruncIOp> {
    using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

    TruncConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::TruncIOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        arith::TruncIOp op,
        arith::TruncIOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto width = op.getType().getIntOrFloatBitWidth();
        rewriter
            .replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getIn(), 0, width);

        return success();
    }
};

struct CompConversion : public OpConversionPattern<arith::CmpIOp> {
    using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

    CompConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::CmpIOp>(typeConverter, context){};

    static comb::ICmpPredicate
    arithToCombPredicate(arith::CmpIPredicate predicate)
    {
        switch (predicate) {
        case arith::CmpIPredicate::eq: return comb::ICmpPredicate::eq;
        case arith::CmpIPredicate::ne: return comb::ICmpPredicate::ne;
        case arith::CmpIPredicate::slt: return comb::ICmpPredicate::slt;
        case arith::CmpIPredicate::ult: return comb::ICmpPredicate::ult;
        case arith::CmpIPredicate::sle: return comb::ICmpPredicate::sle;
        case arith::CmpIPredicate::ule: return comb::ICmpPredicate::ule;
        case arith::CmpIPredicate::sgt: return comb::ICmpPredicate::sgt;
        case arith::CmpIPredicate::ugt: return comb::ICmpPredicate::ugt;
        case arith::CmpIPredicate::sge: return comb::ICmpPredicate::sge;
        case arith::CmpIPredicate::uge: return comb::ICmpPredicate::uge;
        }
        llvm_unreachable("Unknown predicate");
    }

    LogicalResult matchAndRewrite(
        arith::CmpIOp op,
        arith::CmpIOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<comb::ICmpOp>(
            op,
            arithToCombPredicate(op.getPredicate()),
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }
};

// func.func -> hw.module
struct FuncConversion : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    FuncConversion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<func::FuncOp>(typeConverter, context){};

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
            ArrayRef<NamedAttribute>{},
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
    func::FuncOp &genFuncOp,
    PatternRewriter &rewriter)
{
    int idxOperand = 0;
    for (auto operand : newCalcOp->getOperands()) {
        if (auto idxArg =
                getNewIndexOrArg<int, Value>(operand, pulledValueIdx)) {
            newCalcOp->setOperand(
                idxOperand++,
                genFuncOp.getBody().getArgument(idxArg.value()));
        } else {
            auto definingOp = operand.getDefiningOp();
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
        }
    }
    if (newCalcOp->getRegions().size() != 0) {
        for (auto &region : newCalcOp->getRegions()) {
            for (auto &opRegion : region.getOps()) {
                processNestedRegions(
                    &opRegion,
                    newCalcOps,
                    pulledValueIdx,
                    calcOpIdx,
                    genFuncOp,
                    rewriter);
            }
        }
    }
}
struct WrapProcessOps : public OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    WrapProcessOps(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto context = rewriter.getContext();
        auto operatorName = op.getSymName();
        auto funcTy = op.getFunctionType();
        auto ops = op.getOps();
        auto moduleOp = op->getParentOfType<ModuleOp>();

        LoopOp loopOp = nullptr;
        bool hasLoopOp = false;
        SmallVector<int> loopInChanIdx, loopOutChanIdx;
        if (auto oldLoop = dyn_cast<LoopOp>(*ops.begin())) {
            ops = oldLoop.getOps();
            loopOp = oldLoop;
            hasLoopOp = true;
            for (auto inChan : oldLoop.getInChans()) {
                auto idxChan = inChan.cast<BlockArgument>().getArgNumber();
                loopInChanIdx.push_back(idxChan);
            }
            for (auto outChan : oldLoop.getOutChans()) {
                auto idxChan = outChan.cast<BlockArgument>().getArgNumber();
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
        for (auto &opi : ops)
            if (auto pushOp = dyn_cast<PushOp>(opi))
                pushedValueRepeat.push_back(pushOp.getInp());
        for (auto &opi : ops) {
            if (auto pullOp = dyn_cast<PullOp>(opi)) {
                auto pullValue = pullOp.getOutp();
                pulledValue.push_back(pullValue);
                auto pullChan = pullOp.getChan();
                pulledTypes.push_back(pullChan.getType().getElementType());
                auto idxChan = pullChan.cast<BlockArgument>().getArgNumber();
                pulledChanIdx.push_back(idxChan);
                pulledValueIdx.push_back(std::make_pair(idxPull++, pullValue));
                numPull++;
            } else if (!isa<PushOp>(opi)) {
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
                auto idxChan = pushChan.cast<BlockArgument>().getArgNumber();
                pushedChanIdx.push_back(idxChan);
                if (!isPushPulledValue) {
                    auto idx =
                        getNewIndexOrArg<int, Value>(pushValue, calcResultIdx);
                    pushedValueIdx.push_back(idx.value());
                } else {
                    pushedValueIdx.push_back(-1);
                }
                numPush++;
            }
        }

        auto newOperator =
            rewriter.create<ProcessOp>(op.getLoc(), op.getSymName(), funcTy);
        Block* entryBlock = rewriter.createBlock(&newOperator.getBody());
        for (auto inTy : funcTy.getInputs())
            entryBlock->addArgument(inTy, newOperator.getLoc());
        for (auto outTy : funcTy.getResults())
            entryBlock->addArgument(outTy, newOperator.getLoc());

        SmallVector<Value> newPulledValue;
        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPointToStart(entryBlock);
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
            auto newLoop =
                rewriter.create<LoopOp>(loc, loopInChans, loopOutChans);
            Block* loopEntryBlock = rewriter.createBlock(&newLoop.getBody());
            rewriter.setInsertionPointToStart(loopEntryBlock);
        }
        for (int i = 0; i < numPull; i++) {
            auto newPull = rewriter.create<PullOp>(
                loc,
                newOperator.getBody().getArgument(pulledChanIdx[i]));
            newPulledValue.push_back(newPull.getResult());
        }

        auto nameExtModule = "hls_" + operatorName.str() + "_calc";
        auto instanceOp = rewriter.create<HWInstanceOp>(
            loc,
            pushedTypes,
            SymbolRefAttr::get(context, nameExtModule),
            newPulledValue);

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

        rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
        auto genFuncOp = rewriter.create<func::FuncOp>(
            loc,
            nameExtModule,
            rewriter.getFunctionType(pulledTypes, pushedTypes));
        Block* funcEntryBlock = rewriter.createBlock(&genFuncOp.getBody());
        for (int i = 0; i < numPull; i++)
            funcEntryBlock->addArgument(pulledTypes[i], genFuncOp.getLoc());
        rewriter.setInsertionPointToStart(funcEntryBlock);

        SmallVector<Operation*> newCalcOps;
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
                genFuncOp,
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

        rewriter.create<func::ReturnOp>(loc, returnValues);

        writeFuncToFile(genFuncOp, genFuncOp.getSymName());
        rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
        SmallVector<hw::PortInfo> ports;
        for (size_t i = 0; i < pulledTypes.size(); i++) {
            auto type = pulledTypes[i];
            std::string name;
            hw::PortInfo in_data;
            name = "in" + std::to_string(i);
            in_data.name = rewriter.getStringAttr(name);
            in_data.dir = hw::ModulePort::Direction::Input;
            in_data.type = type;
            ports.push_back(in_data);
            hw::PortInfo in_valid;
            name = "in" + std::to_string(i) + "_valid";
            in_valid.name = rewriter.getStringAttr(name);
            in_valid.dir = hw::ModulePort::Direction::Input;
            in_valid.type = rewriter.getI1Type();
            ports.push_back(in_valid);
            hw::PortInfo in_ready;
            name = "in" + std::to_string(i) + "_ready";
            in_ready.name = rewriter.getStringAttr(name);
            in_ready.dir = hw::ModulePort::Direction::Output;
            in_ready.type = rewriter.getI1Type();
            ports.push_back(in_ready);
        }
        hw::PortInfo ctrl_valid;
        ctrl_valid.name = rewriter.getStringAttr(
            "in" + std::to_string(pulledTypes.size()) + "_valid");
        ctrl_valid.dir = hw::ModulePort::Direction::Input;
        ctrl_valid.type = rewriter.getI1Type();
        ports.push_back(ctrl_valid);
        hw::PortInfo clock;
        clock.name = rewriter.getStringAttr("clock");
        clock.dir = hw::ModulePort::Direction::Input;
        clock.type = rewriter.getI1Type();
        ports.push_back(clock);
        hw::PortInfo reset;
        reset.name = rewriter.getStringAttr("reset");
        reset.dir = hw::ModulePort::Direction::Input;
        reset.type = rewriter.getI1Type();
        ports.push_back(reset);
        for (size_t i = 0; i < pushedTypes.size(); i++) {
            auto type = pushedTypes[i];
            std::string name;
            hw::PortInfo out_ready;
            name = "out" + std::to_string(i) + "_ready";
            out_ready.name = rewriter.getStringAttr(name);
            out_ready.dir = hw::ModulePort::Direction::Input;
            out_ready.type = rewriter.getI1Type();
            ports.push_back(out_ready);
            hw::PortInfo out_data;
            name = "out" + std::to_string(i);
            out_data.name = rewriter.getStringAttr(name);
            out_data.dir = hw::ModulePort::Direction::Output;
            out_data.type = type;
            ports.push_back(out_data);
            hw::PortInfo out_valid;
            name = "out" + std::to_string(i) + "_valid";
            out_valid.name = rewriter.getStringAttr(name);
            out_valid.dir = hw::ModulePort::Direction::Output;
            out_valid.type = rewriter.getI1Type();
            ports.push_back(out_valid);
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

} // namespace

void mlir::populateStdToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // arith -> comb
    // patterns.add<OneToOneConversion<arith::ConstantOp, hw::ConstantOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::AddIOp, comb::AddOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::SubIOp, comb::SubOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::MulIOp, comb::MulOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::DivSIOp, comb::DivSOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::DivUIOp, comb::DivUOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::RemSIOp, comb::ModSOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::RemUIOp, comb::ModUOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::AndIOp, comb::AndOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::OrIOp, comb::OrOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::XOrIOp, comb::XorOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::ShLIOp, comb::ShlOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::ShRSIOp, comb::ShrSOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::ShRUIOp, comb::ShrUOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<OneToOneConversion<arith::SelectOp, comb::MuxOp>>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<ExtSConversion>(typeConverter, patterns.getContext());
    // patterns.add<ExtZConversion>(typeConverter, patterns.getContext());
    // patterns.add<TruncConversion>(typeConverter, patterns.getContext());
    // patterns.add<CompConversion>(typeConverter, patterns.getContext());

    // func.func -> hw.module
    patterns.add<FuncConversion>(typeConverter, patterns.getContext());

    // operator calc ops -> handshake.func
    patterns.add<WrapProcessOps>(typeConverter, patterns.getContext());
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
    // target.addIllegalDialect<func::FuncDialect>();

    func::FuncOp lastFunc;
    auto module = dyn_cast<ModuleOp>(getOperation());
    module.walk([&](func::FuncOp funcOp) { lastFunc = funcOp; });

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

    target.addDynamicallyLegalOp<ProcessOp>([&](ProcessOp op) {
        auto ops = op.getBody().getOps();
        if (auto loopOp = dyn_cast<LoopOp>(*ops.begin()))
            ops = loopOp.getBody().getOps();
        for (auto &opi : ops) {
            if (!isa<PullOp>(opi) && !isa<HWInstanceOp>(opi)
                && !isa<PushOp>(opi)) {
                return false;
            }
        }
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
