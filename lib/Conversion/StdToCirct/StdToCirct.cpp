/// Implementation of StdToCirct pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "dfg-mlir/Conversion/StdToCirct/StdToCirct.h"

#include "../PassDetails.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

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
        From::Adaptor adaptor,
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
            newArguments.push_back(
                std::make_pair(hwModule.getArgument(i), op.getArgument(i)));

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
                    newArg.value());
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

} // namespace

void mlir::populateStdToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    // arith -> comb
    patterns.add<OneToOneConversion<arith::ConstantOp, hw::ConstantOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::AddIOp, comb::AddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::SubIOp, comb::SubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::MulIOp, comb::MulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::DivSIOp, comb::DivSOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::DivUIOp, comb::DivUOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::RemSIOp, comb::ModSOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::RemUIOp, comb::ModUOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::AndIOp, comb::AndOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::OrIOp, comb::OrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::XOrIOp, comb::XorOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::ShLIOp, comb::ShlOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::ShRSIOp, comb::ShrSOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::ShRUIOp, comb::ShrUOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<OneToOneConversion<arith::SelectOp, comb::MuxOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ExtSConversion>(typeConverter, patterns.getContext());
    patterns.add<ExtZConversion>(typeConverter, patterns.getContext());
    patterns.add<TruncConversion>(typeConverter, patterns.getContext());
    patterns.add<CompConversion>(typeConverter, patterns.getContext());

    // func.func -> hw.module
    patterns.add<FuncConversion>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertStdToCirctPass
        : public ConvertStdToCirctBase<ConvertStdToCirctPass> {
    void runOnOperation() final;
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
        comb::CombDialect,
        hw::HWDialect,
        sv::SVDialect,
        dfg::DfgDialect>();
    target.addIllegalDialect<arith::ArithDialect, func::FuncDialect>();

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
