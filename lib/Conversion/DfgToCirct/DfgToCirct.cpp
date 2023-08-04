/// Implementation of DfgToCirct pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "../PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "dfg-mlir/Conversion/DfgToCirct/DfgToCirct.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
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

// Helper to determine which arguments should to be used in
// the new FModule when an op inside is trying to use the
// old arguments.
SmallVector<std::pair<int, Value>> oldArgNums;
SmallVector<Value> newArgs;
SmallVector<std::pair<SmallVector<Value>, Value>> newArguments;
SmallVector<std::pair<Value, Value>> newOperands;
template<typename T1, typename T2>
std::optional<T1>
getNewArgOrOperand(T2 find, SmallVector<std::pair<T1, T2>> args)
{
    for (const auto &kv : args)
        if (kv.second == find) return kv.first;
    return std::nullopt;
}

// Helper to determine which module and operators to use
// when instantiating an operator.
std::map<std::string, firrtl::FModuleOp> operatorList;
SmallVector<std::pair<SmallVector<Value>, Value>> newChannels;

struct ConvertOperator : OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    ConvertOperator(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp op,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Needed arguments for creation of FModuleOp
        auto context = rewriter.getContext();
        auto name = op.getSymNameAttr();
        auto convention =
            firrtl::ConventionAttr::get(context, Convention::Internal);
        SmallVector<firrtl::PortInfo> ports;
        auto args = op.getBody().getArguments();
        auto types = op.getBody().getArgumentTypes();
        size_t size = types.size();

        // Create new ports for the FModule
        int in_num = 1;
        int out_num = 1;
        int port_num = 0;
        oldArgNums.clear();
        newArgs.clear();
        for (size_t i = 0; i < size; i++) {
            const auto arg = args[i];
            const auto type = types[i];
            if (const auto inTy = dyn_cast<OutputType>(type)) {
                auto elemTy = inTy.getElementType().dyn_cast<IntegerType>();
                assert(elemTy && "only integers are supported on hardware");
                assert(
                    !elemTy.isSignless()
                    && "signless integers are not supported in firrtl");
                auto width = elemTy.getWidth();
                ports.push_back(firrtl::PortInfo(
                    rewriter.getStringAttr(
                        "io_in" + std::to_string(in_num) + "_ready"),
                    firrtl::UIntType::get(context, 1),
                    firrtl::Direction::Out));
                ports.push_back(firrtl::PortInfo(
                    rewriter.getStringAttr(
                        "io_in" + std::to_string(in_num) + "_valid"),
                    firrtl::UIntType::get(context, 1),
                    firrtl::Direction::In));
                if (elemTy.isSigned()) {
                    ports.push_back(firrtl::PortInfo(
                        rewriter.getStringAttr(
                            "io_in" + std::to_string(in_num) + "_bits"),
                        firrtl::SIntType::get(context, width),
                        firrtl::Direction::In));
                } else if (elemTy.isUnsigned()) {
                    ports.push_back(firrtl::PortInfo(
                        rewriter.getStringAttr(
                            "io_in" + std::to_string(in_num) + "_bits"),
                        firrtl::UIntType::get(context, width),
                        firrtl::Direction::In));
                }
                oldArgNums.push_back(std::make_pair(port_num, arg));
                in_num++;
                port_num++;
            } else if (const auto outTy = dyn_cast<InputType>(type)) {
                auto elemTy = outTy.getElementType().dyn_cast<IntegerType>();
                assert(elemTy && "only integers are supported on hardware");
                assert(
                    !elemTy.isSignless()
                    && "signless integers are not supported in firrtl");
                auto width = elemTy.getWidth();
                ports.push_back(firrtl::PortInfo(
                    rewriter.getStringAttr(
                        "io_out" + std::to_string(out_num) + "_ready"),
                    firrtl::UIntType::get(context, 1),
                    firrtl::Direction::In));
                ports.push_back(firrtl::PortInfo(
                    rewriter.getStringAttr(
                        "io_out" + std::to_string(out_num) + "_valid"),
                    firrtl::UIntType::get(context, 1),
                    firrtl::Direction::Out));
                if (elemTy.isSigned()) {
                    ports.push_back(firrtl::PortInfo(
                        rewriter.getStringAttr(
                            "io_out" + std::to_string(out_num) + "_bits"),
                        firrtl::SIntType::get(context, width),
                        firrtl::Direction::Out));
                } else if (elemTy.isUnsigned()) {
                    ports.push_back(firrtl::PortInfo(
                        rewriter.getStringAttr(
                            "io_out" + std::to_string(out_num) + "_bits"),
                        firrtl::UIntType::get(context, width),
                        firrtl::Direction::Out));
                }
                oldArgNums.push_back(std::make_pair(port_num, arg));
                out_num++;
                port_num++;
            }
        }
        assert(size_t(port_num) == types.size());

        // Create new FIRRTL Module
        auto module = rewriter.create<firrtl::FModuleOp>(
            op.getLoc(),
            name,
            convention,
            ports);
        operatorList.emplace(op.getSymName(), module);

        // Store the new args for later use for pull/push
        for (const auto &arg : module.getArguments()) newArgs.push_back(arg);

        // Move the Ops in Operator into new FIRRTL Module
        rewriter.setInsertionPointToStart(&module.getBody().front());
        for (auto &ops : llvm::make_early_inc_range(op.getBody().getOps()))
            ops.moveBefore(module.getBodyBlock(), module.getBodyBlock()->end());

        rewriter.eraseOp(op);

        return success();
    }
};

// struct ConvertLoop : OpConversionPattern<LoopOp> {
//     using OpConversionPattern<LoopOp>::OpConversionPattern;

//     ConvertLoop(TypeConverter &typeConverter, MLIRContext* context)
//             : OpConversionPattern<LoopOp>(typeConverter, context){};

//     LogicalResult matchAndRewrite(
//         LoopOp op,
//         LoopOpAdaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         rewriter.eraseOp(op);

//         return success();
//     }
// };

// TODO: Logic is incorrect, check what to do!!!

struct ConvertPull : OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    ConvertPull(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto context = this->getContext();
        auto loc = op.getLoc();

        // Get the position of the original argument
        auto port = getNewArgOrOperand<int, Value>(op.getChan(), oldArgNums);
        assert(port && "cannot use undefined argument");

        // Output ready signal
        auto deq_ready = newArgs[3 * port.value()];
        // Input valid signal and data
        auto deq_valid = newArgs[3 * port.value() + 1];
        auto deq_bits = newArgs[3 * port.value() + 2];

        auto const0 = rewriter.create<firrtl::ConstantOp>(
            loc,
            firrtl::UIntType::get(context, 1),
            llvm::APInt(/*numBits=*/1, /*value=*/0));
        rewriter.create<firrtl::StrictConnectOp>(
            loc,
            /*dest=*/deq_ready,
            /*src=*/deq_valid);
        auto wire = rewriter.create<firrtl::WireOp>(loc, deq_bits.getType());
        rewriter.create<firrtl::StrictConnectOp>(
            loc,
            wire.getResult(),
            deq_bits);
        rewriter.create<firrtl::StrictConnectOp>(
            loc,
            deq_ready,
            const0.getResult());
        newOperands.push_back(std::make_pair(wire.getResult(), op.getResult()));

        rewriter.eraseOp(op);

        return success();
    }
};

struct ConvertPush : OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    ConvertPush(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp op,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto context = this->getContext();
        auto loc = op.getLoc();

        // Get the position of the original argument
        auto port = getNewArgOrOperand<int, Value>(op.getChan(), oldArgNums);
        assert(port && "cannot use undefined argument");

        // Output ready signal
        auto enq_ready = newArgs[3 * port.value()];
        // Input valid signal and data
        auto enq_valid = newArgs[3 * port.value() + 1];
        auto enq_bits = newArgs[3 * port.value() + 2];

        auto const0 = rewriter.create<firrtl::ConstantOp>(
            loc,
            firrtl::UIntType::get(context, 1),
            llvm::APInt(/*numBits=*/1, /*value=*/0));
        rewriter.create<firrtl::StrictConnectOp>(
            loc,
            /*dest=*/enq_valid,
            /*src=*/enq_ready);

        auto toPush =
            getNewArgOrOperand<Value, Value>(op.getInp(), newOperands);
        assert(toPush && "cannot push nothing to channel");

        rewriter.create<firrtl::StrictConnectOp>(loc, enq_bits, toPush.value());
        rewriter.create<firrtl::StrictConnectOp>(
            loc,
            enq_valid,
            const0.getResult());

        rewriter.eraseOp(op);

        return success();
    }
};

// Helper to instantiate Queue according to the channel
std::map<std::string, firrtl::FModuleOp> queueList;
firrtl::FModuleOp insertQueue(
    OpBuilder &builder,
    MLIRContext* context,
    Location loc,
    std::string name,
    bool isSigned,
    unsigned portBit,
    unsigned size)
{
    // Arguments needed to create a FModuleOp for Queue
    auto nameStrAttr = builder.getStringAttr(name);
    auto convention =
        firrtl::ConventionAttr::get(context, Convention::Internal);
    SmallVector<firrtl::PortInfo> ports;
    ports.push_back(firrtl::PortInfo(
        builder.getStringAttr("clock"),
        firrtl::ClockType::get(context),
        firrtl::Direction::In));
    ports.push_back(firrtl::PortInfo(
        builder.getStringAttr("reset"),
        firrtl::UIntType::get(context, 1),
        firrtl::Direction::In));
    ports.push_back(firrtl::PortInfo(
        builder.getStringAttr("io_enq_ready"),
        firrtl::UIntType::get(context, 1),
        firrtl::Direction::Out));
    ports.push_back(firrtl::PortInfo(
        builder.getStringAttr("io_enq_valid"),
        firrtl::UIntType::get(context, 1),
        firrtl::Direction::In));
    if (isSigned) {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_enq_bits"),
            firrtl::SIntType::get(context, portBit),
            firrtl::Direction::In));
    } else {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_enq_bits"),
            firrtl::UIntType::get(context, portBit),
            firrtl::Direction::In));
    }
    ports.push_back(firrtl::PortInfo(
        builder.getStringAttr("io_deq_ready"),
        firrtl::UIntType::get(context, 1),
        firrtl::Direction::In));
    ports.push_back(firrtl::PortInfo(
        builder.getStringAttr("io_deq_valid"),
        firrtl::UIntType::get(context, 1),
        firrtl::Direction::Out));
    if (isSigned) {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_deq_bits"),
            firrtl::SIntType::get(context, portBit),
            firrtl::Direction::Out));
    } else {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_deq_bits"),
            firrtl::UIntType::get(context, portBit),
            firrtl::Direction::Out));
    }

    auto moduleOp =
        builder.create<firrtl::FModuleOp>(loc, nameStrAttr, convention, ports);

    // Create the Queue behavior one by one
    OpBuilder newBuilder(context);
    newBuilder.setInsertionPointToEnd(&moduleOp.getBodyRegion().front());
    auto newLoc = newBuilder.getUnknownLoc();

    // Constants
    uint64_t bitDepth = std::ceil(std::log2(size));
    auto isTwoPower = size > 0 && (size & (size - 1)) == 0;
    auto const0 = newBuilder.create<firrtl::ConstantOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        llvm::APInt(/*numBits=*/1, /*value=*/1));
    auto const1 = newBuilder.create<firrtl::ConstantOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        llvm::APInt(1, 0));
    firrtl::ConstantOp const2, const3;
    if (bitDepth > 1) {
        const2 = newBuilder.create<firrtl::ConstantOp>(
            newLoc,
            firrtl::UIntType::get(context, bitDepth),
            llvm::APInt(bitDepth, 0));
        if (!isTwoPower) {
            const3 = newBuilder.create<firrtl::ConstantOp>(
                newLoc,
                firrtl::UIntType::get(context, bitDepth),
                llvm::APInt(bitDepth, size - 1));
        }
    }

    // Memory declaration
    SmallVector<firrtl::BundleType::BundleElement> bundleMportField;
    bundleMportField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("addr"),
        false,
        firrtl::UIntType::get(context, bitDepth)));
    bundleMportField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("en"),
        false,
        firrtl::UIntType::get(context, 1)));
    bundleMportField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("clk"),
        false,
        firrtl::ClockType::get(context)));
    if (isSigned) {
        bundleMportField.push_back(firrtl::BundleType::BundleElement(
            newBuilder.getStringAttr("data"),
            false,
            firrtl::SIntType::get(context, portBit)));
    } else {
        bundleMportField.push_back(firrtl::BundleType::BundleElement(
            newBuilder.getStringAttr("data"),
            false,
            firrtl::UIntType::get(context, portBit)));
    }
    bundleMportField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("mask"),
        false,
        firrtl::UIntType::get(context, 1)));
    auto bundleMport = firrtl::BundleType::get(context, bundleMportField);
    SmallVector<firrtl::BundleType::BundleElement> bundleDeqField;
    bundleDeqField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("addr"),
        false,
        firrtl::UIntType::get(context, bitDepth)));
    bundleDeqField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("en"),
        false,
        firrtl::UIntType::get(context, 1)));
    bundleDeqField.push_back(firrtl::BundleType::BundleElement(
        newBuilder.getStringAttr("clk"),
        false,
        firrtl::ClockType::get(context)));
    if (isSigned) {
        bundleDeqField.push_back(firrtl::BundleType::BundleElement(
            newBuilder.getStringAttr("data"),
            true,
            firrtl::SIntType::get(context, portBit)));
    } else {
        bundleDeqField.push_back(firrtl::BundleType::BundleElement(
            newBuilder.getStringAttr("data"),
            true,
            firrtl::UIntType::get(context, portBit)));
    }
    auto bundleDeq = firrtl::BundleType::get(context, bundleDeqField);
    SmallVector<Type> memResultTypes;
    memResultTypes.push_back(bundleMport);
    memResultTypes.push_back(bundleDeq);
    SmallVector<Attribute> portNames;
    portNames.push_back(newBuilder.getStringAttr("MPORT"));
    portNames.push_back(newBuilder.getStringAttr("io_deq_bits_MPORT"));
    auto memOp = newBuilder.create<firrtl::MemOp>(
        newLoc,
        memResultTypes,
        0,
        1,
        size,
        RUWAttr::Undefined,
        portNames,
        "ram");

    // Extract value from memory bundle field
    // auto test = memOp.getPortNamed("MPORT");
    auto addrMport = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("MPORT"),
        "addr");
    auto enMport = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("MPORT"),
        "en");
    auto clkMport = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("MPORT"),
        "clk");
    auto dataMport = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("MPORT"),
        "data");
    auto maskMport = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("MPORT"),
        "mask");
    auto addrDeq = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("io_deq_bits_MPORT"),
        "addr");
    auto enDeq = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("io_deq_bits_MPORT"),
        "en");
    auto clkDeq = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("io_deq_bits_MPORT"),
        "clk");
    auto dataDeq = newBuilder.create<firrtl::SubfieldOp>(
        newLoc,
        memOp.getPortNamed("io_deq_bits_MPORT"),
        "data");

    // Get RegReset value
    auto value = newBuilder.create<firrtl::RegResetOp>(
        newLoc,
        firrtl::UIntType::get(context, bitDepth),
        moduleOp.getArgument(0),
        moduleOp.getArgument(1),
        const2.getResult(),
        "value");
    value->setAttr(
        "firrtl.random_init_start",
        newBuilder.getIntegerAttr(newBuilder.getIntegerType(64, false), 0));
    auto value_1 = newBuilder.create<firrtl::RegResetOp>(
        newLoc,
        firrtl::UIntType::get(context, bitDepth),
        moduleOp.getArgument(0),
        moduleOp.getArgument(1),
        const2.getResult(),
        "value_1");
    value_1->setAttr(
        "firrtl.random_init_start",
        newBuilder.getIntegerAttr(
            newBuilder.getIntegerType(64, false),
            bitDepth));
    auto maybe_full = newBuilder.create<firrtl::RegResetOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        moduleOp.getArgument(0),
        moduleOp.getArgument(1),
        const1.getResult(),
        "maybe_full");
    maybe_full->setAttr(
        "firrtl.random_init_start",
        newBuilder.getIntegerAttr(
            newBuilder.getIntegerType(64, false),
            bitDepth * 2));

    // Semantics used to determine empty or full
    auto ptr_match = newBuilder.create<firrtl::EQPrimOp>(
        newLoc,
        value.getResult(),
        value_1.getResult());
    auto _empty_T =
        newBuilder.create<firrtl::NotPrimOp>(newLoc, maybe_full.getResult());
    auto empty = newBuilder.create<firrtl::AndPrimOp>(
        newLoc,
        ptr_match.getResult(),
        _empty_T.getResult());
    auto full = newBuilder.create<firrtl::AndPrimOp>(
        newLoc,
        ptr_match.getResult(),
        maybe_full.getResult());
    auto _do_enq_T = newBuilder.create<firrtl::AndPrimOp>(
        newLoc,
        moduleOp.getArgument(2),
        moduleOp.getArgument(3));
    auto do_enq = newBuilder.create<firrtl::WireOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        "do_enq");
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        do_enq.getResult(),
        _do_enq_T.getResult());
    auto _do_deq_T = newBuilder.create<firrtl::AndPrimOp>(
        newLoc,
        moduleOp.getArgument(5),
        moduleOp.getArgument(6));
    auto do_deq = newBuilder.create<firrtl::WireOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        "do_deq");
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        do_deq.getResult(),
        _do_deq_T.getResult());

    // Connect enq to memory
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        addrMport.getResult(),
        value.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        enMport.getResult(),
        do_enq.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        clkMport.getResult(),
        moduleOp.getArgument(0));
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        maskMport.getResult(),
        const0.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        dataMport.getResult(),
        moduleOp.getArgument(4));

    // Addr +1 when enq -1 when deq
    // if full then back to 0
    firrtl::EQPrimOp wrap;
    if (!isTwoPower)
        wrap = newBuilder.create<firrtl::EQPrimOp>(
            newLoc,
            value.getResult(),
            const3.getResult());
    auto _value_T = newBuilder.create<firrtl::AddPrimOp>(
        newLoc,
        value.getResult(),
        const0.getResult());
    auto _value_T_1 = newBuilder.create<firrtl::BitsPrimOp>(
        newLoc,
        _value_T.getResult(),
        bitDepth - 1,
        0);
    firrtl::MuxPrimOp mux_do_enq;
    if (isTwoPower)
        mux_do_enq = newBuilder.create<firrtl::MuxPrimOp>(
            newLoc,
            do_enq.getResult(),
            _value_T_1.getResult(),
            value.getResult());
    else {
        auto mux_wrap = newBuilder.create<firrtl::MuxPrimOp>(
            newLoc,
            wrap.getResult(),
            const2.getResult(),
            _value_T_1.getResult());
        mux_do_enq = newBuilder.create<firrtl::MuxPrimOp>(
            newLoc,
            do_enq.getResult(),
            mux_wrap.getResult(),
            value.getResult());
    }
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        value.getResult(),
        mux_do_enq.getResult());
    firrtl::EQPrimOp wrap_1;
    if (!isTwoPower)
        wrap_1 = newBuilder.create<firrtl::EQPrimOp>(
            newLoc,
            value_1.getResult(),
            const3.getResult());
    auto _value_T_2 = newBuilder.create<firrtl::AddPrimOp>(
        newLoc,
        value_1.getResult(),
        const0.getResult());
    auto _value_T_3 = newBuilder.create<firrtl::BitsPrimOp>(
        newLoc,
        _value_T_2.getResult(),
        bitDepth - 1,
        0);
    firrtl::MuxPrimOp mux_do_deq;
    if (isTwoPower)
        mux_do_deq = newBuilder.create<firrtl::MuxPrimOp>(
            newLoc,
            do_deq.getResult(),
            _value_T_3.getResult(),
            value_1.getResult());
    else {
        auto mux_wrap_1 = newBuilder.create<firrtl::MuxPrimOp>(
            newLoc,
            wrap_1.getResult(),
            const2.getResult(),
            _value_T_3.getResult());
        mux_do_deq = newBuilder.create<firrtl::MuxPrimOp>(
            newLoc,
            do_deq.getResult(),
            mux_wrap_1.getResult(),
            value_1.getResult());
    }
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        value_1.getResult(),
        mux_do_deq.getResult());

    // Check if full or empty
    auto check_maybe_full = newBuilder.create<firrtl::NEQPrimOp>(
        newLoc,
        do_enq.getResult(),
        do_deq.getResult());
    auto mux_maybe_full = newBuilder.create<firrtl::MuxPrimOp>(
        newLoc,
        check_maybe_full.getResult(),
        do_enq.getResult(),
        maybe_full.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        maybe_full.getResult(),
        mux_maybe_full.getResult());
    auto _io_deq_valid_T =
        newBuilder.create<firrtl::NotPrimOp>(newLoc, empty.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        moduleOp.getArgument(6),
        _io_deq_valid_T.getResult());
    auto _io_enq_ready_T =
        newBuilder.create<firrtl::NotPrimOp>(newLoc, full.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        moduleOp.getArgument(2),
        _io_enq_ready_T.getResult());

    // Connect deq to memory
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        addrDeq.getResult(),
        value_1.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        enDeq.getResult(),
        const0.getResult());
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        clkDeq.getResult(),
        moduleOp.getArgument(0));
    newBuilder.create<firrtl::StrictConnectOp>(
        newLoc,
        moduleOp.getArgument(7),
        dataDeq.getResult());

    return moduleOp;
}

// Help to determine the queue number
int queueNum = 0;
// SmallVector<>
struct ConvertChannel : OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ConvertChannel(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ChannelOp op,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const auto bufferSize = op.getBufferSize();
        assert(bufferSize && "need to specify the size of channel");

        auto channelTy = op.getEncapsulatedType();
        auto portTy = dyn_cast<IntegerType>(channelTy);
        assert(portTy && "only integer type is supported on hardware");
        auto portBit = portTy.getWidth();
        auto size = op.getBufferSize();
        assert(size && "cannot create infinite fifo buffer on hardware");
        auto suffix = (portTy.isSigned() ? "si" : "ui")
                      + std::to_string(portBit) + "_"
                      + std::to_string(size.value());
        auto name = "Queue_" + suffix;

        auto queueModule = queueList.find(name)->second;

        auto queueName =
            queueNum == 0 ? "queue" : "queue" + std::to_string(queueNum);
        auto queueOp = rewriter.create<firrtl::InstanceOp>(
            op.getLoc(),
            queueModule,
            rewriter.getStringAttr(queueName));
        queueNum++;

        SmallVector<Value> enqValues;
        for (int i = 2; i < 5; i++) enqValues.push_back(queueOp.getResult(i));
        newChannels.push_back(std::make_pair(enqValues, op.getResult(0)));
        SmallVector<Value> deqValues;
        for (int i = 5; i < 8; i++) deqValues.push_back(queueOp.getResult(i));
        newChannels.push_back(std::make_pair(deqValues, op.getResult(1)));

        rewriter.eraseOp(op);

        return success();
    }
};

// Helper to determine if an operator is already instantiated.
// If so, change the name by multiple instantiation.
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

struct ConvertInstantiate : OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    ConvertInstantiate(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto context = this->getContext();
        auto loc = op.getLoc();

        auto name = op.getCallee().str();
        auto fmoduleOp = operatorList.find(name)->second;

        auto nameIndex = getNameOfInstance(name);
        auto instanceName =
            nameIndex == 0 ? name : name + std::to_string(nameIndex);

        auto newInstance = rewriter.create<firrtl::InstanceOp>(
            loc,
            fmoduleOp,
            rewriter.getStringAttr(instanceName));

        auto inputs = op.getInputs();
        int i = 0;
        int j = 0;
        for (const auto &input : inputs) {
            j = 0;
            auto deqBundle = getNewArgOrOperand<SmallVector<Value>, Value>(
                input,
                newChannels);
            rewriter.create<firrtl::StrictConnectOp>(
                loc,
                deqBundle.value()[j++],
                newInstance.getResult(i++));
            rewriter.create<firrtl::StrictConnectOp>(
                loc,
                newInstance.getResult(i++),
                deqBundle.value()[j++]);
            rewriter.create<firrtl::StrictConnectOp>(
                loc,
                newInstance.getResult(i++),
                deqBundle.value()[j]);
        }
        auto outputs = op.getOutputs();
        for (const auto &output : outputs) {
            j = 0;
            auto enqBundle = getNewArgOrOperand<SmallVector<Value>, Value>(
                output,
                newChannels);
            rewriter.create<firrtl::StrictConnectOp>(
                loc,
                newInstance.getResult(i++),
                enqBundle.value()[j++]);
            rewriter.create<firrtl::StrictConnectOp>(
                loc,
                enqBundle.value()[j++],
                newInstance.getResult(i++));
            rewriter.create<firrtl::StrictConnectOp>(
                loc,
                enqBundle.value()[j],
                newInstance.getResult(i++));
        }

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
    patterns.add<ConvertPull>(typeConverter, patterns.getContext());
    patterns.add<ConvertPush>(typeConverter, patterns.getContext());
    // patterns.add<ConvertLoop>(typeConverter, patterns.getContext());
    patterns.add<ConvertChannel>(typeConverter, patterns.getContext());
    patterns.add<ConvertInstantiate>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDfgToCirctPass
        : public ConvertDfgToCirctBase<ConvertDfgToCirctPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToCirctPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });
    // TODO: need new conversion here, to convert Input/Output type
    // into FIRRTL type.
    // ??? how could we convert a signless(MLIR) into signed(FIRRTL)?
    // treat everything as UInt?
    // converter.addConversion([&](InputType type) -> Type {
    //     return converter.convertType(type.getElementType());
    // });
    // converter.addConversion([&](OutputType type) -> Type {
    //     return converter.convertType(type.getElementType());
    // });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<firrtl::FIRRTLDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();

    // Insert Queue at the top if there is one channel
    // also check if a specific queue already exists
    // std::map<unsigned, Type> queueList;
    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&module.getBodyRegion().front());
    module.walk([&](OperatorOp operatorOp) {
        operatorOp.walk([&](ChannelOp channelOp) {
            auto channelTy = channelOp.getEncapsulatedType();
            auto portTy = dyn_cast<IntegerType>(channelTy);
            assert(portTy && "only integer type is supported on hardware");
            auto portBit = portTy.getWidth();
            auto size = channelOp.getBufferSize();
            assert(size && "cannot create infinite fifo buffer on hardware");
            auto suffix = (portTy.isSigned() ? "si" : "ui")
                          + std::to_string(portBit) + "_"
                          + std::to_string(size.value());
            auto name = "Queue_" + suffix;
            auto isExist = queueList.find(name);
            if (isExist == queueList.end()) {
                auto queueModule = insertQueue(
                    builder,
                    &getContext(),
                    builder.getUnknownLoc(),
                    name,
                    portTy.isSigned(),
                    portBit,
                    size.value());
                queueList.emplace(name, queueModule);
            }
        });
    });
    // func::FuncOp top;
    // module.walk([&](func::FuncOp funcOp) { top = funcOp; });

    // Insert CircuitOp into module and move every op into it
    auto circuitOp = builder.create<firrtl::CircuitOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("TopModule")); // or, walk to find the last FuncOp
                                             // and then use its name
    for (auto &op :
         llvm::make_early_inc_range(module.getBodyRegion().getOps())) {
        if (dyn_cast<firrtl::CircuitOp>(op)) continue;
        op.moveBefore(
            circuitOp.getBodyBlock(),
            circuitOp.getBodyBlock()->end());
    }

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
