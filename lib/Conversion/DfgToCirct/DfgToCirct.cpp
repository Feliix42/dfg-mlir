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

// Helper map&func to determine which three arguments should
// be use in the new FModule when an op inside is trying to
// use the old arguments.
std::map<int, Value> newArgNums;
std::optional<int> getArgNum(Value value)
{
    for (const auto &kv : newArgNums)
        if (kv.second == value) return kv.first;
    return std::nullopt;
}

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
        newArgNums.clear();
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
                newArgNums.emplace(port_num, arg);
                ports.push_back(firrtl::PortInfo(
                    rewriter.getStringAttr(
                        "io_in" + std::to_string(in_num) + "_ready"),
                    firrtl::UIntType::get(context, 1),
                    firrtl::Direction::Out));
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
                newArgNums.emplace(port_num, arg);
                ports.push_back(firrtl::PortInfo(
                    rewriter.getStringAttr(
                        "io_out" + std::to_string(out_num) + "_valid"),
                    firrtl::UIntType::get(context, 1),
                    firrtl::Direction::Out));
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

// struct ConvertPull : OpConversionPattern<PullOp> {
//     using OpConversionPattern<PullOp>::OpConversionPattern;

//     ConvertPull(TypeConverter &typeConverter, MLIRContext* context)
//             : OpConversionPattern<PullOp>(typeConverter, context){};

//     LogicalResult matchAndRewrite(
//         PullOp op,
//         PullOpAdaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         rewriter.eraseOp(op);

//         return success();
//     }
// };

// struct ConvertPush : OpConversionPattern<PushOp> {
//     using OpConversionPattern<PushOp>::OpConversionPattern;

//     ConvertPush(TypeConverter &typeConverter, MLIRContext* context)
//             : OpConversionPattern<PushOp>(typeConverter, context){};

//     LogicalResult matchAndRewrite(
//         PushOp op,
//         PushOpAdaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         rewriter.eraseOp(op);

//         return success();
//     }
// };

void insertQueue(
    OpBuilder &builder,
    MLIRContext* context,
    Location loc,
    Type dataTy,
    std::optional<int> size)
{
    assert(size && "cannot create infinite fifo buffer on hardware");
    auto portTy = dyn_cast<IntegerType>(dataTy);
    assert(
        !portTy.isSignless()
        && "signless integers are not supported in firrtl");

    auto sizeNum = size.value();
    auto suffix = (portTy.isSigned() ? "si" : "ui")
                  + std::to_string(portTy.getWidth()) + "_"
                  + std::to_string(sizeNum);

    // Arguments needed to create a FModuleOp for Queue
    auto name = builder.getStringAttr("Queue_" + suffix);
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
    if (portTy.isSigned()) {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_enq_bits"),
            firrtl::SIntType::get(context, sizeNum),
            firrtl::Direction::In));
    } else if (portTy.isUnsigned()) {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_enq_bits"),
            firrtl::UIntType::get(context, sizeNum),
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
    if (portTy.isSigned()) {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_deq_bits"),
            firrtl::SIntType::get(context, sizeNum),
            firrtl::Direction::Out));
    } else if (portTy.isUnsigned()) {
        ports.push_back(firrtl::PortInfo(
            builder.getStringAttr("io_deq_bits"),
            firrtl::UIntType::get(context, sizeNum),
            firrtl::Direction::Out));
    }

    auto moduleOp =
        builder.create<firrtl::FModuleOp>(loc, name, convention, ports);

    // Create the Queue behavior one by one
    OpBuilder newBuilder(context);
    newBuilder.setInsertionPointToEnd(&moduleOp.getBodyRegion().front());
    auto newLoc = newBuilder.getUnknownLoc();

    auto bitDepth = std::ceil(std::log2(sizeNum));
    auto isTwoPower = sizeNum > 0 && (sizeNum & (sizeNum - 1)) == 0;
    newBuilder.create<firrtl::ConstantOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        llvm::APInt(/*numBits=*/1, /*value=*/1));
    newBuilder.create<firrtl::ConstantOp>(
        newLoc,
        firrtl::UIntType::get(context, 1),
        llvm::APInt(1, 0));
    if (bitDepth > 1) {
        newBuilder.create<firrtl::ConstantOp>(
            newLoc,
            firrtl::UIntType::get(context, bitDepth),
            llvm::APInt(bitDepth, 0));
        if (!isTwoPower) {
            newBuilder.create<firrtl::ConstantOp>(
                newLoc,
                firrtl::UIntType::get(context, bitDepth),
                llvm::APInt(bitDepth, sizeNum - 1));
        }
    }
    // newBuilder.create<firrtl::MemOp>();
}

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
    // patterns.add<ConvertPull>(typeConverter, patterns.getContext());
    // patterns.add<ConvertPush>(typeConverter, patterns.getContext());
    // patterns.add<ConvertLoop>(typeConverter, patterns.getContext());
    patterns.add<ConvertChannel>(typeConverter, patterns.getContext());
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
    converter.addConversion([&](InputType type) -> Type {
        return converter.convertType(type.getElementType());
    });
    converter.addConversion([&](OutputType type) -> Type {
        return converter.convertType(type.getElementType());
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<firrtl::FIRRTLDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();

    // if (failed(applyPartialConversion(
    //         getOperation(),
    //         target,
    //         std::move(patterns)))) {
    //     signalPassFailure();
    // }

    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&module.getBodyRegion().front());
    module.walk([&](OperatorOp operatorOp) {
        operatorOp.walk([&](ChannelOp channelOp) {
            insertQueue(
                builder,
                &getContext(),
                builder.getUnknownLoc(),
                channelOp.getEncapsulatedType(),
                channelOp.getBufferSize());
        });
    });
    // func::FuncOp top;
    // module.walk([&](func::FuncOp funcOp) { top = funcOp; });
    auto circuitOp = builder.create<firrtl::CircuitOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("TopModule"));
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
