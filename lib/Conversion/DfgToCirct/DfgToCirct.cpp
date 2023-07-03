/// Implementation of DfgToCirct pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "../PassDetails.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dfg-mlir/Conversion/DfgToCirct/DfgToCirct.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {

struct ConvertOperator : OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    ConvertOperator(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp op,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Create new FunctionType using both
        // input and output ports
        SmallVector<Type> inputTypes;
        for (const auto inTy : op.getInputTypes()) inputTypes.push_back(inTy);
        for (const auto outTy : op.getOutputTypes())
            inputTypes.push_back(outTy);
        auto newFuncTy = rewriter.getFunctionType(inputTypes, {});

        // create a handshake.func using the same name
        // and function_type of dfg.operator
        auto funcOp = rewriter.create<handshake::FuncOp>(
            op.getLoc(),
            op.getSymNameAttr(),
            newFuncTy);

        // copy all the region inside the FuncOp
        rewriter.inlineRegionBefore(
            op.getBody(),
            funcOp.getBody(),
            funcOp.end());

        funcOp.setPrivate();
        if (!funcOp.isExternal()) funcOp.resolveArgAndResNames();

        // new builder to insert func.return
        auto builder = OpBuilder::atBlockEnd(&funcOp.getBody().front());

        // return void
        SmallVector<Value> values;
        builder.create<handshake::ReturnOp>(funcOp.getLoc(), values);

        // Erase the original OperatorOp
        rewriter.eraseOp(op);

        return success();
    }
};

struct ConvertPull : OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    ConvertPull(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
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
    converter.addConversion([&](InputType type) -> Type {
        return converter.convertType(
            esi::ChannelType::get(&getContext(), type.getElementType()));
    });
    converter.addConversion([&](OutputType type) -> Type {
        return converter.convertType(
            esi::ChannelType::get(&getContext(), type.getElementType()));
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToCirctConversionPatterns(converter, patterns);

    populateFunctionOpInterfaceTypeConversionPattern<handshake::FuncOp>(
        patterns,
        converter);
    target.addDynamicallyLegalOp<handshake::FuncOp>([&](handshake::FuncOp op) {
        return converter.isSignatureLegal(op.getFunctionType())
               && converter.isLegal(&op.getBody());
    });

    target.addLegalDialect<esi::ESIDialect, handshake::HandshakeDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();

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
