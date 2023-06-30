/// Implementation of DfgToCirct pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "../PassDetails.h"
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
        TypeConverter* converter = this->typeConverter;

        auto name = op.getSymNameAttr();
        // auto inputTypes = op.getFunctionType().getInputs();
        // auto outputTypes = op.getFunctionType().getResults();

        // SmallVector<Type> inputs;
        // for (const auto inTy : inputTypes)
        //     inputs.push_back(converter->convertType(inTy));

        // SmallVector<Type> outputs;
        // for (const auto outTy : outputTypes)
        //     inputs.push_back(converter->convertType(outTy));

        // auto fnTy = rewriter.getFunctionType(inputs, outputs);

        auto funcOp = rewriter.create<handshake::FuncOp>(
            op.getLoc(),
            name,
            op.getFunctionType());
        rewriter.inlineRegionBefore(
            op.getBody(),
            funcOp.getBody(),
            funcOp.end());
        funcOp.setPrivate();

        rewriter.eraseOp(op);

        return success();
    }
};

struct ConvertEnd : OpConversionPattern<EndOp> {
    using OpConversionPattern<EndOp>::OpConversionPattern;

    ConvertEnd(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<EndOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        EndOp op,
        EndOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        ValueRange values;
        rewriter.create<handshake::ReturnOp>(op.getLoc(), values);

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
    patterns.add<ConvertEnd>(typeConverter, patterns.getContext());
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
        return converter.convertType(type.getElementType());
    });
    converter.addConversion([&](OutputType type) -> Type {
        return converter.convertType(type.getElementType());
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<handshake::HandshakeDialect>();
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
