/// Implementation of MemrefToVitis pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/MemrefToVitis/MemrefToVitis.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::vitis;

namespace {
template<typename OpFrom>
struct ConvertMemrefAlloc : OpConversionPattern<OpFrom> {
    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertMemrefAlloc(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OpFrom op,
        typename OpFrom::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto converter = this->getTypeConverter();
        auto memrefTy = op.getType();
        auto newVarOp = rewriter.create<VariableOp>(
            op.getLoc(),
            converter->convertType(memrefTy));

        for (auto user : op.getResult().getUsers()) {
            rewriter.setInsertionPoint(user);
            // If it's used in memref.load
            if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
                auto arrayRead = rewriter.create<ArrayReadOp>(
                    loadOp.getLoc(),
                    memrefTy.getElementType(),
                    newVarOp.getResult(),
                    loadOp.getIndices());
                rewriter.replaceOp(user, arrayRead);
            }
            // If it's used in memref.store
            else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
                auto arrayWrite = rewriter.create<ArrayWriteOp>(
                    storeOp.getLoc(),
                    storeOp.getValueToStore(),
                    newVarOp.getResult(),
                    storeOp.getIndices());
                rewriter.replaceOp(user, arrayWrite);
            }
            // If it's used in dfg.push
            else if (auto pushOp = dyn_cast<dfg::PushOp>(user)) {
                auto unrealizedCastOp =
                    rewriter.create<UnrealizedConversionCastOp>(
                        pushOp.getLoc(),
                        memrefTy,
                        newVarOp.getResult());
                auto newPush = rewriter.create<dfg::PushOp>(
                    pushOp.getLoc(),
                    unrealizedCastOp.getResult(0),
                    pushOp.getChan());
                rewriter.replaceOp(user, newPush);
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertPullBoundMemrefLoad : OpConversionPattern<memref::LoadOp> {
    using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

    ConvertPullBoundMemrefLoad(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<memref::LoadOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        memref::LoadOp op,
        memref::LoadOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto converter = this->getTypeConverter();
        auto memref = op.getMemRef();
        auto memrefTy = dyn_cast<MemRefType>(memref.getType());
        auto definingOp = memref.getDefiningOp();
        if (!isa<dfg::PullOp>(definingOp)) return success();

        auto pullOp = dyn_cast<dfg::PullOp>(definingOp);
        auto unrealizedCastOp = rewriter.create<UnrealizedConversionCastOp>(
            pullOp.getLoc(),
            converter->convertType(memrefTy),
            pullOp.getResult());
        auto arrayRead = rewriter.create<ArrayReadOp>(
            op.getLoc(),
            memrefTy.getElementType(),
            unrealizedCastOp.getResult(0),
            op.getIndices());

        rewriter.replaceOp(op, arrayRead);
        return success();
    }
};
} // namespace

void mlir::populateMemrefToVitisConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertMemrefAlloc<memref::AllocOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertMemrefAlloc<memref::AllocaOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertPullBoundMemrefLoad>(
        typeConverter,
        patterns.getContext());
}

namespace {
struct ConvertMemrefToVitisPass
        : public impl::ConvertMemrefToVitisBase<ConvertMemrefToVitisPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertMemrefToVitisPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](MemRefType memrefTy) {
        return vitis::ArrayType::get(
            memrefTy.getShape(),
            memrefTy.getElementType());
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateMemrefToVitisConversionPatterns(converter, patterns);

    target.addLegalDialect<vitis::VitisDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertMemrefToVitisPass()
{
    return std::make_unique<ConvertMemrefToVitisPass>();
}
