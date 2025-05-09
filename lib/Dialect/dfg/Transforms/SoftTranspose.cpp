/// Implementation of SoftTranspose transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/SoftTranspose.h"

#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace linalg {
#define GEN_PASS_DEF_LINALGSOFTTRANSPOSE
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace linalg
} // namespace mlir

using namespace mlir;
using namespace linalg;

namespace {
template<typename T>
void getTransposedVector(
    std::vector<T> data,
    std::vector<int64_t> oldShape,
    std::vector<int64_t> newShape,
    std::vector<int64_t> permutation,
    std::vector<T> &newData)
{
    auto numDims = oldShape.size();
    // First, calculate the stride
    std::vector<int64_t> oldStrides(numDims, 1);
    for (int i = numDims - 2; i >= 0; --i)
        oldStrides[i] = oldStrides[i + 1] * oldShape[i + 1];
    std::vector<int64_t> newStrides(numDims, 1);
    for (int i = numDims - 2; i >= 0; --i)
        newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    // Transpose based on permutation
    for (auto [flatIdx, value] : llvm::enumerate(data)) {
        // Turn flat index into multi-dim
        std::vector<int64_t> indices(numDims);
        auto remaining = flatIdx;
        for (unsigned i = 0; i < numDims; ++i) {
            indices[i] = remaining / oldStrides[i];
            remaining %= oldStrides[i];
        }
        // Get new index based on permutation
        std::vector<int64_t> newIndices(numDims);
        for (unsigned i = 0; i < numDims; ++i)
            newIndices[i] = indices[permutation[i]];
        // Flatten new index
        unsigned newFlatIdx = 0;
        for (unsigned i = 0; i < numDims; ++i)
            newFlatIdx += newIndices[i] * newStrides[i];
        // Copy
        newData[newFlatIdx] = value;
    }
}
struct TransposeAtPlace : public OpRewritePattern<TransposeOp> {
    TransposeAtPlace(MLIRContext* context)
            : OpRewritePattern<TransposeOp>(context){};

    LogicalResult
    matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override
    {
        auto input = op.getInput();
        auto inputType = input.getType();
        auto shape = inputType.getShape().vec();
        auto elemTy = inputType.getElementType();
        auto init = op.getInit();
        auto initType = init.getType();
        auto shapeT = initType.getShape().vec();
        auto permutation = op.getPermutation().vec();

        auto defOp = dyn_cast<arith::ConstantOp>(input.getDefiningOp());
        auto defLoc = defOp.getLoc();
        auto denseAttr = dyn_cast<DenseElementsAttr>(defOp.getValueAttr());
        if (!denseAttr)
            return rewriter.notifyMatchFailure(
                defLoc,
                "Incorrect constant, expected dense value!");

        // Transpose
        DenseElementsAttr newDenseAttr;
        if (isa<IntegerType>(elemTy)) {
            auto values = denseAttr.getValues<llvm::APInt>();
            std::vector<llvm::APInt> integers(values.begin(), values.end());
            std::vector<llvm::APInt> newIntegers(
                integers.size(),
                llvm::APInt());
            getTransposedVector<llvm::APInt>(
                integers,
                shape,
                shapeT,
                permutation,
                newIntegers);
            newDenseAttr = DenseElementsAttr::get(initType, newIntegers);
        } else if (isa<FloatType>(elemTy)) {
            auto values = denseAttr.getValues<llvm::APFloat>();
            std::vector<llvm::APFloat> floats(values.begin(), values.end());
            std::vector<llvm::APFloat> newFloats(
                floats.size(),
                llvm::APFloat(0.0));
            getTransposedVector<llvm::APFloat>(
                floats,
                shape,
                shapeT,
                permutation,
                newFloats);
            newDenseAttr = DenseElementsAttr::get(initType, newFloats);
        }

        rewriter.setInsertionPoint(defOp);
        auto newConst =
            rewriter.create<arith::ConstantOp>(defLoc, initType, newDenseAttr);
        op.getResult().front().replaceAllUsesWith(newConst.getResult());
        rewriter.eraseOp(op);
        rewriter.eraseOp(defOp);
        rewriter.eraseOp(init.getDefiningOp());
        return success();
    }
};
} // namespace

void mlir::linalg::populateSoftTransposeConversionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<TransposeAtPlace>(patterns.getContext());
}

namespace {
struct LinalgSoftTransposePass : public linalg::impl::LinalgSoftTransposeBase<
                                     LinalgSoftTransposePass> {
    void runOnOperation() override;
};
} // namespace

void LinalgSoftTransposePass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateSoftTransposeConversionPatterns(patterns);

    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });
    target.addDynamicallyLegalOp<TransposeOp>([](TransposeOp op) {
        auto input = op.getInput();
        auto defOp = input.getDefiningOp();
        if (isa<arith::ConstantOp>(defOp)) return false;
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::linalg::createLinalgSoftTransposePass()
{
    return std::make_unique<LinalgSoftTransposePass>();
}
