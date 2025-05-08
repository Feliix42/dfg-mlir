/// Implementation of RemoveScalarGlobals transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/RemoveScalarGlobals.h"

#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_MEMREFREMOVESCALARGLOBALS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;
using namespace memref;

namespace {
std::map<std::string, TypedAttr> scalarMemrefMap;
struct EraseScalarMemrefGlobal : public OpRewritePattern<GlobalOp> {
    EraseScalarMemrefGlobal(MLIRContext* context)
            : OpRewritePattern<GlobalOp>(context){};

    LogicalResult
    matchAndRewrite(GlobalOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto name = op.getName().str();
        auto initValueAttr =
            dyn_cast<DenseElementsAttr>(op.getInitialValue().value());
        auto initValueType = initValueAttr.getElementType();
        // Store the name and the number and erase
        LogicalResult status =
            llvm::TypeSwitch<Type, LogicalResult>(initValueType)
                .Case([&](IntegerType type) {
                    auto value = initValueAttr.getSplatValue<llvm::APInt>();
                    auto attr = rewriter.getIntegerAttr(type, value);
                    scalarMemrefMap.insert({name, attr});
                    return success();
                })
                .Case([&](FloatType type) {
                    auto value = initValueAttr.getSplatValue<llvm::APFloat>();
                    auto attr = rewriter.getFloatAttr(type, value);
                    scalarMemrefMap.insert({name, attr});
                    return success();
                })
                .Default([&](Type) { return failure(); });
        if (failed(status))
            return rewriter.notifyMatchFailure(
                loc,
                "Unknown type in global op @" + name);
        rewriter.eraseOp(op);
        return success();
    }
};
struct MakeGetGlobalConstant : public OpRewritePattern<GetGlobalOp> {
    MakeGetGlobalConstant(MLIRContext* context)
            : OpRewritePattern<GetGlobalOp>(context){};

    LogicalResult
    matchAndRewrite(GetGlobalOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto name = op.getName().str();
        auto memref = op.getResult();
        auto type = memref.getType().getElementType();
        auto constantOp = rewriter.create<arith::ConstantOp>(
            loc,
            type,
            scalarMemrefMap[name]);

        // Change linalg.generic operation
        IRMapping mapper;
        for (auto opi : memref.getUsers()) {
            // Before using this pass, it should be at linalg.generic level
            if (auto genericOp = dyn_cast<linalg::GenericOp>(opi)) {
                // Information from original generic
                auto genericInputs = genericOp.getInputs();
                Block* genericBlock = genericOp.getBlock();
                auto affineMaps = genericOp.getIndexingMapsArray();
                // The usage of the memref must be in the inputs
                SmallVector<Value> newInputs;
                for (auto [idx, input] : llvm::enumerate(genericInputs)) {
                    if (input == memref) {
                        // Replace the affine map
                        auto affineMap = affineMaps[idx];
                        auto newMap = AffineMap::get(
                            affineMap.getNumDims(),
                            0,
                            affineMap.getContext());
                        affineMaps[idx] = newMap;
                        newInputs.push_back(constantOp.getResult());
                    } else {
                        newInputs.push_back(input);
                    }
                }
                // Replace the old generic with new one
                rewriter.setInsertionPoint(genericOp);
                rewriter.replaceOpWithNewOp<linalg::GenericOp>(
                    genericOp,
                    newInputs,
                    genericOp.getOutputs(),
                    affineMaps,
                    genericOp.getIteratorTypesArray(),
                    /*doc*/ rewriter.getStringAttr(""),
                    /*libraryCall*/ rewriter.getStringAttr(""),
                    [&](OpBuilder &opBuilder,
                        Location loc,
                        ValueRange blockArgs) {
                        for (auto [oldArg, newArg] :
                             llvm::zip(genericBlock->getArguments(), blockArgs))
                            mapper.map(oldArg, newArg);
                        for (auto &opOrig : genericBlock->getOperations())
                            opBuilder.clone(opOrig, mapper);
                    });
            }
        }
        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

void mlir::memref::populateRemoveScalarGlobalsConversionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<EraseScalarMemrefGlobal>(patterns.getContext());
    patterns.add<MakeGetGlobalConstant>(patterns.getContext());
}

namespace {
struct MemrefRemoveScalarGlobalsPass
        : public memref::impl::MemrefRemoveScalarGlobalsBase<
              MemrefRemoveScalarGlobalsPass> {
    void runOnOperation() override;
};
} // namespace

void MemrefRemoveScalarGlobalsPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateRemoveScalarGlobalsConversionPatterns(patterns);

    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });
    target.addDynamicallyLegalOp<GlobalOp>([](GlobalOp op) {
        if (auto initValue = op.getInitialValue(); initValue) {
            auto values = dyn_cast<DenseElementsAttr>(initValue.value());
            return !values.isSplat();
        }
        return true;
    });
    target.addDynamicallyLegalOp<GetGlobalOp>([&](GetGlobalOp op) {
        auto globalName = op.getName();
        Operation* symbolOp = SymbolTable::lookupSymbolIn(
            op->getParentOfType<ModuleOp>(),
            globalName);
        if (auto globalOp = dyn_cast<GlobalOp>(symbolOp); globalOp)
            return !target.isIllegal(globalOp);
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::memref::createMemrefRemoveScalarGlobalsPass()
{
    return std::make_unique<MemrefRemoveScalarGlobalsPass>();
}
