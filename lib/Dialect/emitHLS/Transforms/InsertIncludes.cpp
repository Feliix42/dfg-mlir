/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/emitHLS/Transforms/InsertIncludes.h"

#include "dfg-mlir/Dialect/emitHLS/IR/Ops.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Types.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir {
namespace emitHLS {
#define GEN_PASS_DEF_EMITHLSINSERTINCLUDES
#include "dfg-mlir/Dialect/emitHLS/Transforms/Passes.h.inc"
} // namespace emitHLS
} // namespace mlir

using namespace mlir;
using namespace emitHLS;

namespace {
struct emitHLSInsertIncludesPass
        : public emitHLS::impl::emitHLSInsertIncludesBase<
              emitHLSInsertIncludesPass> {
    void runOnOperation() override;
};
} // namespace

void emitHLSInsertIncludesPass::runOnOperation()
{
    ModuleOp module = cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    auto loc = module.getLoc();
    builder.setInsertionPointToStart(&module.getBodyRegion().front());

    bool hasSizeT = false;
    bool hasStream = false;
    bool hasInteger = false;
    bool hasFixed = false;
    bool hasMath = false;

    auto processIntegerFixedPoint = [&](Type type) {
        if (!hasInteger && isa<IntegerType>(type))
            hasInteger = true;
        else if (!hasFixed && isa<APFixedType, APFixedUType>(type))
            hasFixed = true;
        else if (!hasSizeT && isa<IndexType>(type))
            hasSizeT = true;
    };

    module.walk([&](Operation* op) {
        // If any of the operands or results of an op is integer
        if (!op->getOperands().empty()) {
            for (auto operandTy : op->getOperandTypes()) {
                processIntegerFixedPoint(operandTy);
                if (!hasStream && isa<StreamType>(operandTy)) {
                    hasStream = true;
                    processIntegerFixedPoint(
                        cast<StreamType>(operandTy).getStreamType());
                }
            }
        }
        if (!op->getResults().empty()) {
            for (auto resultTy : op->getResultTypes()) {
                processIntegerFixedPoint(resultTy);
                if (!hasStream && isa<StreamType>(resultTy)) {
                    hasStream = true;
                    processIntegerFixedPoint(
                        cast<StreamType>(resultTy).getStreamType());
                }
            }
        }
        // If there is stream type in func op signature
        if (auto funcOp = dyn_cast<FuncOp>(op)) {
            auto inputs = funcOp.getFunctionType().getInputs();
            for (auto inTy : inputs) {
                if (!hasStream) {
                    if (auto streamTy = dyn_cast<StreamType>(inTy)) {
                        hasStream = true;
                        processIntegerFixedPoint(streamTy.getStreamType());
                    }
                }
            }
        }
        // If there is any math ops
        if (isa<MathSinOp, MathCosOp>(op)) hasMath = true;
    });

    if (hasSizeT)
        builder.create<IncludeOp>(loc, builder.getStringAttr("cstddef"));
    if (hasInteger)
        builder.create<IncludeOp>(loc, builder.getStringAttr("ap_int.h"));
    if (hasFixed)
        builder.create<IncludeOp>(loc, builder.getStringAttr("ap_fixed.h"));
    if (hasStream)
        builder.create<IncludeOp>(loc, builder.getStringAttr("hls_stream.h"));
    if (hasMath)
        builder.create<IncludeOp>(loc, builder.getStringAttr("hls_math.h"));
}

std::unique_ptr<Pass> mlir::emitHLS::createEmitHLSInsertIncludesPass()
{
    return std::make_unique<emitHLSInsertIncludesPass>();
}
