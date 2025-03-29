/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/vitis/Transforms/InsertIncludes.h"

#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {
namespace vitis {
#define GEN_PASS_DEF_VITISINSERTINCLUDES
#include "dfg-mlir/Dialect/vitis/Transforms/Passes.h.inc"
} // namespace vitis
} // namespace mlir

using namespace mlir;
using namespace vitis;

namespace {
struct VitisInsertIncludesPass
        : public vitis::impl::VitisInsertIncludesBase<VitisInsertIncludesPass> {
    void runOnOperation() override;
};
} // namespace

void VitisInsertIncludesPass::runOnOperation()
{
    ModuleOp module = cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    auto loc = module.getLoc();
    builder.setInsertionPointToStart(&module.getBodyRegion().front());

    bool hasStream = false;
    bool hasInteger = false;
    bool hasFixed = false;
    bool hasMath = false;

    module.walk([&](Operation* op) {
        // If any of the operands or results of an op is integer
        if (!op->getOperands().empty()) {
            for (auto operandTy : op->getOperandTypes())
                if (!hasInteger && isa<IntegerType>(operandTy))
                    hasInteger = true;
                else if (!hasFixed && isa<APFixedType, APFixedUType>(operandTy))
                    hasFixed = true;
        }
        if (!op->getResults().empty()) {
            for (auto resultTy : op->getResultTypes())
                if (!hasInteger && isa<IntegerType>(resultTy))
                    hasInteger = true;
                else if (!hasFixed && isa<APFixedType, APFixedUType>(resultTy))
                    hasFixed = true;
        }
        // If there is stream type in func op signature
        if (auto funcOp = dyn_cast<FuncOp>(op)) {
            auto inputs = funcOp.getFunctionType().getInputs();
            for (auto inTy : inputs) {
                if (!hasStream) {
                    if (auto streamTy = dyn_cast<StreamType>(inTy))
                        hasStream = true;
                }
            }
        }
        // If there is any math ops
        if (isa<MathSinOp, MathCosOp>(op)) hasMath = true;
    });

    if (hasInteger)
        builder.create<IncludeOp>(loc, builder.getStringAttr("ap_int.h"));
    if (hasFixed)
        builder.create<IncludeOp>(loc, builder.getStringAttr("ap_fixed.h"));
    if (hasStream)
        builder.create<IncludeOp>(loc, builder.getStringAttr("hls_stream.h"));
    if (hasMath)
        builder.create<IncludeOp>(loc, builder.getStringAttr("hls_math.h"));
}

std::unique_ptr<Pass> mlir::vitis::createVitisInsertIncludesPass()
{
    return std::make_unique<VitisInsertIncludesPass>();
}
