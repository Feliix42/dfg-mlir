/// Implementation of StdToCirct pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "../PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "dfg-mlir/Conversion/StdToCirct/StdToCirct.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace mlir::dfg;
using namespace circt;

namespace {} // namespace

void mlir::populateStdToCirctConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{}

namespace {
struct ConvertStdToCirctPass
        : public ConvertStdToCirctBase<ConvertStdToCirctPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertStdToCirctPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateStdToCirctConversionPatterns(converter, patterns);

    target.addLegalDialect<firrtl::FIRRTLDialect>();
    target.addIllegalDialect<arith::ArithDialect>();

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
