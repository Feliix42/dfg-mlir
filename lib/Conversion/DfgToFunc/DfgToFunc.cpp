/// Implementation of DfgToAsync pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"

// #include "../PassDetails.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct ConvertDfgToFuncPass
        : public ConvertDfgToFuncBase<ConvertDfgToFuncPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToFuncPass::runOnOperation()
{
    // TODO
}

std::unique_ptr<Pass> mlir::createConvertDfgToFuncPass()
{
    return std::make_unique<ConvertDfgToFuncPass>();
}