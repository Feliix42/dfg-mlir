/// Main entry point for the dfg-mlir optimizer driver.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "dfg-mlir/Conversion/Passes.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);
    circt::registerAllDialects(registry);

    registerAllPasses();
    dfg::registerConversionPasses();
    dfg::registerDfgPasses();
    circt::registerAllPasses();

    registry.insert<dfg::DfgDialect>();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "dfg-mlir optimizer driver\n", registry));
}
