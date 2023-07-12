/// Main entry point for the dfg-mlir optimizer driver.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "circt/InitAllDialects.h"
#include "dfg-mlir/Conversion/Passes.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registerAllPasses();
    dfg::registerConversionPasses();

    registry.insert<dfg::DfgDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "dfg-mlir optimizer driver\n", registry));
}
