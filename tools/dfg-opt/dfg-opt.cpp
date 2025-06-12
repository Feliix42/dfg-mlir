/// Main entry point for the dfg-mlir optimizer driver.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/InitAllDialects.h"
#include "dfg-mlir/InitAllPasses.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    DialectRegistry registry;

    registerAllDialects(registry);
    registerAllDFGMLIRDialects(registry);
    registerPrepareForMdcPipelines();
    registerAllPasses();
    registerAllDFGMLIRPasses();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "dfg-mlir optimizer driver\n", registry));
}
