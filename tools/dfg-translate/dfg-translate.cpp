/// Main entry point for the dfg-mlir optimizer driver.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/InitAllTranslations.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

// namespace mlir {
// namespace vitis {
// void registerToVitisCppTranslation();
// void registerToVitisTclTranslation();
// } // namespace vitis
// namespace dfg {
// void registerToVivadoTclTranslation();
// } // namespace dfg
// } // namespace mlir

int main(int argc, char* argv[])
{
    registerAllTranslations();
    dfg::registerAllTranslations();

    return failed(
        mlirTranslateMain(argc, argv, "DFG Translation Testing Tool"));
}
