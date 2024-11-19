/// Main entry point for the dfg-mlir optimizer driver.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "circt/InitAllTranslations.h"
#include "dfg-mlir/Target/VitisCpp/VitisEmitter.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

namespace mlir {
namespace vitis {
void registerToVitisCppTranslation();
} // namespace vitis
} // namespace mlir

int main(int argc, char* argv[])
{
    registerAllTranslations();
    circt::registerAllTranslations();
    vitis::registerToVitisCppTranslation();

    return failed(
        mlirTranslateMain(argc, argv, "DFG Translation Testing Tool"));
}
