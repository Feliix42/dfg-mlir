//===- TranslateRegistration.cpp - Register translation -------------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Dialect.h"
#include "dfg-mlir/Target/VitisCpp/VitisEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
namespace vitis {

void registerToVitisCppTranslation()
{
    TranslateFromMLIRRegistration reg(
        "vitis-to-cpp",
        "translate from vitis dialect to cpp",
        [](Operation* op, raw_ostream &output) {
            return vitis::translateToVitisCpp(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<vitis::VitisDialect>();
        });
}

} // namespace vitis
} // namespace mlir
