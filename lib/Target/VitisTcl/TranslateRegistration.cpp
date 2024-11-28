//===- TranslateRegistration.cpp - Register translation -------------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Dialect.h"
#include "dfg-mlir/Target/VitisTcl/VitisTclEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
namespace vitis {

static llvm::cl::opt<std::string> TargetDevice(
    "target-device",
    llvm::cl::desc("Target FPGA device"),
    llvm::cl::init("xck26-sfvc784-2LV-c"));

void registerToVitisTclTranslation()
{
    TranslateFromMLIRRegistration reg(
        "vitis-to-tcl",
        "translate from vitis dialect to tcl",
        [](Operation* op, raw_ostream &output) {
            return vitis::translateToVitisTcl(op, output, TargetDevice);
        },
        [](DialectRegistry &registry) {
            registry.insert<vitis::VitisDialect>();
        });
}

} // namespace vitis
} // namespace mlir
