//===- TranslateRegistration.cpp - Register translation -------------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//dfg/IR/Dialect.h"
#include "dfg-mlir/Target/VivadoTcl/VivadoTclEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
namespace dfg {

static llvm::cl::opt<std::string> TargetDevice(
    "vivado-target-device", // Unique name
    llvm::cl::desc("Target FPGA device for Vivado Tcl"),
    llvm::cl::init("xck26-sfvc784-2LV-c"));

void registerToVivadoTclTranslation()
{
    TranslateFromMLIRRegistration reg(
        "dfg-to-vivado-tcl",
        "translate from dfg top region to tcl",
        [](Operation* op, raw_ostream &output) {
            return dfg::translateToVivadoTcl(op, output, TargetDevice);
        },
        [](DialectRegistry &registry) {
            registry.insert<dfg::DfgDialect>();
            registry.insert<arith::ArithDialect>();
        });
}

} // namespace dfg
} // namespace mlir
