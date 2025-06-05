//===- TranslateRegistration.cpp - Register translation -------------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//emitHLS/IR/Dialect.h"
#include "dfg-mlir/Target/GenerateVitisProject/GenerateVitisProjectEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
namespace emitHLS {

static llvm::cl::opt<std::string> TargetDevice(
    "target-device", // Unique name
    llvm::cl::desc("Target FPGA device model name"),
    llvm::cl::init("xck26-sfvc784-2LV-c"));

static llvm::cl::opt<std::string> OutputDirOpt(
    "output-dir",
    llvm::cl::desc("Output directory for emitHLS project generation"),
    llvm::cl::value_desc("path to the directory"),
    llvm::cl::init("."));

void registerGenerateemitHLSProject()
{
    TranslateFromMLIRRegistration reg(
        "emitHLS-generate-project",
        "translate from emitHLS dialect to HLS file with scripts to generate "
        "the "
        "full project",
        [](Operation* op, raw_ostream &output) {
            return emitHLS::generateemitHLSProject(
                op,
                output,
                OutputDirOpt,
                TargetDevice);
        },
        [](DialectRegistry &registry) {
            registry.insert<emitHLS::emitHLSDialect>();
        });
}

} // namespace emitHLS
} // namespace mlir
