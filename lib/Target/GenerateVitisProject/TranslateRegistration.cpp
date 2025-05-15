//===- TranslateRegistration.cpp - Register translation -------------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Dialect.h"
#include "dfg-mlir/Target/GenerateVitisProject/GenerateVitisProjectEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
namespace vitis {

static llvm::cl::opt<std::string> TargetDevice(
    "target-device", // Unique name
    llvm::cl::desc("Target FPGA device model name"),
    llvm::cl::init("xck26-sfvc784-2LV-c"));

static llvm::cl::opt<std::string> OutputDirOpt(
    "output-dir",
    llvm::cl::desc("Output directory for Vitis project generation"),
    llvm::cl::value_desc("path to the directory"),
    llvm::cl::init("."));

static llvm::cl::opt<std::string> formdc(
    "for-MDC",
    llvm::cl::desc("Vitis project generation for MDC"),
    llvm::cl::value_desc("generation files for MDC"),
    llvm::cl::init("false"));

void registerGenerateVitisProject()
{
    TranslateFromMLIRRegistration reg(
        "vitis-generate-project",
        "translate from vitis dialect to HLS file with scripts to generate the "
        "full project",
        [](Operation* op, raw_ostream &output) {
            return vitis::generateVitisProject(
                op,
                output,
                OutputDirOpt,
                formdc,
                TargetDevice);
        },
        [](DialectRegistry &registry) {
            registry.insert<vitis::VitisDialect>();
        });
}

} // namespace vitis
} // namespace mlir
