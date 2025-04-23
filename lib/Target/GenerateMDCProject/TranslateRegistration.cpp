// dfg-mlir/lib/Target/GenerateMDCProject/TranslateRegistration.cpp
#include "mlir/Tools/mlir-translate/Translation.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h" 
#include "dfg-mlir/Target/GenerateMDCProject/GenerateMDCProjectEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir {
    namespace dfg {
        static llvm::cl::opt<std::string> calOutputDir(
            "mdc-output-dir",
            llvm::cl::desc("Directory to store CAL files"),
            llvm::cl::init(".")); // Default: current directory
void registerToMDCTranslation() {
    TranslateFromMLIRRegistration reg(
        "dfg-to-mdc",
        "Translate DFG to MDC format",
        [](Operation* op, raw_ostream& output) -> LogicalResult {
            return generateMDCProject(op, calOutputDir);
        },
        [](DialectRegistry& registry) {
            registry.insert<dfg::DfgDialect>();
            
            registry.insert<arith::ArithDialect>();
            registry.insert<index::IndexDialect>();
            registry.insert<math::MathDialect>();
            registry.insert<memref::MemRefDialect>();
            registry.insert<scf::SCFDialect>();            
        });
}

} // namespace dfg
} // namespace mlir