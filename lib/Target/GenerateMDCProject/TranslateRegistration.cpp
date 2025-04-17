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

void registerToMDCTranslation() {
    TranslateFromMLIRRegistration reg(
        "dfg-to-mdc",
        "Translate DFG to MDC format",
        [](Operation* op, raw_ostream& output) -> LogicalResult {
            return generateMDCProject(op, output);
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