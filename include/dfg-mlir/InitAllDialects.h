/// Register all dialects in this project.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>

namespace mlir {


inline void registerAllDFGMLIRDialects(DialectRegistry &registry)
{
    registry.insert<dfg::DfgDialect>();
}

inline void registerAllDFGMLIRDialects(MLIRContext &context)
{
    DialectRegistry registry;
    registerAllDFGMLIRDialects(registry);
    context.appendDialectRegistry(registry);
}

} // namespace mlir
