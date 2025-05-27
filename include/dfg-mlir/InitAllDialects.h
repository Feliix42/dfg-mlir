/// Register all dialects in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/Transforms/BufferizableOpInterfaceImpl.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>

namespace mlir {

inline void registerAllDFGMLIRDialects(DialectRegistry &registry)
{
    registry.insert<dfg::DfgDialect, vitis::VitisDialect>();

    dfg::registerBufferizableOpInterfaceExternalModels(registry);
}

inline void registerAllDFGMLIRDialects(MLIRContext &context)
{
    DialectRegistry registry;
    registerAllDFGMLIRDialects(registry);
    context.appendDialectRegistry(registry);
}

} // namespace mlir
