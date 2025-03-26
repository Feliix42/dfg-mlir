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
namespace dfg {

inline void registerAllDialects(DialectRegistry &registry)
{
    registry.insert<dfg::DfgDialect, vitis::VitisDialect>();

    dfg::registerBufferizableOpInterfaceExternalModels(registry);
}

inline void registerAllDialects(MLIRContext &context)
{
    DialectRegistry registry;
    dfg::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
}

} // namespace dfg
} // namespace mlir
