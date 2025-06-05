/// Register all passes in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Passes.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "dfg-mlir/Dialect/emitHLS/Transforms/Passes.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>

namespace mlir {

inline void registerAllDFGMLIRPasses()
{
    dfg::registerDFGMLIRConversionPasses();

    dfg::registerDFGMLIRDfgPasses();
    emitHLS::registerDFGMLIREmitHLSPasses();

    dfg::registerConvertToEmitHLSPipelines();
}

} // namespace mlir
