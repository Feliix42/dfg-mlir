/// BufferizableOpInterface implementation
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#ifndef MLIR_DIALECT_DFG_BUFFERIZABLEOPINTERFACEIMPL_H
#    define MLIR_DIALECT_DFG_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {

class DialectRegistry;

namespace dfg {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace dfg
} // namespace mlir

#endif // MLIR_DIALECT_DFG_BUFFERIZABLEOPINTERFACEIMPL_H
