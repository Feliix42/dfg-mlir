/// Declaration of the Dfg lowering pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {

namespace handshake {
class HandshakeDialect;
} // namespace handshake

} // namespace circt

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace async {
class AsyncDialect;
} // namespace async

namespace dfg {
class DfgDialect;
} // namespace dfg

namespace func {
class FuncDialect;
} // namespace func

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir