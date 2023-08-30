/// Declaration of the Dfg lowering pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {

namespace comb {
class CombDialect;
} // namespace comb

namespace fsm {
class FSMDialect;
} // namespace fsm

namespace handshake {
class HandshakeDialect;
} // namespace handshake

namespace hw {
class HWDialect;
} // namespace hw

namespace sv {
class SVDialect;
} // namespace sv

} // namespace circt

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

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