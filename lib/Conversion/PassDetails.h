/// Declaration of the Dfg lowering pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace async {
class AsyncDialect;
} // namespace async

namespace comb {
class CombDialect;
} // namespace comb

namespace dfg {
class DfgDialect;
} // namespace dfg

namespace hw {
class HWDialect;
} // namespace hw

namespace SV {
class SVDialect;
} // namespace SV

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir