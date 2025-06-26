#ifndef DFG_INSERT_SBOX_PASS_H
#define DFG_INSERT_SBOX_PASS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace dfg {

std::unique_ptr<Pass> createInsertSBoxPass();

} // namespace dfg
} // namespace mlir

#endif // DFG_MLIR_DIALECT_DFG_TRANSFORMS_ADDSBOX_H
