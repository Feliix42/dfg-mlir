//dfg-mlir/Target/GenerateMDCProject/GenerateMDCProjectEmitter.h
#pragma once
#include "mlir/IR/Operation.h" 
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace dfg {

LogicalResult generateMDCProject(Operation* op, raw_ostream& os);

} // namespace dfg
} // namespace mlir