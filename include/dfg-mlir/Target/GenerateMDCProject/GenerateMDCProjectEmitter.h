//dfg-mlir/Target/GenerateMDCProject/GenerateMDCProjectEmitter.h
#pragma once
#include "mlir/IR/Operation.h" 
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include <llvm/Support/Path.h>
#include <filesystem>

namespace mlir {
namespace dfg {

LogicalResult generateMDCProject(Operation* op, const std::string& outputDir);

} // namespace dfg
} // namespace mlir