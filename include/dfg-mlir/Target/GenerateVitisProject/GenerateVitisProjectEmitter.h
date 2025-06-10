/// Declaration of the emitHLS project emitter.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/StringRef.h>

namespace mlir {
namespace emitHLS {

/// Translates the given operation to a project folder, which contains cpp file
/// for HLS, Tcl scripts for running HLS and Vivado design, and a bash script to
/// run Tcl scripts automatically. The operation or operations in the region of
/// 'op' need almost all be in emitHLS dialect.
LogicalResult generateemitHLSProject(
    Operation* op,
    raw_ostream &os,
    StringRef outputDirOpt,
    StringRef formdc,
    StringRef targetDevice);

} // namespace emitHLS
} // namespace mlir
