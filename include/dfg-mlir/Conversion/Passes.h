/// Declares the conversion passes.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "dfg-mlir/Conversion/DfgToAsync/DfgToAsync.h"
#include "dfg-mlir/Conversion/DfgToCirct/DfgToCirct.h"

namespace mlir::dfg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::dfg