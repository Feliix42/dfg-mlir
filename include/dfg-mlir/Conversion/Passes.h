/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "dfg-mlir/Conversion/DfgToAsync/DfgToAsync.h"
#include "dfg-mlir/Conversion/DfgToCirct/DfgToCirct.h"
#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"
#include "dfg-mlir/Conversion/DfgToLLVM/DfgToLLVM.h"
#include "dfg-mlir/Conversion/StdToCirct/StdToCirct.h"
#include "dfg-mlir/Conversion/DfgToOlympus/DfgToOlympus.h"
#include "dfg-mlir/Conversion/DfgInsertOlympusWrapper/DfgInsertOlympusWrapper.h"

namespace mlir::dfg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::dfg
