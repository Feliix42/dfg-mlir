/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "dfg-mlir/Conversion/ArithIndexToVitis/ArithIndexToVitis.h"
#include "dfg-mlir/Conversion/DfgInsertOlympusWrapper/DfgInsertOlympusWrapper.h"
#include "dfg-mlir/Conversion/DfgToAsync/DfgToAsync.h"
#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"
#include "dfg-mlir/Conversion/DfgToLLVM/DfgToLLVM.h"
#include "dfg-mlir/Conversion/DfgToOlympus/DfgToOlympus.h"
#include "dfg-mlir/Conversion/DfgToVitis/DfgToVitis.h"
#include "dfg-mlir/Conversion/LinalgToDfg/LinalgToDfg.h"
#include "dfg-mlir/Conversion/MathToVitis/MathToVitis.h"
#include "dfg-mlir/Conversion/MemrefToVitis/MemrefToVitis.h"
#include "dfg-mlir/Conversion/ScfToVitis/ScfToVitis.h"

namespace mlir::dfg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::dfg
