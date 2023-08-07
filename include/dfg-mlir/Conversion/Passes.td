//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_CONVERSION_PASSES
#define DFG_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDfgToAsync : Pass<"convert-dfg-to-async"> {
    let summary = "Convert DFG Dialect to Async";

    let constructor = "mlir::createConvertDfgToAsyncPass()";

    let dependentDialects = [
        "async::AsyncDialect",
        "dfg::DfgDialect"
    ];
}

def ConvertDfgToCirct : Pass<"convert-dfg-to-circt"> {
    let summary = "Convert DFG dialect to CIRCT";

    let constructor = "mlir::createConvertDfgToCirctPass()";

    let dependentDialects = [
        "circt::comb::CombDialect",
        "circt::fsm::FSMDialect",
        "circt::hw::HWDialect",
        "circt::sv::SVDialect",

        "dfg::DfgDialect"
    ];
}

def ConvertStdToCirct : Pass<"convert-std-to-circt"> {
    let summary = "Convert Std dialect to CIRCT";

    let constructor = "mlir::createConvertStdToCirctPass()";

    let dependentDialects = [
        "arith::ArithDialect",
        "func::FuncDialect",

        "dfg::DfgDialect"
    ];
}

#endif // DFG_MLIR_CONVERSION_PASSES