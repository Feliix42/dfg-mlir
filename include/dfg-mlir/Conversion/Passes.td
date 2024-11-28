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

def ConvertDfgToHW : Pass<"convert-dfg-to-hw"> {
    let summary = "Convert DFG dialect to CIRCT HW";

    let constructor = "mlir::createConvertDfgToHWPass()";

    let dependentDialects = [
        "circt::handshake::HandshakeDialect",
        "circt::comb::CombDialect",
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
        "scf::SCFDialect",

        "circt::comb::CombDialect",
        "circt::handshake::HandshakeDialect",
        "circt::hw::HWDialect",
        "circt::sv::SVDialect",

        "dfg::DfgDialect"
    ];
}

def ConvertDfgToFunc : Pass<"convert-dfg-nodes-to-func"> {
    let summary = "Convert DFG dialect graph nodes to Func declarations and invocations";

    let constructor = "mlir::createConvertDfgToFuncPass()";

    let dependentDialects = [
        "func::FuncDialect",
        "cf::ControlFlowDialect",
        "omp::OpenMPDialect",
        "arith::ArithDialect",

        "dfg::DfgDialect"
    ];
}

def ConvertDfgToLLVM : Pass<"convert-dfg-edges-to-llvm"> {
    let summary = "Convert DFG dialect graph edges to channels implemented in LLVM.";

    let constructor = "mlir::createConvertDfgToLLVMPass()";

    let dependentDialects = [
        "cf::ControlFlowDialect",
        "func::FuncDialect",
        "LLVM::LLVMDialect",

        "dfg::DfgDialect"
    ];
}

def ConvertDfgToOlympus : Pass<"convert-dfg-to-olympus"> {
    let summary  = "Convert offloaded DFG nodes to olympus-mlir.";

    let constructor = "mlir::createConvertDfgToOlympusPass()";

    let  dependentDialects = [
        "func::FuncDialect",
        "dfg::DfgDialect"
    ];
}

def ConvertDfgInsertOlympusWrapper : Pass<"insert-olympus-wrappers"> {
    let summary  = "Convert offloaded DFG nodes to olympus wrappers.";

    let constructor = "mlir::createConvertDfgInsertOlympusWrapperPass()";

    let  dependentDialects = [
        "func::FuncDialect",
        "dfg::DfgDialect",
        "LLVM::LLVMDialect"
    ];
}

def ConvertDfgToVitis : Pass<"convert-dfg-to-vitis"> {
    let summary = "Convert DFG dialect graph nodes to vitis dialect.";

    let constructor = "mlir::createConvertDfgToLLVMPass()";

    let dependentDialects = [
        "dfg::DfgDialect",

        "arith::ArithDialect"
    ];
}

#endif // DFG_MLIR_CONVERSION_PASSES
