//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_CONVERSION_PASSES
#define DFG_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithIndexToEmitHLS : Pass<"convert-arith-index-to-emitHLS"> {
    let summary = "Convert Arith and Math dialect to emitHLS";

    let constructor = "mlir::createConvertArithIndexToEmitHLSPass()";

    let dependentDialects = [
        "arith::ArithDialect",
        "index::IndexDialect",
        "emitHLS::emitHLSDialect"
    ];
}

def ConvertDfgToAsync : Pass<"convert-dfg-to-async"> {
    let summary = "Convert DFG Dialect to Async";

    let constructor = "mlir::createConvertDfgToAsyncPass()";

    let dependentDialects = [
        "async::AsyncDialect",
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

def ConvertDfgToEmitHLS : Pass<"convert-dfg-to-emitHLS"> {
    let summary = "Convert DFG dialect graph nodes to emitHLS dialect.";

    let constructor = "mlir::createConvertDfgToEmitHLSPass()";

    let dependentDialects = [
        "dfg::DfgDialect",
        "emitHLS::emitHLSDialect",

        "arith::ArithDialect"
    ];
}

def ConvertFuncToDfg : Pass<"convert-func-to-dfg"> {
    let summary = "Convert functions to dfg representations.";

    let constructor = "mlir::createConvertFuncToDfgPass()";

    let dependentDialects = [
        "dfg::DfgDialect",
        "func::FuncDialect"
    ];
}

def ConvertLinalgToDfg : Pass<"convert-linalg-to-dfg"> {
    let summary = "Convert linalg programs to dfg representation.";

    let constructor = "mlir::createConvertLinalgToDfgPass()";

    let dependentDialects = [
        "dfg::DfgDialect",

        "arith::ArithDialect",
        "linalg::LinalgDialect"
    ];
}

def ConvertMathToEmitHLS : Pass<"convert-math-to-emitHLS"> {
    let summary = "Convert math operations to emitHLS math lib call.";

    let constructor = "mlir::createConvertMathToEmitHLSPass()";

    let dependentDialects = [
        "emitHLS::emitHLSDialect",
        "math::MathDialect"
    ];
}

def ConvertMemrefToEmitHLS : Pass<"convert-memref-to-emitHLS"> {
    let summary = "Convert memref operations to emitHLS array operations.";

    let constructor = "mlir::createConvertMemrefToEmitHLSPass()";

    let dependentDialects = [
        "emitHLS::emitHLSDialect",
        "memref::MemRefDialect"
    ];
}

def ConvertScfToEmitHLS : Pass<"convert-scf-to-emitHLS"> {
    let summary = "Convert scf operations to emitHLS array operations.";

    let constructor = "mlir::createConvertScfToEmitHLSPass()";

    let dependentDialects = [
        "emitHLS::emitHLSDialect",
        "scf::SCFDialect"
    ];
}

#endif // DFG_MLIR_CONVERSION_PASSES
