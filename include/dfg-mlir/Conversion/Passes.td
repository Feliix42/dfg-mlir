//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_CONVERSION_PASSES
#define DFG_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithIndexToVitis : Pass<"convert-arith-index-to-vitis"> {
    let summary = "Convert Arith and Math dialect to vitis";

    let constructor = "mlir::createConvertArithIndexToVitisPass()";

    let dependentDialects = [
        "arith::ArithDialect",
        "index::IndexDialect",
        "vitis::VitisDialect"
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

def ConvertDfgToVitis : Pass<"convert-dfg-to-vitis"> {
    let summary = "Convert DFG dialect graph nodes to vitis dialect.";

    let constructor = "mlir::createConvertDfgToVitisPass()";

    let dependentDialects = [
        "dfg::DfgDialect",
        "vitis::VitisDialect",

        "arith::ArithDialect"
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

def ConvertMathToVitis : Pass<"convert-math-to-vitis"> {
    let summary = "Convert math operations to vitis math lib call.";

    let constructor = "mlir::createConvertMathToVitisPass()";

    let dependentDialects = [
        "vitis::VitisDialect",
        "math::MathDialect"
    ];
}

def ConvertMemrefToVitis : Pass<"convert-memref-to-vitis"> {
    let summary = "Convert memref operations to vitis array operations.";

    let constructor = "mlir::createConvertMemrefToVitisPass()";

    let dependentDialects = [
        "vitis::VitisDialect",
        "memref::MemRefDialect"
    ];
}

def ConvertScfToVitis : Pass<"convert-scf-to-vitis"> {
    let summary = "Convert scf operations to vitis array operations.";

    let constructor = "mlir::createConvertScfToVitisPass()";

    let dependentDialects = [
        "vitis::VitisDialect",
        "scf::SCFDialect"
    ];
}

#endif // DFG_MLIR_CONVERSION_PASSES
