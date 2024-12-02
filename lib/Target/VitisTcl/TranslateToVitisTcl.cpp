//===- TranslateToVitisTcl.cpp - Translating to Vitis Tcl -----------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Ops.h"
#include "dfg-mlir/Target/VitisTcl/VitisTclEmitter.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace vitis {

LogicalResult
translateToVitisTcl(Operation* op, raw_ostream &os, std::string &targetDevice)
{
    raw_indented_ostream ios(os);

    ios << "set TARGET_DEVICE \"" << targetDevice << "\"\n";
    ios << "set CLOCK_PERIOD 10\n";
    ios << "set PROJECT_DIR \"./hls_project\"\n";

    ios << "set cpp_files [glob -nocomplain \"./*.cpp\"]\n";
    ios << "if {[llength $cpp_files] == 0} {\n";
    ios.indent() << "puts \"ERROR: No .cpp file found in the directory!\"\n";
    ios << "exit 1\n";
    ios.unindent() << "}\n";
    ios << "set SOURCE_FILE [lindex $cpp_files 0]\n";
    ios << "puts \"INFO: Using source file: $SOURCE_FILE\"\n";

    ios << "open_project $PROJECT_DIR\n";
    ios << "add_files $SOURCE_FILE\n";

    llvm::StringSet<> functionNames;
    op->walk([&functionNames](FuncOp funcOp) {
        if (funcOp->getDialect()->getNamespace() == "vitis")
            functionNames.insert(funcOp.getSymName().str());
    });

    if (functionNames.empty()) {
        return emitError(
            op->getLoc(),
            "No Vitis functions found to translate.");
    }

    ios << "set FUNCTION_NAMES {";
    llvm::interleave(
        functionNames,
        ios,
        [&](const auto &funcName) { ios << funcName.first(); },
        " ");
    ios << "}\n";

    ios << "foreach func $FUNCTION_NAMES {\n";
    ios.indent();
    ios << "open_solution \"solution_$func\"\n";
    ios << "set_part $TARGET_DEVICE\n";
    ios << "create_clock -period $CLOCK_PERIOD -name default\n";
    ios << "set_top $func\n";
    ios << "csynth_design\n";
    ios << "export_design -rtl verilog";
    ios << "close_solution\n";
    ios.unindent();
    ios << "}\n";

    ios << "close_project\n";
    ios << "exit\n";

    return success();
}

} // namespace vitis
} // namespace mlir
