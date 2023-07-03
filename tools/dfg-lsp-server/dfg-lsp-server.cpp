/// Main entry point for the dfg-mlir MLIR language server.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "circt/InitAllDialects.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

static int asMainReturnCode(LogicalResult r)
{
    return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<dfg::DfgDialect>();
    registry.insert<circt::handshake::HandshakeDialect>();
    registry.insert<circt::esi::ESIDialect>();

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
