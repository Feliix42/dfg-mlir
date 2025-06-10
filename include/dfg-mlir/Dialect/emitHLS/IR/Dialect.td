//===- Dialect.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_BASE
#define EMITHLS_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def emitHLS_Dialect : Dialect {
    let name = "emitHLS";
    let cppNamespace = "::mlir::emitHLS";

    let useDefaultTypePrinterParser = 1;
    // This section has been removed since Feb 24.
    // let useFoldAPI = kEmitFoldAdaptorFolder;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class emitHLS_Attr<string name, list<Trait> traits = []>
        : AttrDef<emitHLS_Dialect, name, traits>;
// Template for ops.
class emitHLS_Op<string mnemonic, list<Trait> traits = []>
        : Op<emitHLS_Dialect, mnemonic, traits>;
// Template for types.
class emitHLS_Type<string name, list<Trait> traits = []>
        : TypeDef<emitHLS_Dialect, name, traits>;

#endif // EMITHLS_BASE
