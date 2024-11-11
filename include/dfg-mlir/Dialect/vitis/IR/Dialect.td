//===- Dialect.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Vitis dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_BASE
#define VITIS_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Vitis_Dialect : Dialect {
    let name = "vitis";
    let cppNamespace = "::mlir::vitis";

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
class Vitis_Attr<string name, list<Trait> traits = []>
        : AttrDef<Vitis_Dialect, name, traits>;
// Template for ops.
class Vitis_Op<string mnemonic, list<Trait> traits = []>
        : Op<Vitis_Dialect, mnemonic, traits>;
// Template for types.
class Vitis_Type<string name, list<Trait> traits = []>
        : TypeDef<Vitis_Dialect, name, traits>;

#endif // VITIS_BASE
