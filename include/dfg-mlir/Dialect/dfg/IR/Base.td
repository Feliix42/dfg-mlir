//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the DFG dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_BASE
#define DFG_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Dfg_Dialect : Dialect {
    let name = "dfg";
    let cppNamespace = "::mlir::dfg";

    let useDefaultAttributePrinterParser = 1;
    let useFoldAPI = kEmitFoldAdaptorFolder;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class Dfg_Attr<string name, list<Trait> traits = []>
        : AttrDef<Dfg_Dialect, name, traits>;
// Template for ops.
class Dfg_Op<string mnemonic, list<Trait> traits = []>
        : Op<Dfg_Dialect, mnemonic, traits>;
// Template for types.
class Dfg_Type<string name, list<Trait> traits = []>
        : TypeDef<Dfg_Dialect, name, traits>;

#endif
