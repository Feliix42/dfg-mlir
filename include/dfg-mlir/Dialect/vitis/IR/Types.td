//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_TYPES
#define VITIS_TYPES

include "dfg-mlir/Dialect/vitis/IR/Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def Vitis_APFixedType : Vitis_Type<"APFixed", []> {
    let mnemonic = "ap_fixed";
    let summary = "ap_fixed<DATAWIDTH, INT_WIDTH>";
    let parameters = (ins "int64_t":$datawidth,
                          "int64_t":$int_width);
    let assemblyFormat = "`<` $datawidth `,` $int_width `>`";
}

def Vitis_APFixedUType : Vitis_Type<"APFixedU", []> {
    let mnemonic = "ap_ufixed";
    let summary = "ap_fixed<DATAWIDTH, INT_WIDTH>";
    let parameters = (ins "int64_t":$datawidth,
                          "int64_t":$int_width);
    let assemblyFormat = "`<` $datawidth `,` $int_width `>`";
}

def Vitis_ArrayType : Vitis_Type<"Array", []> {
    let mnemonic = "array";
    let summary = "C/C++ arrays with fixed size and data type";
    let parameters = (ins "int64_t":$size,
                          "Type":$elem_type);
    let assemblyFormat = "`<`$size `x` $elem_type`>`";
}

def Vitis_PointerType : Vitis_Type<"Pointer", []> {
    let mnemonic = "ptr";
    let summary = "type *";
    let parameters = (ins "Type":$pointer_type);
    let assemblyFormat = "`<` $pointer_type `>`";
    let genVerifyDecl = 1;
}

def Vitis_StreamType : Vitis_Type<"Stream", []> {
    let mnemonic = "stream";
    let summary = "hls::stream<type>";
    let parameters = (ins "Type":$stream_type);
    let assemblyFormat = "`<` $stream_type `>`";
    let genVerifyDecl = 1;
}

#endif
