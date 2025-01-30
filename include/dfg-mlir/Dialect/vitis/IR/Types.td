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

def Vitis_SizeTType : Vitis_Type<"SizeT", []> {
    let mnemonic = "size_t";
    let summary = "size_t in C/C++";
}

def Vitis_ArrayType : Vitis_Type<"Array", []> {
    let mnemonic = "array";
    let summary = "C/C++ arrays with fixed size and data type";
    let parameters = (ins "int64_t":$size,
                          "Type":$elem_type);
    let assemblyFormat = "`<`$size`x`$elem_type`>`";
}

def Vitis_APFixedType : Vitis_Type<"APFixed", []> {
    let mnemonic = "ap_fixed";
    let summary = "ap_fixed<DATAWIDTH, INT_WIDTH>";
    let parameters = (ins "int64_t":$datawidth,
                          "int64_t":$int_width);
    let assemblyFormat = "`<` $datawidth `,` $int_width";
}

def Vitis_APFixedUType : Vitis_Type<"APFixedU", []> {
    let mnemonic = "ap_ufixed";
    let summary = "ap_fixed<DATAWIDTH, INT_WIDTH>";
    let parameters = (ins "int64_t":$datawidth,
                          "int64_t":$int_width);
    let assemblyFormat = "`<` $datawidth `,` $int_width";
}

def Vitis_APAxisType : Vitis_Type<"APAxis", []> {
    let mnemonic = "hls_axis";
    let summary = "hls::axis<DATATYPE, keep, user, dest, AXIS_ENABLE_LAST>";
    let parameters = (ins "Type":$elem_type,
                          "int64_t":$keep,
                          "int64_t":$user,
                          "int64_t":$dest,
                          "bool":$tlast);
    let assemblyFormat = "`<` $elem_type `,` $keep `,` $user `,` $dest `,` $tlast `>`";
}

def Vitis_StreamType : Vitis_Type<"Stream", []> {
    let mnemonic = "stream";
    let summary = "hls::stream<AXIS>";
    let parameters = (ins "Type":$stream_type);
    let assemblyFormat = "`<` $stream_type `>`";
    let genVerifyDecl = 1;
}

#endif
