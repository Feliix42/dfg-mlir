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

def Vitis_APUIntType : Vitis_Type<"APUInt", []> {
    let mnemonic = "ap_uint";
    let summary = "ap_uint<DATAWIDTH>";
    let parameters = (ins "int64_t":$datawidth);
    let assemblyFormat = "`<` $datawidth `>`";
}

def Vitis_APSIntType : Vitis_Type<"APSInt", []> {
    let mnemonic = "ap_sint";
    let summary = "ap_int<DATAWIDTH>";
    let parameters = (ins "int64_t":$datawidth);
    let assemblyFormat = "`<` $datawidth `>`";
}

def Vitis_APAxiUType : Vitis_Type<"APAxiU", []> {
    let mnemonic = "ap_axiu";
    let summary = "ap_axiu<DATAWIDTH, keep, user, dest, AXIS_ENABLE_LAST>";
    let parameters = (ins "int64_t":$datawidth,
                          "int64_t":$keep,
                          "int64_t":$user,
                          "int64_t":$dest,
                          "bool":$tlast);
    let assemblyFormat = "`<` $datawidth `,` $keep `,` $user `,` $dest `,` $tlast `>`";
}

def Vitis_APAxiSType : Vitis_Type<"APAxiS", []> {
    let mnemonic = "ap_axis";
    let summary = "ap_axis<DATAWIDTH, keep, user, dest, AXIS_ENABLE_LAST>";
    let parameters = (ins "int64_t":$datawidth,
                          "int64_t":$keep,
                          "int64_t":$user,
                          "int64_t":$dest,
                          "bool":$tlast);
    let assemblyFormat = "`<` $datawidth `,` $keep `,` $user `,` $dest `,` $tlast `>`";
}

def Vitis_StreamType : Vitis_Type<"Stream", []> {
    let mnemonic = "stream";
    let summary = "hls::stream<AXIS>";
    let parameters = (ins "Type":$stream_type);
    let assemblyFormat = "`<` $stream_type `>`";
    let genVerifyDecl = 1;
}

#endif
