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

def Vitis_ArrayType : Vitis_Type<"Array",
        [
            ShapedTypeInterface
        ]>
{
    let mnemonic = "array";
    let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                          "Type":$elementType);

    let builders = [
        TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                            "Type":$elementType), [{
            return $_get(elementType.getContext(), shape, elementType);
        }]>
    ];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // ShapedTypeInterface Methods
        //===------------------------------------------------------------------===//

        // For HLS, an array should always have rank
        bool hasRank() const { return true; }

        ArrayType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                              Type elementType) const;
    }];
}

def Vitis_PointerType : Vitis_Type<"Pointer", []> {
    let mnemonic = "ptr";
    let summary = "type *";
    let parameters = (ins "Type":$pointer_type);
    let assemblyFormat = "`<` qualified($pointer_type) `>`";
    let genVerifyDecl = 1;
}

def Vitis_StreamType : Vitis_Type<"Stream", []> {
    let mnemonic = "stream";
    let summary = "hls::stream<type>";
    let parameters = (ins "Type":$stream_type);
    let assemblyFormat = "`<` qualified($stream_type) `>`";
    let genVerifyDecl = 1;
}

#endif
