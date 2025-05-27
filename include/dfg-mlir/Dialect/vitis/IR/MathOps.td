//===- MathOp.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis ops related to math lib.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_MATHOPS
#define VITIS_MATHOPS

class Vitis_MathOpBase<string mnemonic, list<Trait> traits = []>
        : Vitis_Op< "math." # mnemonic, traits> {}

def Vitis_MathSinOp : Vitis_MathOpBase<"sin", [Pure, AllTypesMatch<["value", "sin_value"]>]>
{
    let arguments = (ins AnyFloat:$value);
    let results = (outs AnyFloat:$sin_value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

def Vitis_MathCosOp : Vitis_MathOpBase<"cos", [Pure, AllTypesMatch<["value", "cos_value"]>]>
{
    let arguments = (ins AnyFloat:$value);
    let results = (outs AnyFloat:$cos_value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // VITIS_MATHOPS