//===- MathOp.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS ops related to math lib.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_MATHOPS
#define EMITHLS_MATHOPS

class emitHLS_MathOpBase<string mnemonic, list<Trait> traits = []>
        : emitHLS_Op< "math." # mnemonic, traits> {}

def emitHLS_MathSinOp : emitHLS_MathOpBase<"sin", [Pure, AllTypesMatch<["value", "sin_value"]>]>
{
    let arguments = (ins AnyFloat:$value);
    let results = (outs AnyFloat:$sin_value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

def emitHLS_MathCosOp : emitHLS_MathOpBase<"cos", [Pure, AllTypesMatch<["value", "cos_value"]>]>
{
    let arguments = (ins AnyFloat:$value);
    let results = (outs AnyFloat:$cos_value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // EMITHLS_MATHOPS