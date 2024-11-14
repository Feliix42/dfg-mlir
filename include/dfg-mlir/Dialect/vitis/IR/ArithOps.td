//===- ArithOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis ops related to arithmetic.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_ARITHOPS
#define VITIS_ARITHOPS

class Vitis_ArithOpBase<string mnemonic, list<Trait> traits = []>
        : Vitis_Op< "arith." # mnemonic, traits> {}

class Vitis_BinaryArithOp<string mnemonic, list<Trait> traits = []>
        : Vitis_ArithOpBase<mnemonic, traits # [Pure, SameOperandsAndResultType]>
{
    let arguments = (ins AnyType:$lhs, AnyType: $rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Vitis_ArithAddOp : Vitis_BinaryArithOp<"add", []> {}
def Vitis_ArithSubOp : Vitis_BinaryArithOp<"sub", []> {}
def Vitis_ArithMulOp : Vitis_BinaryArithOp<"mul", []> {}
def Vitis_ArithAndOp : Vitis_BinaryArithOp<"and", []> {}
def Vitis_ArithOrOp : Vitis_BinaryArithOp<"or", []> {}

#endif // VITIS_ARITHOPS