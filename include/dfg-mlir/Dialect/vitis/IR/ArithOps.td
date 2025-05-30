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
def Vitis_ArithDivOp : Vitis_BinaryArithOp<"div", []> {}
def Vitis_ArithRemOp : Vitis_BinaryArithOp<"rem", []> {}
def Vitis_ArithAndOp : Vitis_BinaryArithOp<"and", []> {}
def Vitis_ArithOrOp : Vitis_BinaryArithOp<"or", []> {}

def Vitis_ArithCastOp : Vitis_ArithOpBase<"cast", [Pure, SameOperandsAndResultShape]>
{
    let arguments = (ins AnyType:$from);
    let results = (outs AnyType:$to);
    let assemblyFormat = "$from attr-dict `:` type($from) `to` type($to)";
}

def Vitis_ArithCmpOp
    : Vitis_ArithOpBase<"cmp",
        [Pure,
         SameTypeOperands,
         TypesMatchWith<"result type has i1 element type and same shape as operands", 
                        "lhs", "result", "::getI1SameShape($_self)">]>
{
    let arguments = (ins Vitis_CmpPredicateAttr:$predicate,
                         AnyType:$lhs,
                         AnyType:$rhs);
    let results = (outs BoolLike:$result);
    let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Vitis_ArithSelectOp
    : Vitis_ArithOpBase<"select",
        [Pure,
         AllTypesMatch<["true_value", "false_value", "result"]>]>
{
    let arguments = (ins BoolLike:$condition,
                         AnyType:$true_value,
                         AnyType:$false_value);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

#endif // VITIS_ARITHOPS