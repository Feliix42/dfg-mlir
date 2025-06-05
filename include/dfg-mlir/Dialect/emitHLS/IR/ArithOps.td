//===- ArithOps.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS ops related to arithmetic.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_ARITHOPS
#define EMITHLS_ARITHOPS

class emitHLS_ArithOpBase<string mnemonic, list<Trait> traits = []>
        : emitHLS_Op< "arith." # mnemonic, traits> {}

class emitHLS_BinaryArithOp<string mnemonic, list<Trait> traits = []>
        : emitHLS_ArithOpBase<mnemonic, traits # [Pure, SameOperandsAndResultType]>
{
    let arguments = (ins AnyType:$lhs, AnyType: $rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def emitHLS_ArithAddOp : emitHLS_BinaryArithOp<"add", []> {}
def emitHLS_ArithSubOp : emitHLS_BinaryArithOp<"sub", []> {}
def emitHLS_ArithMulOp : emitHLS_BinaryArithOp<"mul", []> {}
def emitHLS_ArithDivOp : emitHLS_BinaryArithOp<"div", []> {}
def emitHLS_ArithRemOp : emitHLS_BinaryArithOp<"rem", []> {}
def emitHLS_ArithAndOp : emitHLS_BinaryArithOp<"and", []> {}
def emitHLS_ArithOrOp : emitHLS_BinaryArithOp<"or", []> {}

def emitHLS_ArithCastOp : emitHLS_ArithOpBase<"cast", [Pure, SameOperandsAndResultShape]>
{
    let arguments = (ins AnyType:$from);
    let results = (outs AnyType:$to);
    let assemblyFormat = "$from attr-dict `:` type($from) `to` type($to)";
}

def emitHLS_ArithCmpOp
    : emitHLS_ArithOpBase<"cmp",
        [Pure,
         SameTypeOperands,
         TypesMatchWith<"result type has i1 element type and same shape as operands", 
                        "lhs", "result", "::getI1SameShape($_self)">]>
{
    let arguments = (ins emitHLS_CmpPredicateAttr:$predicate,
                         AnyType:$lhs,
                         AnyType:$rhs);
    let results = (outs BoolLike:$result);
    let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def emitHLS_ArithSelectOp
    : emitHLS_ArithOpBase<"select",
        [Pure,
         AllTypesMatch<["true_value", "false_value", "result"]>]>
{
    let arguments = (ins BoolLike:$condition,
                         AnyType:$true_value,
                         AnyType:$false_value);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

#endif // EMITHLS_ARITHOPS