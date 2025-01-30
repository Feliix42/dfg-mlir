//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the vitis dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_OPS
#define VITIS_OPS

include "dfg-mlir/Dialect/vitis/IR/Types.td"
include "dfg-mlir/Dialect/vitis/IR/Dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

include "dfg-mlir/Dialect/vitis/IR/ArrayOps.td"
include "dfg-mlir/Dialect/vitis/IR/ArithOps.td"
include "dfg-mlir/Dialect/vitis/IR/LoopOps.td"
include "dfg-mlir/Dialect/vitis/IR/StreamOps.td"
include "dfg-mlir/Dialect/vitis/IR/PragmaOps.td"

def Vitis_ConstantOp : Vitis_Op<"constant",
        [
            ConstantLike,
            Pure,
            AllTypesMatch<["value", "result"]>
        ]>
{
    let summary = "Constant operation";
    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "attr-dict $value";
}

def Vitis_VariableOp : Vitis_Op<"variable", [Pure]>
{
    let summary = "define a variable as in C";
    let arguments = (ins Optional<AnyType>:$init);
    let results = (outs AnyType:$variable);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Vitis_UpdateOp : Vitis_Op<"update", [Pure, AllTypesMatch<["variable", "new_value"]>]>
{
    let summary = "update avariable with a new value";
    let arguments = (ins AnyType:$variable, AnyType: $new_value);
    let assemblyFormat = "$variable `,` $new_value  attr-dict `:` type($variable)";
}

def Vitis_TypedefOp : Vitis_Op<"typedef", []>
{
    let summary = "typedef";
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
    let assemblyFormat = "$sym_name `:` $type attr-dict";   
}

def Vitis_FuncOp : Vitis_Op<"func",
        [
            AutomaticAllocationScope,
            FunctionOpInterface,
            IsolatedFromAbove
        ]>
{
    let summary = "Function";
    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<DictArrayAttr>:$arg_attrs,
                         OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Vitis_ReturnOp : Vitis_Op<"return",
        [
            Pure,
            HasParent<"FuncOp">,
            Terminator
        ]>
{
    let summary = "return";
    // let arguments = (ins Optional<AnyType>:$operand);
    // let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";
    // let hasVerifier = 1;
    let assemblyFormat = "attr-dict";
}

def Vitis_IfBreakOp : Vitis_Op<"if_break",
        [
            Pure,
            NoTerminator
        ]>
{

    let summary = "if last signal is asserted, break to end exec";
    let arguments = (ins I1:$condition);
    let assemblyFormat = "$condition attr-dict";
}

def Vitis_ForOp : Vitis_Op<"for",
        [
            Pure,
            NoTerminator,
            AllTypesMatch<["lowerBound", "upperBound", "step"]>
        ]>
{
    let summary = "for-loop";
    let arguments = (ins Index:$lowerBound,
                         Index:$upperBound,
                         Index:$step);
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            function_ref<void(OpBuilder &, Location, Value)>;
        Value getInductionVar() { return getBody().getArgument(0); }
        void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
        void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
        void setStep(Value step) { getOperation()->setOperand(2, step); }
    }];

    let hasCustomAssemblyFormat = 1;
}

def Vitis_WhileOp : Vitis_Op<"while",
        [
            Pure,
            NoTerminator
        ]>
{
    let summary = "while-loop";
    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$condition $body attr-dict";
}

def Vitis_WhileTrueOp : Vitis_Op<"while_true",
        [
            Pure,
            NoTerminator
        ]>
{
    let summary = "while-true-loop";
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$body attr-dict";
}

#endif // VITIS_OPS
