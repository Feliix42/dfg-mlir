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
    let hasVerifier = 1;
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
                        //  OptionalAttr<StrAttr>:$sym_visibility,
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

#endif // VITIS_OPS
