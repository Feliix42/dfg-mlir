//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the emitHLS dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_OPS
#define EMITHLS_OPS

include "dfg-mlir/Dialect/emitHLS/Enums.td"
include "dfg-mlir/Dialect/emitHLS/IR/Types.td"
include "dfg-mlir/Dialect/emitHLS/IR/Dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

include "dfg-mlir/Dialect/emitHLS/IR/ArrayOps.td"
include "dfg-mlir/Dialect/emitHLS/IR/ArithOps.td"
include "dfg-mlir/Dialect/emitHLS/IR/MathOps.td"
include "dfg-mlir/Dialect/emitHLS/IR/PragmaOps.td"
include "dfg-mlir/Dialect/emitHLS/IR/StreamOps.td"

def emitHLS_IncludeOp : emitHLS_Op<"include", []>
{
    let summary = "C/C++ include statement";
    let arguments = (ins
        Arg<StrAttr, "source file to include">:$include
    );
    let hasCustomAssemblyFormat = 1;
}

def emitHLS_VariableOp : emitHLS_Op<"variable",
        [
            DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
        ]>
{
    let summary = "define a variable as in C";
    let arguments = (ins OptionalAttr<AnyAttr>:$init, UnitAttr:$is_const);
    let results = (outs AnyType:$variable);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Type":$type,
                       CArg<"Attribute", "{}">:$init,
                       CArg<"bool", "false">:$is_const)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        bool isVariableConst() {
            return getOperation()->hasAttr("is_const");
        }
    }];
}

def emitHLS_UpdateOp : emitHLS_Op<"update", [AllTypesMatch<["variable", "new_value"]>,
                                         MemoryEffectsOpInterface]>
{
    let summary = "update avariable with a new value";
    let arguments = (ins AnyType:$variable, AnyType: $new_value);

    let assemblyFormat = "$variable `,` $new_value  attr-dict `:` type($variable)";

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), &getVariableMutable(),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

def emitHLS_FuncOp : emitHLS_Op<"func",
        [
            AutomaticAllocationScope,
            FunctionOpInterface,
            IsolatedFromAbove,
            NoTerminator
        ]>
{
    let summary = "Function";
    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<DictArrayAttr>:$arg_attrs,
                         OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

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


        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//
        ArrayRef<Type> getFuncElementTypes();
    }];
}

def emitHLS_CallOp : emitHLS_Op<"call",
        [
            CallOpInterface,
        ]>
{
    let summary = "funtion call";

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$operands,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Variadic<AnyType>);

    let builders = [
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>
    ];

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallOpInterface Methods
        //===------------------------------------------------------------------===//
        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
            return getOperandsMutable();
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
        }
    }];
}

def emitHLS_ForOp : emitHLS_Op<"for",
        [
            RecursiveMemoryEffects,
            NoTerminator,
            DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
        ]>
{
    let summary = "for-loop";
    let arguments = (ins IndexAttr:$lowerBound,
                         IndexAttr:$upperBound,
                         IndexAttr:$step);
    let regions = (region SizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "int64_t":$lowerBound, "int64_t":$upperBound, CArg<"int64_t", "1">:$step)>,
        OpBuilder<(ins "Attribute":$lowerBound, "Attribute":$upperBound, "Attribute":$step)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        Value getInductionVar() { return getBody().getArgument(0); }
    }];
}

def emitHLS_WhileOp : emitHLS_Op<"while",
        [
            RecursiveMemoryEffects,
            NoTerminator
        ]>
{
    let summary = "while-loop";
    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$condition $body attr-dict";
}

#endif // EMITHLS_OPS
