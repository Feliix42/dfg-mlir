//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_OPS
#define DFG_OPS

include "dfg-mlir/Dialect/dfg/IR/Types.td"
include "dfg-mlir/Dialect/dfg/IR/Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"


def Dfg_OperatorOp : Dfg_Op<
        "operator",
        [
            AffineScope,
            AutomaticAllocationScope,
            CallableOpInterface,
            FunctionOpInterface,
            IsolatedFromAbove,
            OpAsmOpInterface
        ]> {
    let summary = "Defines a node in the Dataflow Graph";
    let description = [{
        The `operator` operation produces a dataflow node which is linked to other
        operators by hooking up their input and output ports.
        This is done using the `instanciate` function.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    // TODO: CONTINUE HERE -> make this similar to how `func.func` works, especially regarding the customAssemblyParser!
    // NOTE(feliix42): One could add visibility in the form of `OptionalAttr<StrArray>:$sym_visibility` here
    let arguments = (ins SymbolNameAttr: $sym_name,
                         TypeAttrOf<FunctionType>: $function_type,
                         OptionalAttr<DictArrayAttr>: $arg_attrs,
                         OptionalAttr<DictArrayAttr>: $res_attrs);

    // The FunctionOpInterface requires exactly one region attached to the operator
    let regions = (region AnyRegion: $body);

    // TODO(feliix42):
    // let builders = [];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "dfg"; }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    // CONTINUE: Implement custom assembly printer/parser
    let hasCustomAssemblyFormat = 1;
}


def Dfg_ChannelOp : Dfg_Op<
        "channel",
        [
            // THIS ENFORCES OneResult ON THE OP!
            // ConstantLike,
            Pure
            // OpAsmOpInterface
        ]> {
    let summary = "Defines a channel with one input and one output port";
    let description = [{
        The `channel` operation produces a typed channel (i.e., an edge in the
        dataflow graph) that links two operators (i.e., nodes) in the dataflow
        graph.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    // NOTE(feliix42): I want to _materialize_ the type attribute here -> derived attribute?
    //                 Alternatively have it as attribute and derive from custom assembly
    let arguments = (ins TypeAttr:$encapsulatedType);
    let results = (outs
        Dfg_InputType:$in_chan,
        Dfg_OutputType:$out_chan
    );

    // TODO(feliix42): Switch to custom assembly format + parser
    let assemblyFormat = [{ attr-dict `:` $encapsulatedType `(` type($in_chan) `->` type($out_chan) `)` }];

    // TODO(feliix42): Ensure type matching on both results and the attribute
}

#endif
