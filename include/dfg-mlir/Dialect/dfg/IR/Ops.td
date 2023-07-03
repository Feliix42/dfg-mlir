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
            // CallableOpInterface,
            // FunctionOpInterface,
            IsolatedFromAbove,
            OpAsmOpInterface,
            NoTerminator
        ]> {
    let summary = "Defines a node in the Dataflow Graph";
    let description = [{
        The `operator` operation produces a dataflow node which is linked to other
        operators by hooking up their input and output ports.
        This is done using the `instantiate` function.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    // TODO(feliix42): Ensure that all arguments are of type `channel`!

    let arguments = (ins SymbolNameAttr: $sym_name,
                        TypeAttrOf<FunctionType>: $function_type,
                        DefaultValuedAttr<BoolAttr, "false">: $looped);

    // The FunctionOpInterface requires exactly one region attached to the operator
    let regions = (region AnyRegion: $body);

    // how can an option like this be *so* under-documented?!
    let skipDefaultBuilders = 1;

    let builders = [OpBuilder<(ins
        "StringRef": $sym_name, "FunctionType": $type)>];

    let extraClassDeclaration = [{
        bool isExternal();

        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        // ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        // ArrayRef<Type> getCallableResults() { return ArrayRef<Type>(); }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getInputTypes() { return getFunctionType().getInputs(); }

        /// Return the number of types of inputs
        // unsigned getNumInputTypes() { return getInputTypes().size(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getOutputTypes() { return getFunctionType().getResults(); }

        /// Return the number of types of outputs
        // unsigned getNumOutputTypes() { return getOutputTypes().size(); }

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

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;
}

// def Dfg_EndOp : Dfg_Op<"end", [Pure, HasParent<"OperatorOp">, Terminator]>
// {
//     // TODO: need more semantics

//     let builders = [OpBuilder<(ins), [{
//         build($_builder, $_state, std::nullopt);
//     }]>];

//     let assemblyFormat = [{ attr-dict }];
// }


def Dfg_ChannelOp : Dfg_Op<
        "channel",
        [
            // THIS ENFORCES OneResult ON THE OP!
            // ConstantLike,
            Pure,
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
        The results of this op have a strict order, the first one should be the
        input port of channel and the second is the output port.
        ```

        Examples:

        ```
        TODO
        %input, %output : !dfg.channel<i32>
        ```
    }];

    // TODO(feliix42): Verify that the encapsulated types in both results match the `encapsulatedType`
    // Thoughts(jibi): don't think we need one, we only take one type and then the elementType of
    //                 Intput/Output type would automatically be the same as encapulatedType
    let arguments = (ins TypeAttr:$encapsulatedType);
    let results = (outs
        Dfg_InputType:$in_chan,
        Dfg_OutputType:$out_chan
    );

    let hasCustomAssemblyFormat = 1;
}


def Dfg_InstantiateOp : Dfg_Op<
    "instantiate",
    [
        // Pure
        AttrSizedOperandSegments
    ]> {
    let summary = "Instantiates an operator with the given inputs, forming (part of) a data flow graph.";
    let description = [{
        The `instantiate` operation instantiates a data flow graph operator with
        a concrete set of inputs and outputs. This instanciation creates a node
        in the data flow graph. The data flow is derived from the arguments to
        the instanciation as well as the results and their uses/origin.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```

    }];

    let arguments = (ins SymbolNameAttr:$callee,
                         Variadic<Dfg_OutputType>:$inputs,
                         Variadic<Dfg_InputType>:$outputs);

    // let assemblyFormat = [{
    //     $callee `inputs` `(` $inputs `)` `outputs` `(` $outputs `)` attr-dict `:` functional-type($inputs, $outputs)
    // }];

    let hasCustomAssemblyFormat = 1;
}


def Dfg_KernelOp : Dfg_Op<
    "kernel",
    [
        AttrSizedOperandSegments
    ]> {
    let summary = "Instantiates a kernel with the given inputs, forming (part of) a data flow graph.";
    let description = [{
        The `kernel` operation instantiates a data flow graph operator with
        a concrete set of inputs and outputs. This instanciation creates a node
        in the data flow graph. The data flow is derived from the arguments to
        the instanciation as well as the results and their uses/origin.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```

    }];

    let arguments = (ins FlatSymbolRefAttr:$callee,
                         Variadic<Dfg_OutputType>:$inputs,
                         Variadic<Dfg_InputType>:$outputs);

    let hasCustomAssemblyFormat = 1;
}


def Dfg_PushOp : Dfg_Op<
    "push",
    [
        TypesMatchWith<"Channel type matches pushed type",
                       "chan", "inp",
                       "$_self.dyn_cast<InputType>().getElementType()">
        //                "InputType::get($_ctxt, $_self).getElementType()">
    ]
> {
    let summary = "Pushes a value into a channel input to send it to the next operator.";
    let description = [{
        The `push` operation transfers a value to another operator by using a
        channel supplied as argument. Since all channels are typed, the input
        type must match the channel type.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    let arguments = (ins AnyType: $inp, Dfg_InputType: $chan);

    let assemblyFormat = [{
        `(` $inp `)` $chan attr-dict `:` functional-type($inp, $chan)
    }];
}

def Dfg_PullOp : Dfg_Op<
    "pull",
    [
        TypesMatchWith<"Channel type matches pulled type",
                       "chan", "outp",
                       "$_self.dyn_cast<OutputType>().getElementType()">
                    //    "OutputType::get($_ctxt, $_self).getElementType()">
    ]
> {
    let summary = "Pulls a value from a channel.";
    let description = [{
        The `pull` operation retrieves a value from another operator by using a
        channel supplied as argument. Since all channels are typed, the output
        type must match the channel type.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    let arguments = (ins Dfg_OutputType: $chan);
    let results = (outs AnyType: $outp);

    let assemblyFormat = [{
        $chan attr-dict `:` functional-type($chan, $outp)
    }];
}

#endif
