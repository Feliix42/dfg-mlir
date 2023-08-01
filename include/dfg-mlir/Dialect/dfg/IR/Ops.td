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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"


def Dfg_OperatorOp : Dfg_Op<
        "operator",
        [
            AutomaticAllocationScope,
            // AttrSizedOperandSegments,
            IsolatedFromAbove,
            HasParent<"ModuleOp">,
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

    let arguments = (ins SymbolNameAttr: $sym_name,
                         TypeAttrOf<FunctionType>: $function_type);
                        //  TypeArrayAttr:$inputs,
                        //  TypeArrayAttr:$outputs);

    // The FunctionOpInterface requires zero or one region attached to the operator
    let regions = (region AnyRegion: $body);

    let extraClassDeclaration = [{
        /// Returns whether the operator is externally defined
        bool isExternal();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;
}

// TODO:
// , RegionBranchOpInterface]>
def Dfg_LoopOp : Dfg_Op<"loop", [HasParent<"OperatorOp">, NoTerminator]> 
{
    let summary = "Defines a list of input/output ports maybe closed";

    let description = [{
        The `loop` operation defines a list of input or output ports
        that will be closed later. The close behavior will terminate
        the operator and populate to other operators so that the whole
        dataflow graph will be shut down.

        Example:
        ```
        dfg.operator @add inputs(%a:!dfg.output<type>) outputs(%b:!dfg.input<type>)
        {
            dfg.loop (%a: !dfg.output<type>, %b: !dfg.input<type>) {
                ops ...
            }
        }
        ```
    }];

    // TODO: Constraint to any channel types
    let arguments = (ins Variadic<AnyType>:$channels);

    let regions = (region MinSizedRegion<1>: $body);

    let skipDefaultBuilders = 1;
    let builders = [OpBuilder<(ins "ValueRange":$channels)>];

    let hasCustomAssemblyFormat = 1;
}


def Dfg_ChannelOp : Dfg_Op<
        "channel",
        [
            Pure
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
        %input, %output : !dfg.channel<ui32, 4>
        ```
    }];

    let arguments = (ins
        TypeAttr:$encapsulatedType,
        OptionalAttr<I32Attr>:$bufferSize
    );
    let results = (outs
        Dfg_InputType:$in_chan,
        Dfg_OutputType:$out_chan
    );

    let hasCustomAssemblyFormat = 1;
}


def Dfg_InstantiateOp : Dfg_Op<
    "instantiate",
    [
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

    let arguments = (ins SymbolRefAttr:$callee,
                         Variadic<Dfg_OutputType>:$inputs,
                         Variadic<Dfg_InputType>:$outputs,
                         DefaultValuedAttr<BoolAttr, "false">:$offloaded);

    // let assemblyFormat = [{
    //     $callee `inputs` `(` $inputs `)` `outputs` `(` $outputs `)` attr-dict `:` functional-type($inputs, $outputs)
    // }];

    let hasCustomAssemblyFormat = 1;
}


def Dfg_PushOp : Dfg_Op<
    "push",
    [
        TypesMatchWith<"Channel type matches pushed type",
                       "chan", "inp",
                       "$_self.dyn_cast<InputType>().getElementType()">
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

    // let assemblyFormat = [{
    //     `(` $inp `)` $chan attr-dict `:` functional-type($inp, $chan)
    // }];

    let hasCustomAssemblyFormat = 1;
}

def Dfg_PullOp : Dfg_Op<
    "pull",
    [
        TypesMatchWith<"Channel type matches pulled type",
                       "chan", "outp",
                       "$_self.dyn_cast<OutputType>().getElementType()">
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

    // let assemblyFormat = [{
    //     $chan attr-dict `:` functional-type($chan, $outp)
    // }];

    let hasCustomAssemblyFormat = 1;
}

#endif
