//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_OPS
#define DFG_OPS

include "dfg-mlir/Dialect/dfg/Enums.td"
include "dfg-mlir/Dialect/dfg/IR/Types.td"
include "dfg-mlir/Dialect/dfg/Interfaces/Interfaces.td"
include "dfg-mlir/Dialect/dfg/IR/Dialect.td"
include "dfg-mlir/Dialect/dfg/IR/HWOps.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def Dfg_RegionOp : Dfg_Op<
        "region",
        [
            Dfg_DataflowGraphOp,

            IsolatedFromAbove,
            NoTerminator
        ]>
{
    let summary = "Defines a region in a graph.";

    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type);
    let regions = (region SizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$function_type)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // DataflowGraphOp
        //===--------------------------------------------------------------===//

        void foo() { return; }
    }];
}

def Dfg_EmbedOp : Dfg_Op<
        "embed",
        [
            AttrSizedOperandSegments//,
            // HasParent<"RegionOp">
        ]>
{
    let summary = "Embed a region at current location.";

    let arguments = (ins SymbolRefAttr:$callee,
                         Variadic<Dfg_OutputType>:$inputs,
                         Variadic<Dfg_InputType>:$outputs);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_ConnectInputOp : Dfg_Op<
        "connect.input",
        []>
{
    let summary = "Connect region input ports with channels.";

    let arguments = (ins Dfg_OutputType:$regionPort,
                         Dfg_InputType:$channelPort);

    let hasCustomAssemblyFormat = 1;
}

def Dfg_ConnectOutputOp : Dfg_Op<
        "connect.output",
        []>
{
    let summary = "Connect region output ports with channels.";

    let arguments = (ins Dfg_InputType:$regionPort,
                         Dfg_OutputType:$channelPort);

    let hasCustomAssemblyFormat = 1;
}

def Dfg_ProcessOp : Dfg_Op<
        "process",
        [
            AutomaticAllocationScope,
            // AttrSizedOperandSegments,
            IsolatedFromAbove,
            HasParent<"ModuleOp">,
            NoTerminator
        ]> {
    let summary = "Defines a process in the KPN MoC";
    let description = [{
        The `process` operation produces a dataflow node which is linked to other
        processes by hooking up their input and output ports.
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
                         TypeAttrOf<FunctionType>: $function_type,
                         // the multiplicity of the operation, i.e. how many elements from each channel are required for execution
                         DefaultValuedAttr<DenseI64ArrayAttr, "{}">: $multiplicity);
                         // OptionalAttr<DenseI32ArrayAttr>: $multiplicity);
                        //  TypeArrayAttr:$inputs,
                        //  TypeArrayAttr:$outputs);

    // The FunctionOpInterface requires zero or one region attached to the process
    let regions = (region AnyRegion: $body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$function_type,
                       CArg<"ArrayRef<int64_t>", "{}">:$multiplicity)>
    ];

    let extraClassDeclaration = [{
        /// Returns whether the process is externally defined
        bool isExternal();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;
}

def Dfg_OperatorOp : Dfg_Op<"operator", [
            AutomaticAllocationScope,
            // AttrSizedOperandSegments,
            IsolatedFromAbove,
            HasParent<"ModuleOp">,
            NoTerminator
            // SingleBlockImplicitTerminator<"YieldOp">,
        ]> {
    let summary = "define an SDF node in DFG";

    let description = [{
        ```
        TODO
        ```
    }];

    let arguments = (ins SymbolNameAttr: $sym_name,
                         TypeAttrOf<FunctionType>: $function_type,
                         Variadic<AnyType>: $iterArgs);
    let regions = (region AnyRegion: $initBody,
                          AnyRegion: $body);

    // let builders = [
    //     OpBuilder<(ins "StringRef":$name, "FunctionType":$function_type,
    //                    CArg<"ValueRange", "{}">: $iterArgs)>
    // ];

    let extraClassDeclaration = [{
        /// Returns whether the process is externally defined
        bool isExternal();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_OutputOp : Dfg_Op<"output", [
            // Terminator,
            HasParent<"OperatorOp">
        ]> {
    let summary = "Yield op for a dfg.operator";

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let hasVerifier = 1;
}

def Dfg_YieldOp : Dfg_Op<"yield", [
            // Terminator,
            ParentOneOf<["OperatorOp", "LoopOp"]>
        ]> {
    let summary = "Yield op for a dfg.operator";

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let hasVerifier = 1;
}

// TODO:
// , RegionBranchOpInterface]>
def Dfg_LoopOp : Dfg_Op<
        "loop",
        [
            AutomaticAllocationScope,
            HasParent<"ProcessOp">,
            NoTerminator,
            AttrSizedOperandSegments,
            RecursiveMemoryEffects
        ]> {
    let summary = "Defines a list of input/output ports maybe closed";

    let description = [{
        The `loop` operation defines a list of input or output ports
        that will be closed later. The close behavior will terminate
        the process and populate to other processes so that the whole
        dataflow graph will be shut down.

        Example:
        ```
        dfg.process @add inputs(%a: type) outputs(%b: type)
        {
            %init_value = arith.constant ...
            dfg.loop inputs(%a: type) outputs(%b: type)
                    iter_args(%sum_iter = %init_value) {
                ops ...
            }
        }
        ```
    }];

    let arguments = (ins Variadic<Dfg_OutputType>: $inChans,
                         Variadic<Dfg_InputType>: $outChans,
                         Variadic<AnyType>: $iterArgs);

    let regions = (region MinSizedRegion<1>: $body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"ValueRange", "{}">:$inChans,
                       CArg<"ValueRange", "{}">:$outChans,
                       CArg<"ValueRange", "{}">: $iterArgs)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_ChannelOp : Dfg_Op<
        "channel",
        [
            Pure
        ]> {
    let summary = "Defines a channel with one input and one output port";
    let description = [{
        The `channel` operation produces a typed channel (i.e., an edge in the
        dataflow graph) that links two processes (i.e., nodes) in the dataflow
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

    let builders = [
        OpBuilder<(ins "Type":$encapsulatedType,
                       "int":$bufferSize)>
    ];

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;
}

def Dfg_SetChannelStyleOp : Dfg_Op<
    "set_channel_style",
    [
        HasParent<"ModuleOp">
    ]>
{
    let summary = "Set up global channel style";

    let arguments = (ins Dfg_ChannelStyleDefaultAttr:$channel_style);

    let assemblyFormat = [{
        $channel_style attr-dict
    }];
}

def Dfg_InstantiateOp : Dfg_Op<
    "instantiate",
    [
        AttrSizedOperandSegments//,
        // HasParent<"RegionOp">
    ]> {
    let summary = "Instantiates an process with the given inputs, forming (part of) a data flow graph.";
    let description = [{
        The `instantiate` operation instantiates a data flow graph process with
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

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}


def Dfg_PushOp : Dfg_Op<
    "push",
    [
        TypesMatchWith<"Channel type matches pushed type",
                       "chan", "inp",
                       "$_self.dyn_cast<InputType>().getElementType()">
    ]
> {
    let summary = "Pushes a value into a channel input to send it to the next process.";
    let description = [{
        The `push` operation transfers a value to another process by using a
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
        The `pull` operation retrieves a value from another process by using a
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


def Dfg_PushNOp : Dfg_Op<
    "push_n",
    []
> {
    let summary = "Pushes a list of values into a channel input to send it to the next operator.";
    let description = [{
        The `push_n` operation transfers a list of values to another operator
        by using a channel supplied as argument. Since all channels are typed,
        the element type of the memref must match the channel type. The shape
        of the memref dictates how many elements will be pushed.

        Syntax:

        ```
        TODO
        ```

        Examples:

        ```
        TODO
        ```
    }];

    let arguments = (ins AnyMemRef: $inp, Dfg_InputType: $chan);

    // let assemblyFormat = [{
    //     `(` $inp `)` $chan attr-dict `:` functional-type($inp, $chan)
    // }];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_PullNOp : Dfg_Op<
    "pull_n",
    []
> {
    let summary = "Pulls a specific number of elements from a channel.";
    let description = [{
        The `pull_n` operation retrieves a list of values from another operator
        by using a channel supplied as argument. Since all channels are typed,
        the type of the memref must match the channel type. The number of
        elements to be pulled is determined by the shape of the memred.

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
    let results = (outs AnyMemRef: $outp);

    // let assemblyFormat = [{
    //     $chan attr-dict `:` functional-type($chan, $outp)
    // }];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Intermediate HW operations
//===----------------------------------------------------------------------===//

// // Template for hw ops.
// class Dfg_HWOpBase<string mnemonic, list<Trait> traits = []>
//         : Dfg_Op< "hw." # mnemonic, traits> {}

// def Dfg_HWConnectOp
//         : Dfg_HWOpBase<"connect", []>
// {
//     let summary = "intermediate connect between ports";

//     let description = [{
//         This op will replace the dfg.push during conversion
//         of StdToCirct. It represents the actual connection
//         between top module ports and head/tail fifo buffer
//         ports.
//     }];

//     let arguments = (ins
//         Dfg_InputType: $portArgument,
//         Dfg_InputType: $portQueue
//     );

//     let hasCustomAssemblyFormat = 1;
// }

// // def Dfg_HWJoinOp
// //         : Dfg_HWOpBase<"join", []>
// // {
// //     let summary = "";
// //     let description = [{}];

// //     let arguments = (ins
// //         AnyType: $toPush,
// //         Dfg_OutputType: $channel
// //     );
// // }

// def Dfg_HWInstanceOp
//         : Dfg_HWOpBase<"instance", []>
// {
//     let summary = "";
//     let description = [{}];

//     let arguments = (ins
//         SymbolRefAttr:$module,
//         Variadic<AnyType>:$operands
//     );
//     let results = (outs Variadic<AnyType>:$results);

//     let assemblyFormat = [{
//         $module `(` $operands `)` attr-dict `:` functional-type($operands, $results)
//     }];
// }

#endif
