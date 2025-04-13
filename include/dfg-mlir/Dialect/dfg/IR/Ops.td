//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_OPS
#define DFG_OPS

include "dfg-mlir/Dialect/dfg/Enums.td"
include "dfg-mlir/Dialect/dfg/Interfaces/Interfaces.td"
include "dfg-mlir/Dialect/dfg/IR/Dialect.td"
include "dfg-mlir/Dialect/dfg/IR/Types.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"

def Dfg_RegionOp : Dfg_Op<
        "region",
        [
            Dfg_NodeOpInterface,
            Dfg_GraphOpInterface,
            IsolatedFromAbove,
            NoTerminator,
            DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
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
        //===------------------------------------------------------------------===//
        // DFGMLIR NodeOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName() { return getSymName().str(); }
        unsigned getNumInputPorts() { return getFunctionType().getNumInputs(); }
        Value getInputPort(unsigned index) {
            return getBody().getArgument(index);
        }
        SmallVector<Type> getInputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getFunctionType().getInputs())
                types.push_back(cast<OutputType>(type).getElementType());
            return types;
        }
        SmallVector<Value> getInputPorts() {
            SmallVector<Value> inputs;
            for (size_t i = 0; i < getNumInputPorts(); ++i) {
                inputs.push_back(getInputPort(i));
            }
            return inputs;
        }
        unsigned getNumOutputPorts() { return getFunctionType().getNumResults(); }
        Value getOutputPort(unsigned index) {
            unsigned idx = index + getNumInputPorts();
            return getBody().getArgument(idx);
        }
        SmallVector<Type> getOutputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getFunctionType().getResults()) {
                auto inTy = cast<InputType>(type);
                types.push_back(inTy.getElementType());
            }
            return types;
        }
        SmallVector<Value> getOutputPorts() {
            SmallVector<Value> outputs;
            for (size_t i = 0; i < getNumOutputPorts(); ++i) {
                outputs.push_back(getOutputPort(i));
            }
            return outputs;
        }

        //===------------------------------------------------------------------===//
        // DFGMLIR GraphOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getGraphName() { return getNodeName(); }
        bool isSubGraph() {
            bool isSub = false;
            auto module = getOperation()->getParentOfType<ModuleOp>();
            module.walk([&](EmbedOp embedOp) {
                if (embedOp.getNodeName() == getGraphName()) {
                    isSub = true;
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            });
            return isSub;
        }
        SmallVector<Operation*> getGraphNodes() {
            SmallVector<Operation*> nodes;
            for (auto &opi : getBody().getOps())
                if (isa<NodeOpInterface>(opi) && !isa<GraphOpInterface>(opi))
                    nodes.push_back(&opi);
            return nodes;
        }
        SmallVector<Operation*> getGraphSubGs() {
            SmallVector<Operation*> subgraphs;
            for (auto &opi : getBody().getOps())
                if (isa<NodeOpInterface>(opi) && isa<GraphOpInterface>(opi))
                    subgraphs.push_back(&opi);
            return subgraphs;
        }
        SmallVector<Operation*> getGraphEdges() {
            SmallVector<Operation*> edges;
            for (auto &opi : getBody().getOps())
                if (isa<EdgeOpInterface>(opi))
                    edges.push_back(&opi);
            return edges;
        }
    }];
}

def Dfg_EmbedOp : Dfg_Op<
        "embed",
        [
            Dfg_NodeOpInterface,
            Dfg_GraphOpInterface,
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

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // DFGMLIR NodeOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName() { return getCallee().getRootReference().str(); }
        unsigned getNumInputPorts() { return std::ranges::size(getInputs()); }
        Value getInputPort(unsigned index) {
            return getOperation()->getOperand(index);
        }
        SmallVector<Type> getInputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getInputs().getTypes()) {
                auto outTy = cast<OutputType>(type);
                types.push_back(outTy.getElementType());
            }
            return types;
        }
        SmallVector<Value> getInputPorts() { return getInputs(); }
        unsigned getNumOutputPorts() { return std::ranges::size(getOutputs()); }
        Value getOutputPort(unsigned index) {
            unsigned idx = index + getNumInputPorts();
            return getOperation()->getOperand(idx);
        }
        SmallVector<Type> getOutputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getOutputs().getTypes()) {
                auto inTy = cast<InputType>(type);
                types.push_back(inTy.getElementType());
            }
            return types;
        }
        SmallVector<Value> getOutputPorts() { return getOutputs(); }

        //===------------------------------------------------------------------===//
        // DFGMLIR GraphOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getGraphName() { return getNodeName(); }
        bool isSubGraph() { return true; }

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        FunctionType getFunctionType() {
            FunctionType::get(
                getContext(),
                getInputs().getTypes(),
                getOutputs().getTypes()
            );
        }
    }];
}

def Dfg_ConnectInputOp : Dfg_Op<
        "connect.input",
        []>
{
    let summary = "Connect region input ports with channels.";

    let arguments = (ins Dfg_OutputType:$regionPort,
                         Dfg_InputType:$channelPort);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_ConnectOutputOp : Dfg_Op<
        "connect.output",
        []>
{
    let summary = "Connect region output ports with channels.";

    let arguments = (ins Dfg_InputType:$regionPort,
                         Dfg_OutputType:$channelPort);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_ProcessOp : Dfg_Op<
        "process",
        [
            Dfg_NodeOpInterface,
            AffineScope,
            AutomaticAllocationScope,
            // AttrSizedOperandSegments,
            IsolatedFromAbove,
            HasParent<"ModuleOp">,
            NoTerminator,
            DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
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

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // DFGMLIR NodeOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName() { return getSymName().str(); }
        unsigned getNumInputPorts() { return getFunctionType().getNumInputs(); }
        Value getInputPort(unsigned index) {
            return getBody().getArgument(index);
        }
        SmallVector<Type> getInputPortTypes() {
            SmallVector<Type> types;
            // types.clear();
            for (auto type : getFunctionType().getInputs())
                types.push_back(cast<OutputType>(type).getElementType());
            return types;
        }
        SmallVector<Value> getInputPorts() {
            SmallVector<Value> inputs;
            for (size_t i = 0; i < getNumInputPorts(); ++i) {
                inputs.push_back(getInputPort(i));
            }
            return inputs;
        }
        unsigned getNumOutputPorts() { return getFunctionType().getNumResults(); }
        Value getOutputPort(unsigned index) {
            unsigned idx = index + getNumInputPorts();
            return getBody().getArgument(idx);
        }
        SmallVector<Type> getOutputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getFunctionType().getResults()) {
                auto inTy = cast<InputType>(type);
                types.push_back(inTy.getElementType());
            }
            return types;
        }
        SmallVector<Value> getOutputPorts() {
            SmallVector<Value> outputs;
            for (size_t i = 0; i < getNumOutputPorts(); ++i) {
                outputs.push_back(getOutputPort(i));
            }
            return outputs;
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        /// Returns whether the process is externally defined
        bool isExternal();
    }];
}

def Dfg_OperatorOp : Dfg_Op<"operator", [
            Dfg_NodeOpInterface,
            AffineScope,
            AutomaticAllocationScope,
            // AttrSizedOperandSegments,
            IsolatedFromAbove,
            HasParent<"ModuleOp">,
            NoTerminator
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

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$function_type,
                       CArg<"ArrayRef<Type>", "{}">:$iter_args)>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // DFGMLIR NodeOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName() { return getSymName().str(); }
        unsigned getNumInputPorts() { return getFunctionType().getNumInputs(); }
        Value getInputPort(unsigned index) {
            return getBody().getArgument(index);
        }
        SmallVector<Type> getInputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getFunctionType().getInputs())
                types.push_back(type);
            return types;
        }
        SmallVector<Value> getInputPorts() {
            SmallVector<Value> inputs;
            for (size_t i = 0; i < getNumInputPorts(); ++i) {
                inputs.push_back(getInputPort(i));
            }
            return inputs;
        }
        unsigned getNumOutputPorts() { return getFunctionType().getNumResults(); }
        Value getOutputPort(unsigned index) {
            unsigned idx = index + getNumInputPorts();
            return getBody().getArgument(idx);
        }
        SmallVector<Type> getOutputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getFunctionType().getResults())
                types.push_back(type);
            return types;
        }
        SmallVector<Value> getOutputPorts() {
            SmallVector<Value> outputs;
            for (size_t i = 0; i < getNumOutputPorts(); ++i) {
                outputs.push_back(getOutputPort(i));
            }
            return outputs;
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        /// Returns whether the process is externally defined
        bool isExternal();
    }];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_OutputOp : Dfg_Op<"output", [
            HasParent<"OperatorOp">
        ]> {
    let summary = "Yield op for a dfg.operator";

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let hasVerifier = 1;
}

def Dfg_YieldOp : Dfg_Op<"yield", [
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
            Dfg_EdgeOpInterface,
            MemoryEffectsOpInterface,
            DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
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
        OpBuilder<(ins "Type":$encapsulatedType)>,
        OpBuilder<(ins "Type":$encapsulatedType,
                       "int":$bufferSize)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // DFGMLIR EdgeOpInterface Methods
        //===------------------------------------------------------------------===//

        Operation* getInputConnection() { return *getInChan().getUsers().begin(); }

        Operation* getOutputConnection() { return *getOutChan().getUsers().begin(); }

        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//

        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), getOperation()->getOpResult(0),
                                 SideEffects::DefaultResource::get());
            effects.emplace_back(MemoryEffects::Write::get(), getOperation()->getOpResult(0),
                                 SideEffects::DefaultResource::get());
        }
    }];
}

def Dfg_InstantiateOp : Dfg_Op<
    "instantiate",
    [
        Dfg_NodeOpInterface,
        AttrSizedOperandSegments
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

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // DFGMLIR NodeOpInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName() { return getCallee().getRootReference().str(); }
        unsigned getNumInputPorts() { return std::ranges::size(getInputs()); }
        Value getInputPort(unsigned index) {
            return getOperation()->getOperand(index);
        }
        SmallVector<Type> getInputPortTypes() {
            SmallVector<Type> types;
            // types.clear();
            for (auto type : getInputs().getTypes()) {
                auto outTy = cast<OutputType>(type);
                types.push_back(outTy.getElementType());
            }
            return types;
        }
        SmallVector<Value> getInputPorts() { return getInputs(); }
        unsigned getNumOutputPorts() { return std::ranges::size(getOutputs()); }
        Value getOutputPort(unsigned index) {
            unsigned idx = index + getNumInputPorts();
            return getOperation()->getOperand(idx);
        }
        SmallVector<Type> getOutputPortTypes() {
            SmallVector<Type> types;
            for (auto type : getOutputs().getTypes()) {
                auto inTy = cast<InputType>(type);
                types.push_back(inTy.getElementType());
            }
            return types;
        }
        SmallVector<Value> getOutputPorts() { return getOutputs(); }

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        FunctionType getFunctionType() {
            FunctionType::get(
                getContext(),
                getInputs().getTypes(),
                getOutputs().getTypes()
            );
        }
    }];
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

    let hasCustomAssemblyFormat = 1;
}

def Dfg_PullOp : Dfg_Op<
    "pull",
    [
        TypesMatchWith<"Channel type matches pulled type",
                       "chan", "outp",
                       "$_self.dyn_cast<OutputType>().getElementType()">,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
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

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

#endif
