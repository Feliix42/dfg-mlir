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
    let summary = "Defines a reusable subgraph (region) in a dataflow graph.";
    let description = [{
        The `region` operation defines a named subgraph that can be embedded in other
        graphs using `dfg.embed`. It has input and output ports that connect to the
        external graph, and contains an internal dataflow graph composed of channels
        and node instantiations.

        Regions enable hierarchical graph construction and reuse.

        Syntax:

        ```mlir
        dfg.region @name inputs(%in0: type, ...) outputs(%out0: type, ...) {
          // Internal graph: channels, instantiations, embeds
        }
        ```

        Example:

        ```mlir
        dfg.region @compute_pair inputs(%a: i32, %b: i32) outputs(%sum: i32, %diff: i32) {
          %a_in, %a_out = dfg.channel : i32
          %b_in, %b_out = dfg.channel : i32
          %sum_in, %sum_out = dfg.channel : i32
          %diff_in, %diff_out = dfg.channel : i32

          dfg.connect.input %a, %a_in : i32
          dfg.connect.input %b, %b_in : i32
          dfg.connect.output %sum_out, %sum : i32
          dfg.connect.output %diff_out, %diff : i32

          dfg.instantiate @add inputs(%a_out, %b_out) outputs(%sum_in) : (i32, i32) -> (i32)
          dfg.instantiate @sub inputs(%a_out, %b_out) outputs(%diff_in) : (i32, i32) -> (i32)
        }
        ```
    }];

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
    let summary = "Embeds a named region as a subgraph at the current location.";
    let description = [{
        The `embed` operation creates an instance of a named `dfg.region`, connecting its
        ports to the specified input/output channels. This allows for hierarchical graph
        construction and reuse of predefined subgraphs.

        Syntax:

        ```mlir
        dfg.embed @region_name inputs(%in_chan1, %in_chan2, ...) outputs(%out_chan1, ...) : (type, type) -> (type)
        ```

        Example:

        ```mlir
        // Embed a region that computes both sum and difference
        dfg.embed @compute_pair inputs(%x, %y) outputs(%sum, %diff) : (i32, i32) -> (i32, i32)
        ```
    }];

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
    let summary = "Connects a region input port to a channel.";
    let description = [{
        The `connect.input` operation connects a region's input port to an internal
        channel's output port. This establishes data flow from the region boundary into
        the internal graph.

        Used within `dfg.region` to wire up the external inputs to internal channels.

        Syntax:

        ```mlir
        dfg.connect.input %region_input_port, %channel_output_port : element_type
        ```

        Example:

        ```mlir
        dfg.region @my_region inputs(%in: i32) outputs(%out: i32) {
          %chan_in, %chan_out = dfg.channel : i32
          dfg.connect.input %in, %chan_in : i32
          // Now data from %in flows into %chan_in
        }
        ```

        Note: The region port has type `!dfg.output<ElementType>` (receiving end)
        and the channel port has type `!dfg.input<ElementType>` (sending end).
    }];

    let arguments = (ins Dfg_OutputType:$regionPort,
                         Dfg_InputType:$channelPort);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_ConnectOutputOp : Dfg_Op<
        "connect.output",
        []>
{
    let summary = "Connects a region output port to a channel.";
    let description = [{
        The `connect.output` operation connects an internal channel's input port to a
        region's output port. This establishes data flow from the internal graph out to
        the region boundary.

        Used within `dfg.region` to wire up internal channels to external outputs.

        Syntax:

        ```mlir
        dfg.connect.output %region_output_port, %channel_input_port : element_type
        ```

        Example:

        ```mlir
        dfg.region @my_region inputs(%in: i32) outputs(%out: i32) {
          %chan_in, %chan_out = dfg.channel : i32
          dfg.connect.output %chan_out, %out : i32
          // Now data from %chan_out flows to %out
        }
        ```

        Note: The region port has type `!dfg.input<ElementType>` (sending end)
        and the channel port has type `!dfg.output<ElementType>` (receiving end).
    }];

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
    let summary = "Defines a process in the Kahn Process Network (KPN) model";
    let description = [{
        The `process` operation defines a computation node in the dataflow graph that
        executes repeatedly in a loop, pulling data from its input channels, performing
        computation, and pushing results to its output channels.

        A process runs in an infinite loop (via `dfg.loop`) until one of its input
        channels is closed, at which point it terminates and closes its output channels.
        All processes are spawned in parallel using OpenMP.

        Syntax:

        ```mlir
        dfg.process @name inputs(%in0: type, %in1: type) outputs(%out0: type) [attributes] {
          dfg.loop inputs(%in0, %in1) outputs(%out0) {
            // Body: pull, compute, push
          }
        }
        ```

        Attributes:
        - `multiplicity`: Optional array of integers specifying how many items to
          consume from each channel per iteration (default: 1 for all)

        Examples:

        ```mlir
        // Simple adder process
        dfg.process @add inputs(%a: i32, %b: i32) outputs(%sum: i32) {
          dfg.loop inputs(%a, %b) outputs(%sum) {
            %val_a = dfg.pull %a : i32
            %val_b = dfg.pull %b : i32
            %result = arith.addi %val_a, %val_b : i32
            dfg.push (%result) %sum : i32
          }
        }

        // Process with multiplicity (consumes 2 items from each input per iteration)
        dfg.process @batch_add inputs(%a: i32, %b: i32) outputs(%sum: i32) {
          multiplicity = [2, 2]
          dfg.loop inputs(%a, %b) outputs(%sum) {
            // Will pull 2 items from each input channel per iteration
          }
        }
        ```

        Lowering: Each process is lowered to an LLVM function that runs in its own
        OpenMP task. The function blocks on channel operations (push/pull) as needed.
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
    let summary = "Defines a stateful dataflow operator with iteration arguments";

    let description = [{
        The `operator` operation defines a stateful node in the dataflow graph. Unlike
        `process`, it maintains state across iterations through `iter_args` (iteration
        arguments) that are initialized once and updated on each execution.

        An operator has two regions:
        - `initialize`: Computes the initial values for iteration arguments (executed once)
        - `body`: Computes outputs and next state values (executed repeatedly)

        Syntax:

        ```mlir
        dfg.operator @name inputs(%in0: type, ...) outputs(%out0: type, ...) iter_args(%state0: type, ...)
            initialize {
              // Compute initial state
              dfg.yield %init0, %init1 : type, type
            }
        {
          // Compute outputs
          dfg.output %result : type
          // Update state
          dfg.yield %next_state0, %next_state1 : type, type
        }
        ```

        Example:

        ```mlir
        // Stateful accumulator
        dfg.operator @accumulate inputs(%in: i32) outputs(%out: i32) iter_args(%state: i32)
          initialize {
            %zero = arith.constant 0 : i32
            dfg.yield %zero : i32
          }
        {
          %input = dfg.pull %in : i32
          %new_state = arith.addi %state, %input : i32
          dfg.output %new_state : i32
          dfg.yield %new_state : i32
        }
        ```

        Lowering: Can be converted to `dfg.process` via the `--dfg-operator-to-process`
        pass, which wraps the body in a loop with appropriate pull/push operations.
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
    let summary = "Outputs a value to an operator's output port";
    let description = [{
        The `output` operation writes a value to an operator's output port. This is how
        operators produce their outputs during each iteration.

        Must be used within the body region of a `dfg.operator`.

        Syntax:

        ```mlir
        dfg.output %value : element_type
        ```

        Example:

        ```mlir
        dfg.operator @passthrough inputs(%in: i32) outputs(%out: i32) {
          dfg.output %in : i32
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let hasVerifier = 1;
}

def Dfg_YieldOp : Dfg_Op<"yield", [
            ParentOneOf<["OperatorOp", "LoopOp"]>
        ]> {
    let summary = "Yields values from a loop or operator body";
    let description = [{
        The `yield` operation returns values from a loop iteration or operator body.
        
        In `dfg.operator`:
        - In the `initialize` region: yields initial values for iteration arguments
        - In the body region: yields updated state values for the next iteration
        
        In `dfg.loop`:
        - Yields updated iteration arguments for the next loop iteration

        Syntax:

        ```mlir
        dfg.yield %value1, %value2, ... : type1, type2, ...
        ```

        Example:

        ```mlir
        // In an operator
        dfg.operator @counter inputs(%inc: i32) outputs(%count: i32) iter_args(%state: i32)
          initialize {
            %zero = arith.constant 0 : i32
            dfg.yield %zero : i32  // Initial state
          }
        {
          %new_state = arith.addi %state, %inc : i32
          dfg.yield %new_state : i32  // Updated state
        }
        ```
    }];

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
    let summary = "Defines the monitoring loop for a process";

    let description = [{
        The `loop` operation defines which input/output ports a process monitors for
        closure. When all monitored input channels are closed, the process terminates
        and closes its output channels, which propagates termination through the graph.

        A loop can also maintain state through iteration arguments (`iter_args`), which
        are initialized once and updated on each iteration via `dfg.yield`.

        Example:
        ```mlir
        dfg.process @accumulate inputs(%a: i32) outputs(%b: i32)
        {
            %init_value = arith.constant 0 : i32
            dfg.loop inputs(%a: i32) outputs(%b: i32)
                    iter_args(%sum_iter = %init_value) {
                %val = dfg.pull %a : i32
                %new_sum = arith.addi %sum_iter, %val : i32
                dfg.push (%new_sum) %b : i32
                dfg.yield %new_sum : i32
            }
        }
        ```

        The loop body:
        - Pulls data from monitored input channels
        - Performs computation
        - Pushes results to monitored output channels
        - Optionally updates iteration arguments via `dfg.yield`

        If input channels are specified, all process input ports must be monitored.
        If output channels are specified, all process output ports must be monitored.
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
    let summary = "Defines a typed FIFO channel for data transfer between nodes";
    let description = [{
        The `channel` operation creates a typed FIFO channel (edge) in the dataflow
        graph that carries data between nodes (processes/operators). Channels can be
        unbounded or have a fixed buffer size.

        A channel has two ends:
        - Input port (`!dfg.input<ElementType>`): The sending end where data is pushed
        - Output port (`!dfg.output<ElementType>`): The receiving end where data is pulled

        For bounded channels, when the buffer is full, push operations block until
        space is available. When empty, pull operations block until data is available.

        Syntax:

        ```mlir
        %input_port, %output_port = dfg.channel[(buffer_size)] : element_type
        ```

        Examples:

        ```mlir
        // Unbounded channel for i32 values
        %in, %out = dfg.channel : i32

        // Bounded channel with capacity for 4 ui32 values
        %in, %out = dfg.channel(4) : ui32

        // Channel for tensor data
        %tensor_in, %tensor_out = dfg.channel : tensor<4x4xf32>
        ```

        Lowering: Lowered to LLVM as a typed buffer with synchronization primitives.
        Bounded channels use a circular buffer with semaphores for synchronization.
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
    let summary = "Instantiates a process or operator node in the dataflow graph";
    let description = [{
        The `instantiate` operation creates a concrete instance of a named `dfg.process`
        or `dfg.operator` in the dataflow graph, connecting it to the specified input and
        output channels. This is how nodes are actually placed and wired in the graph.

        Syntax:

        ```mlir
        dfg.instantiate @process_name inputs(%in_chan1, %in_chan2, ...) outputs(%out_chan1, ...) : (type, type) -> (type)
        ```

        Attributes:
        - `callee`: Symbol reference to the process/operator to instantiate
        - `offloaded`: Boolean flag (default: false) indicating if the node should be offloaded

        Example:

        ```mlir
        // Create channels
        %input_a, %input_a_out = dfg.channel : i32
        %input_b, %input_b_out = dfg.channel : i32
        %output_sum, %output_sum_in = dfg.channel : i32

        // Instantiate an adder process
        dfg.instantiate @add inputs(%input_a_out, %input_b_out) outputs(%output_sum_in) : (i32, i32) -> (i32)

        // Instantiate with offloading
        dfg.instantiate @expensive_kernel inputs(%in) outputs(%out) offloaded : (f32) -> (f32)
        ```

        Verification: The function type (inputs and outputs) must match the referenced
        process or operator's signature.

        Lowering: Each instantiation becomes a function call in the lowered LLVM code.
    }];

    let arguments = (ins SymbolRefAttr:$callee,
                         Variadic<Dfg_OutputType>:$inputs,
                         Variadic<Dfg_InputType>:$outputs,
                         DefaultValuedAttr<BoolAttr, "false">:$offloaded);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "std::string":$callee,
                       "ValueRange":$inputs,
                       "ValueRange":$outputs,
                       CArg<"bool", "false">: $offloaded)>
    ];

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
                       "dyn_cast<InputType>($_self).getElementType()">
    ]
> {
    let summary = "Pushes a value into a channel to send it to another node";
    let description = [{
        The `push` operation sends a value through a channel to another process or
        operator. If the channel has a bounded buffer that is full, this operation
        blocks until space is available.

        Syntax:

        ```mlir
        dfg.push (%value) %channel_output : element_type
        ```

        Operands:
        - `inp`: The value to push (must match channel's element type)
        - `chan`: Output (sending) end of the channel (`!dfg.input<ElementType>`)

        Example:

        ```mlir
        %computed = arith.addi %a, %b : i32
        dfg.push (%computed) %output_chan : i32

        // Pushing a tensor
        %tensor_result = linalg.matmul %mat_a, %mat_b : tensor<4x4xf32>
        dfg.push (%tensor_result) %tensor_output : tensor<4x4xf32>
        ```

        Lowering: Lowered to a function call that blocks on the channel's send
        operation when the buffer is full.

        Verification: The type of `inp` must match the channel's element type.
    }];

    let arguments = (ins AnyType: $inp, Dfg_InputType: $chan);

    let hasCustomAssemblyFormat = 1;
}

def Dfg_PullOp : Dfg_Op<
    "pull",
    [
        TypesMatchWith<"Channel type matches pulled type",
                       "chan", "outp",
                       "dyn_cast<OutputType>($_self).getElementType()">,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]
> {
    let summary = "Pulls a value from a channel";
    let description = [{
        The `pull` operation receives a value from a channel. If the channel is empty,
        this operation blocks until data is available. The result type matches the
        channel's element type.

        Syntax:

        ```mlir
        %value = dfg.pull %channel_input : element_type
        ```

        Operands:
        - `chan`: Input (receiving) end of the channel (`!dfg.output<ElementType>`)

        Results:
        - `outp`: The pulled value of type `element_type`

        Example:

        ```mlir
        %data = dfg.pull %input_chan : i32
        %result = arith.addi %data, %data : i32

        // Pulling a tensor
        %tensor_data = dfg.pull %tensor_input : tensor<4x4xf32>
        ```

        Lowering: Lowered to a function call that blocks on the channel's receive
        operation when the buffer is empty.
    }];

    let arguments = (ins Dfg_OutputType: $chan);
    let results = (outs AnyType: $outp);

    let hasCustomAssemblyFormat = 1;
}

def Dfg_PushNOp : Dfg_Op<
    "push_n",
    []
> {
    let summary = "Pushes multiple values from a memref into a channel";
    let description = [{
        The `push_n` operation transfers multiple values from a memref to a channel.
        The element type of the memref must match the channel's element type. The number
        of elements pushed is determined by the memref's shape.

        This is useful for batch operations where multiple values need to be sent at once.

        Syntax:

        ```mlir
        dfg.push_n (%memref) %channel_output : memref<shape x element_type>
        ```

        Operands:
        - `inp`: Memref containing values to push
        - `chan`: Output (sending) end of the channel (`!dfg.input<ElementType>`)

        Example:

        ```mlir
        // Push 10 i32 values from a memref
        %data = memref.alloc() : memref<10xi32>
        // ... fill data with values ...
        dfg.push_n (%data) %output_chan : memref<10xi32>

        // Push a 2D tensor of f32 values
        %matrix = memref.alloc() : memref<4x4xf32>
        dfg.push_n (%matrix) %matrix_output : memref<4x4xf32>
        ```

        Lowering: Lowered to a function call that iterates over the memref and pushes
        each element to the channel, blocking as needed.

        Verification: The element type of the memref must match the channel's element type.
    }];

    let arguments = (ins AnyMemRef: $inp, Dfg_InputType: $chan);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Dfg_PullNOp : Dfg_Op<
    "pull_n",
    []
> {
    let summary = "Pulls multiple values from a channel into a memref";
    let description = [{
        The `pull_n` operation receives multiple values from a channel and stores them
        in a memref. The element type of the memref must match the channel's element type.
        The number of elements pulled is determined by the memref's shape.

        This is useful for batch operations where multiple values need to be received at once.

        Syntax:

        ```mlir
        %memref = dfg.pull_n %channel_input : memref<shape x element_type>
        ```

        Operands:
        - `chan`: Input (receiving) end of the channel (`!dfg.output<ElementType>`)

        Results:
        - `outp`: A memref filled with pulled values

        Example:

        ```mlir
        // Pull 10 i32 values into a memref
        %buffer = dfg.pull_n %input_chan : memref<10xi32>

        // Pull a 2D tensor of f32 values
        %matrix = dfg.pull_n %matrix_input : memref<4x4xf32>
        ```

        Lowering: Lowered to a function call that iterates and pulls elements from the
        channel into the memref, blocking as needed.

        Verification: The element type of the memref must match the channel's element type.
    }];

    let arguments = (ins Dfg_OutputType: $chan);
    let results = (outs AnyMemRef: $outp);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

#endif
