//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_CONVERSION_PASSES
#define DFG_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertDfgToFunc : Pass<"convert-dfg-nodes-to-func"> {
    let summary = "Convert DFG dialect graph nodes to Func declarations and invocations";
    let description = [{
        This pass converts DFG dialect operations (`dfg.process`, `dfg.instantiate`,
        `dfg.loop`) to their equivalent representations using the Func dialect and
        Control Flow (cf) dialect, with OpenMP for parallel execution.

        The pass performs the following conversions:
        - `dfg.process` -> `func.func`: Each process becomes a function with combined
          input and output ports as arguments (no return values)
        - `dfg.instantiate` -> `func.call` wrapped in `omp.section`: Each instantiation
          becomes a function call inside an OpenMP section for parallel execution
        - `dfg.loop` -> `cf` operations: Loop operations are converted to control flow
          operations (cf.cond_br, cf.br) with appropriate block splitting

        Processes are grouped into OpenMP parallel regions, with each group of
        successive instantiations wrapped in an `omp.parallel` containing `omp.sections`
        for concurrent execution.

        The lowering preserves the parallel semantics of the dataflow graph:
        - All processes run in parallel via OpenMP
        - Channels are still represented as DFG types (not yet lowered to LLVM)
        - Loop termination is handled by conditional branches

        Example:
        ```mlir
        // Before:
        dfg.process @add inputs(%a: i32, %b: i32) outputs(%sum: i32) {
          dfg.loop inputs(%a, %b) outputs(%sum) {
            // ...
          }
        }
        dfg.instantiate @add inputs(%x, %y) outputs(%z) : (i32, i32) -> (i32)

        // After:
        func.func @add(%a: i32, %b: i32, %sum: i32) {
          // Loop body as CF operations
          // ...
          cf.br ^bb_end
        }

        omp.parallel {
          omp.sections {
            omp.section {
              func.call @add(%x, %y, %z)
              omp.terminator
            }
            omp.terminator
          }
          omp.terminator
        }
        ```

        Note: This pass does not lower channel operations (push/pull). Use
        `--convert-dfg-edges-to-llvm` for that.

        Dependent Dialects: func, cf, omp, arith, dfg
    }];

    let constructor = "mlir::createConvertDfgToFuncPass()";

    let dependentDialects = [
        "func::FuncDialect",
        "cf::ControlFlowDialect",
        "omp::OpenMPDialect",
        "arith::ArithDialect",

        "dfg::DfgDialect"
    ];
}


def ConvertDfgToLLVM : Pass<"convert-dfg-edges-to-llvm"> {
    let summary = "Convert DFG dialect graph edges (channels) to LLVM implementation";
    let description = [{
        This pass lowers DFG channel operations (`dfg.channel`, `dfg.push`, `dfg.pull`,
        `dfg.push_n`, `dfg.pull_n`) to LLVM dialect calls that use external channel
        primitives. The actual channel implementation is in `playground/channels.c`,
        which provides the runtime channel primitives.

        The pass performs the following conversions:

        **Channel Creation (`dfg.channel`):**
        - Lowered to a call to external `channel(bytewidth)` function
        - The byte width is computed from the element type using pointer arithmetic
        - Returns a pointer to the channel struct (cast to dfg types via
          UnrealizedConversionCastOp)

        **Push Operations (`dfg.push`, `dfg.push_n`):**
        - Allocates stack memory for the data using `llvm.alloca`
        - Stores the value into the allocated memory using `llvm.store`
        - Calls external `push(sender_ptr, data_ptr)` or `push_n(sender_ptr, data_ptr, count)`
        - Returns a boolean indicating success
        - On failure (channel full), inserts conditional branch to handle blocking:
          - If push fails, jumps to a teardown block with poison values
          - This models the blocking semantics where the operation yields until space is available

        **Pull Operations (`dfg.pull`, `dfg.pull_n`):**
        - Allocates stack memory for the result using `llvm.alloca`
        - Calls external `pull(receiver_ptr, result_ptr)` or `pull_n(receiver_ptr, result_ptr, count)`
        - Returns a boolean indicating success
        - Loads the value from allocated memory using `llvm.load`
        - On failure (channel empty), inserts conditional branch to handle blocking:
          - If pull fails, jumps to a teardown block with poison values
          - This models the blocking semantics where the operation yields until data is available

        **Channel Closure:**
        - Inserts calls to `close_channel(channel_ptr)` for proper cleanup
        - Uses reference counting to ensure channels are only freed when fully disconnected

        The external channel primitives (defined in `playground/channels.c`) implement:
        - FIFO buffers with atomic operations for thread safety
        - Blocking semantics using busy-wait loops (with commented-out taskyield)
        - Circular buffer management with head/tail indices
        - Support for bounded channels with configurable capacity (CHANSIZE=20000)
        - Atomic occupancy tracking for synchronization

        Example:
        ```mlir
        // Before:
        %in, %out = dfg.channel : i32
        %val = dfg.pull %in : i32
        dfg.push (%val) %out : i32

        // After (simplified):
        // Compute bytewidth from i32 type
        %bytewidth = ... : i64
        %chan_ptr = call @channel(%bytewidth) : (i64) -> !llvm.ptr
        // Cast to dfg types via UnrealizedConversionCastOp
        %in_chan = unrealized_conversion_cast %chan_ptr : !llvm.ptr to !dfg.input<i32>
        %out_chan = unrealized_conversion_cast %chan_ptr : !llvm.ptr to !dfg.output<i32>

        // Pull:
        %alloc = llvm.alloca %result : i32
        %success = call @pull(%in_chan_ptr, %alloc) : (!llvm.ptr, !llvm.ptr) -> i1
        %val = llvm.load %alloc : !llvm.ptr -> i32
        cf.cond_br %success, ^bb_continue, ^bb_teardown(...)

        // Push:
        %alloc = llvm.alloca %val : i32
        llvm.store %val, %alloc
        %casted = llvm.bitcast %alloc : !llvm.ptr to !llvm.ptr
        %success = call @push(%out_chan_ptr, %casted) : (!llvm.ptr, !llvm.ptr) -> i1
        cf.cond_br %success, ^bb_continue, ^bb_teardown(...)
        ```

        Note: This pass requires that node operations have already been converted to
        func/cf/omp via `--convert-dfg-nodes-to-func`. The two passes are typically used
        together in a pipeline.

        Note: The actual channel implementation in `playground/channels.c` uses:
        - Atomic operations for thread-safe access
        - Busy-wait loops for blocking (currently without taskyield)
        - Fixed-size circular buffers (CHANSIZE=20000)

        Dependent Dialects: cf, func, LLVM, dfg
    }];

    let constructor = "mlir::createConvertDfgToLLVMPass()";

    let dependentDialects = [
        "cf::ControlFlowDialect",
        "func::FuncDialect",
        "LLVM::LLVMDialect",

        "dfg::DfgDialect"
    ];
}


def ConvertDfgToOlympus : Pass<"convert-dfg-to-olympus"> {
    let summary = "Convert offloaded DFG nodes to Olympus-MLIR dialect";
    let description = [{
        This pass converts offloaded `dfg.instantiate` operations (those with the
        `offloaded` attribute set to true) to Olympus-MLIR dialect operations. Olympus
        is a dialect for FPGA acceleration, and this pass prepares offloaded nodes for
        hardware acceleration.

        The pass:
        1. Identifies all `dfg.instantiate` operations with `offloaded=true`
        2. For each offloaded instantiation:
           a. Finds the corresponding `dfg.process` definition
           b. Verifies that argument counts match (inputs + outputs must equal operands)
           c. Creates an Olympus wrapper function with the name `<process>_wrapper`
           d. In the wrapper, creates Olympus channels for each port
           e. Inserts `olympus.execute` to invoke the kernel on the FPGA
        3. Replaces the original `dfg.instantiate` with a call to the Olympus wrapper

        The Olympus wrapper function:
        - Has the same signature as the original process
        - Creates Olympus channel objects for each input/output port using `olympus.channel`
        - Configures channels based on multiplicity attributes from the process
        - Uses `olympus.execute` to run the kernel on the FPGA
        - Parses configuration from OLYMPUS_CONFIG environment variable or attributes

        Channel multiplicity from the process is used to configure Olympus channels:
        - If multiplicity is specified, channels are configured accordingly
        - If not specified, defaults to 1 for all ports

        Example:
        ```mlir
        // Before:
        dfg.process @kernel inputs(%a: i32, %b: i32) outputs(%c: i32) {
          // ...
        }
        dfg.instantiate @kernel inputs(%x, %y) outputs(%z) offloaded : (i32, i32) -> (i32)

        // After (simplified):
        func.func @kernel_wrapper(%a: i32, %b: i32, %c: i32) {
          %chan_a = olympus.channel %a : i32
          %chan_b = olympus.channel %b : i32
          %chan_c = olympus.channel %c : i32
          olympus.execute @kernel(%chan_a, %chan_b, %chan_c)
          return
        }

        func.call @kernel_wrapper(%x, %y, %z)
        ```

        Note: This pass only processes offloaded instantiations. Non-offloaded nodes
        remain as DFG operations and should be processed by other passes (e.g.,
        `--convert-dfg-nodes-to-func` and `--convert-dfg-edges-to-llvm`).

        Note: The Olympus dialect must be available for this pass to work.

        Dependent Dialects: func, dfg
    }];

    let constructor = "mlir::createConvertDfgToOlympusPass()";

    let dependentDialects = [
        "func::FuncDialect",
        "dfg::DfgDialect"
    ];
}


def ConvertDfgInsertOlympusWrapper : Pass<"insert-olympus-wrappers"> {
    let summary = "Create Alveo host object wrapper for offloaded DFG nodes";
    let description = [{
        This pass creates a host wrapper for managing Alveo FPGA devices when Olympus
        offloading is used. It inserts a function that creates an Alveo host object and
        manages the lifetime of FPGA resources.

        The pass:
        1. Collects all offloaded `dfg.instantiate` operations
        2. Groups them for batch processing
        3. Creates a wrapper function (default name: `createAlveoHostObjectWrapper`)
        4. In the wrapper, calls `createAlveoHost` to initialize the FPGA device
        5. Stores the host object pointer for use by other passes

        This is a supporting pass for the Olympus offloading workflow. The Alveo host
        object is required for managing FPGA resources when using Xilinx Alveo cards
        with Olympus-MLIR.

        The pass also includes stub patterns for `dfg.push_n` and `dfg.pull_n` lowering
        to Olympus-compatible code, though these are currently marked as TODO.

        Typical usage pipeline:
        ```bash
        dfg-opt --insert-olympus-wrappers \
               --convert-dfg-to-olympus \
               --convert-dfg-nodes-to-func \
               --convert-dfg-edges-to-llvm \
               input.mlir -o output.mlir
        ```

        Note: This pass requires the LLVM dialect for pointer operations.

        Note: The actual FPGA device initialization code (`createAlveoHost`) must be
        provided externally or linked at compilation time. The generated code calls
        this function but does not define it.

        Dependent Dialects: func, dfg, LLVM
    }];

    let constructor = "mlir::createConvertDfgInsertOlympusWrapperPass()";

    let dependentDialects = [
        "func::FuncDialect",
        "dfg::DfgDialect",
        "LLVM::LLVMDialect"
    ];
}


def ConvertFuncToDfg : Pass<"convert-func-to-dfg"> {
    let summary = "Convert Func dialect functions to DFG representation";
    let description = [{
        This pass converts `func.func` operations to DFG dialect representation, creating
        a dataflow graph from imperative function code. This is useful for importing
        existing imperative code into the DFG dialect.

        The pass processes each function in the module and performs one of two conversions:

        **For regular functions (nodes):**
        1. Creates a `dfg.operator` with the function's name and type signature
        2. Copies the function body into the operator's body region
        3. Maps function arguments to operator ports
        4. Stores the operator in a node map for later reference

        **For the special region function:**
        1. Creates a `dfg.region` with the function's name
        2. Creates input and output channels for each function argument/result
        3. For each `func.call` in the function body:
           a. Looks up the corresponding operator in the node map
           b. Creates channels for the operator's outputs
           c. Instantiates the operator with `dfg.instantiate`
           d. Connects the operator's outputs to its output channels
        4. Connects region inputs to channel inputs via `dfg.connect.input`
        5. Connects channel outputs to region outputs via `dfg.connect.output`

        The conversion preserves the data flow between functions:
        - Function arguments become region inputs (via channels)
        - Function return values become region outputs (via channels)
        - Function calls become operator instantiations
        - Intermediate values flow through channels

        Example:
        ```mlir
        // Before:
        func.func @add(%a: i32, %b: i32) -> i32 {
          %sum = arith.addi %a, %b : i32
          return %sum : i32
        }

        func.func @main(%x: i32, %y: i32) -> i32 {
          %result = func.call @add(%x, %y) : (i32, i32) -> i32
          return %result : i32
        }

        // After (simplified):
        dfg.operator @add inputs(%a: i32, %b: i32) outputs(%sum: i32) {
          %sum = arith.addi %a, %b : i32
          dfg.output %sum : i32
        }

        dfg.region @main inputs(%x: i32) outputs(%result: i32) {
          %x_chan_in, %x_chan_out = dfg.channel : i32
          %y_chan_in, %y_chan_out = dfg.channel : i32
          %result_chan_in, %result_chan_out = dfg.channel : i32

          dfg.connect.input %x, %x_chan_in : i32
          dfg.connect.input %y, %y_chan_in : i32

          dfg.instantiate @add inputs(%x_chan_out, %y_chan_out) outputs(%result_chan_in) : (i32, i32) -> (i32)

          dfg.connect.output %result_chan_out, %result : i32
        }
        ```

        Note: The pass uses internal state (regionFunc, numResultUse, nodeMap) to track
        which function is the region and which are nodes, and to map between them.

        Note: The pass currently assumes a specific structure where one function is
        the "region" (identified by comparing with regionFunc) and others are "nodes"
        to be instantiated.

        Dependent Dialects: dfg, func
    }];

    let constructor = "mlir::createConvertFuncToDfgPass()";

    let dependentDialects = [
        "dfg::DfgDialect",
        "func::FuncDialect"
    ];
}


def ConvertLinalgToDfg : Pass<"convert-linalg-to-dfg"> {
    let summary = "Convert Linalg dialect operations to DFG representation";
    let description = [{
        This pass converts `linalg` dialect operations to DFG representation. This is
        useful for bringing high-level linear algebra operations into the dataflow
        graph model.

        Currently, this pass sets up the conversion infrastructure but the actual
        patterns for converting specific linalg operations to DFG are not yet
        implemented (the pattern set is empty via `populateLinalgToDfgConversionPatterns`
        which does nothing).

        The pass:
        1. Sets up a TypeConverter that performs identity conversion (preserves types)
        2. Configures the conversion target to:
           - Keep dfg and arith dialects as legal
           - Remove linalg and func dialects (make illegal)
        3. Applies partial conversion with the (currently empty) pattern set

        Future work: Implement patterns to convert specific linalg operations
        (e.g., `linalg.matmul`, `linalg.add`, `linalg.generic`, etc.) to equivalent
        DFG computations using processes and channels.

        Example of intended usage:
        ```mlir
        // Before (conceptual):
        linalg.matmul %A, %B, %C : tensor<MxKxf32>, tensor<KxNxf32>, tensor<MxNxf32>

        // After (conceptual):
        // Would be converted to a DFG region with appropriate processes
        // for the matrix multiplication, with channels connecting them
        dfg.region @matmul inputs(%A: tensor<MxKxf32>, %B: tensor<KxNxf32>) outputs(%C: tensor<MxNxf32>) {
          // Channels and instantiations for matrix multiply
        }
        ```

        Note: This pass is currently a placeholder and requires pattern
        implementations to be useful. The function
        `populateLinalgToDfgConversionPatterns` currently adds no patterns.

        Dependent Dialects: dfg, arith, linalg, func
    }];

    let constructor = "mlir::createConvertLinalgToDfgPass()";

    let dependentDialects = [
        "dfg::DfgDialect",

        "arith::ArithDialect",
        "linalg::LinalgDialect"
    ];
}

#endif // DFG_MLIR_CONVERSION_PASSES
