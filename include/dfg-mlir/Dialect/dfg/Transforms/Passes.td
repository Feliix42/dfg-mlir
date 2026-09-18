//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_TRANSFORM_PASSES
#define DFG_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def DfgInlineRegion : Pass<"dfg-inline-region">
{
    let summary = "Inlines region contents based on the specified strategy";
    let description = [{
        This pass inlines the contents of `dfg.embed` operations by replacing them
        with the cloned contents of the referenced `dfg.region`. This eliminates the
        hierarchy in the dataflow graph, flattening it to a single level.

        The pass supports multiple inlining strategies:
        - **full**: Inline all regions recursively (default)
        - **smart**: Selective inlining based on heuristics (planned for future implementation)

        After inlining, unused regions (those with no remaining embed sites) are removed.

        Example:
        ```mlir
        // Before:
        dfg.region @compute inputs(%in: i32) outputs(%out: i32) {
          %chan_in, %chan_out = dfg.channel : i32
          dfg.connect.input %in, %chan_in : i32
          dfg.instantiate @double inputs(%chan_out) outputs(%chan_in) : (i32) -> (i32)
          dfg.connect.output %chan_out, %out : i32
        }

        dfg.region @main inputs(%x: i32) outputs(%y: i32) {
          %in, %out = dfg.channel : i32
          dfg.embed @compute inputs(%in) outputs(%out) : (i32) -> (i32)
          // ...
        }

        // After --dfg-inline-region:
        dfg.region @main inputs(%x: i32) outputs(%y: i32) {
          %in, %out = dfg.channel : i32
          %compute_in, %compute_out = dfg.channel : i32
          dfg.connect.input %in, %compute_in : i32
          dfg.instantiate @double inputs(%compute_out) outputs(%compute_in) : (i32) -> (i32)
          dfg.connect.output %compute_out, %out : i32
          // ...
        }
        ```

        Note: The `dfg.region @compute` definition is removed as it's no longer needed.
    }];

    let constructor = "mlir::dfg::createDfgInlineRegionPass()";

    let options = [
        Option<"strategy", "strategy", "std::string", /*default=*/"\"full\"",
            "Strategy to inline regions: 'full' inlines all, 'smart' uses heuristics">
    ];
}

def LinalgInlineScalarArgument : Pass<"linalg-inline-scalar-argument">
{
    let summary = "Inlines scalar arguments in linalg.generic operations";
    let description = [{
        This pass processes `linalg.generic` operations to inline scalar (non-tensor)
        arguments directly into the operation body. This is useful for preparing
        linalg operations for conversion to other dialects or for code generation.

        The pass:
        1. Identifies scalar arguments in `linalg.generic` operations
        2. Removes them from the input list
        3. Maps block arguments to the removed inputs
        4. Rewrites the operation with only tensor/memref arguments

        This transformation is often used before lowering linalg operations to
        other dialects like tosa or dfg.

        Example:
        ```mlir
        // Before:
        linalg.generic {
          indexing_maps = [...],
          iterator_types = [...]
        } ins(%scalar_val: i32, %tensor: tensor<4xf32>) outs(%out: tensor<4xf32>) {
          ^bb0(%arg0: i32, %arg1: tensor<4xf32>):
            // Use %arg0 and %arg1
        }

        // After --linalg-inline-scalar-argument:
        linalg.generic {
          indexing_maps = [...],
          iterator_types = [...]
        } ins(%tensor: tensor<4xf32>) outs(%out: tensor<4xf32>) {
          ^bb0(%arg0: tensor<4xf32>):
            // %arg0 from original is mapped to %scalar_val
            // Use %scalar_val and %arg0
        }
        ```
    }];

    let constructor = "mlir::linalg::createLinalgInlineScalarArgumentPass()";
}

def DfgLowerInsideToLinalg : Pass<"dfg-lower-inside-to-linalg">
{
    let summary = "Lowers the contents of operator/process to linalg dialect level";
    let description = [{
        This pass lowers operations inside `dfg.operator` and `dfg.process` bodies
        from higher-level dialects (like tosa) to the linalg dialect. This is part of
        the progression towards LLVM code generation.

        The pass specifically:
        1. Converts tosa operations to linalg operations
        2. Removes `dfg.operator` operations (which are replaced with their contents)
        3. Preserves the `dfg` dialect structure

        This pass uses tosa's decomposition patterns (depthwise, transpose conv) and
        tosa-to-linalg conversion patterns.

        Example usage:
        ```bash
        dfg-opt --dfg-lower-inside-to-linalg input.mlir -o output.mlir
        ```

        Note: This pass requires that tosa operations are already present in the
        operator/process bodies. It's typically used after importing from ONNX or
        other high-level representations.
    }];

    let constructor = "mlir::dfg::createDfgLowerInsideToLinalgPass()";
}

def DfgOperatorToProcess : Pass<"dfg-operator-to-process">
{
    let summary = "Converts each dfg.operator to an equivalent dfg.process";
    let description = [{
        This pass transforms stateful `dfg.operator` operations into stateless
        `dfg.process` operations with equivalent semantics. The conversion wraps the
        operator's body in a `dfg.loop` and replaces state management (iter_args) with
        explicit pull/push operations on channels.

        The conversion process:
        1. Creates a new `dfg.process` with the same inputs and outputs
        2. Initializes iteration arguments using constants from the operator's initialize region
        3. Creates a `dfg.loop` that monitors all input/output ports
        4. Wraps the operator body in the loop, replacing:
           - `dfg.pull` operations with explicit channel pulls
           - `dfg.output` operations with explicit channel pushes
           - `dfg.yield` operations with channel pushes for state
        5. Removes the original `dfg.operator`

        Example:
        ```mlir
        // Before:
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

        // After --dfg-operator-to-process:
        dfg.process @accumulate inputs(%in: i32) outputs(%out: i32) {
          %zero = arith.constant 0 : i32
          dfg.loop inputs(%in) outputs(%out) iter_args(%state = %zero) {
            %input = dfg.pull %in : i32
            %new_state = arith.addi %state, %input : i32
            dfg.push (%new_state) %out : i32
            dfg.yield %new_state : i32
          }
        }
        ```

        This pass is useful for normalizing the IR to use only `dfg.process`
        operations, which may be easier to work with in certain contexts.
    }];

    let constructor = "mlir::dfg::createDfgOperatorToProcessPass()";
}

def DfgPrintGraph : Pass<"dfg-print-graph">
{
    let summary = "Prints the dataflow graph to DOT format files for visualization";
    let description = [{
        This pass generates DOT format files representing the dataflow graph structure.
        Each graph (region or module-level) is printed to a separate file. The files can
        be processed with Graphviz tools to produce visualizations.

        The pass:
        1. Walks the module to find all `dfg.region` and module-level graphs
        2. For each graph, generates a DOT file with:
           - Nodes (processes, operators, instantiations)
           - Edges (channels)
           - Cluster grouping for regions
        3. By default, outputs `.dot` files; can also generate PDF with --print-to-pdf=1

        Options:
        - `print-to-pdf`: When set to true (1), generates PDF files instead of DOT files.
          Requires `dot` and `inkscape` to be in PATH for PDF generation.

        Example usage:
        ```bash
        # Generate DOT files
        dfg-opt --dfg-print-graph input.mlir

        # Generate PDF files
        dfg-opt --dfg-print-graph --print-to-pdf=1 input.mlir
        ```

        Output files are named based on the graph name and a counter, e.g.,
        `main_graph_0.dot` or `region_name_1.pdf`.

        Note: This is a diagnostic pass for understanding and debugging the graph structure.
    }];

    let constructor = "mlir::dfg::createDfgPrintGraphPass()";

    let options = [
        Option<"printToPdf", "print-to-pdf", "bool", /*default=*/"0",
               "Print the graph to PDF files instead of DOT (requires dot and inkscape)">
    ];
}

def DfgPrintOperatorToYaml : Pass<"print-operator-to-yaml">
{
    let summary = "Prints the inner graph of each operator to YAML files";
    let description = [{
        This pass generates YAML files describing the internal structure of each
        `dfg.operator` in the module. The YAML format is designed for compatibility with
        external tools like [Mocasin](https://github.com/tud-ccc/mocasin).

        Each operator is exported to a separate YAML file containing:
        - Graph metadata (name)
        - Node information (name, type, ports, execution cycles)
        - Channel information (name, source/destination nodes and ports, initial tokens)

        The pass:
        1. Iterates through all `dfg.operator` operations in the module
        2. For each operator, extracts the graph structure
        3. Generates a YAML file with the graph description

        Example YAML output:
        ```yaml
        graphInfo:
          name: "operator_name"
        nodes:
          - name: "node1"
            type: "process"
            ports:
              - name: "in"
                type: "i32"
            exec_cycles: 10
        channels:
          - name: "chan1"
            srcNode: "node1"
            srcPort: "out"
            dstNode: "node2"
            dstPort: "in"
            initToken: 0
        ```

        Example usage:
        ```bash
        dfg-opt --print-operator-to-yaml input.mlir
        ```

        Output files are named after the operator, e.g., `operator_name.yaml`.

        Note: This pass is primarily for integration with external analysis tools.
    }];

    let constructor = "mlir::dfg::createDfgPrintOperatorToYamlPass()";
}

def MemrefRemoveScalarGlobals : Pass<"memref-remove-scalar-globals">
{
    let summary = "Removes scalar memref globals by replacing them with constant operations";
    let description = [{
        This pass eliminates scalar (0-dimensional or 1-element) memref global variables
        by replacing them with direct constant values. This simplifies the IR by
        removing unnecessary global variable declarations.

        The pass works in two stages:
        1. Identifies and removes scalar `memref.global` operations, storing their
           initial values in a map
        2. Replaces `memref.get_global` operations that accessed these globals with
           `arith.constant` operations containing the stored values

        Supported scalar types:
        - Integers (i8, i16, i32, i64, etc.)
        - Floating point (f16, f32, f64, etc.)

        Example:
        ```mlir
        // Before:
        memref.global "const_42" : memref<1xi32> = dense<42> : tensor<1xi32>
        func.func @main() {
          %val = memref.get_global @const_42 : memref<1xi32>
          %loaded = memref.load %val[] : memref<1xi32>
          // Use %loaded
        }

        // After --memref-remove-scalar-globals:
        func.func @main() {
          %val = arith.constant 42 : i32
          // Use %val directly
        }
        ```

        This pass is particularly useful for:
        - Cleaning up IR before code generation
        - Reducing the number of global variables
        - Simplifying constant propagation

        Note: Only works with constants that have dense element attributes.
    }];

    let constructor = "mlir::memref::createMemrefRemoveScalarGlobalsPass()";
}

def FuncRetrieveGraphLayers : Pass<"func-retrieve-graph-layers">
{
    let summary = "Breaks the main func into different layer functions based on JSON info";
    let description = [{
        This pass restructures a monolithic function into multiple layer functions based
        on a JSON configuration file. This is used for layer-based code generation where
        different parts of the computation need to be in separate functions.

        The pass:
        1. Reads a JSON file containing layer information (path specified via `info-path`)
        2. Parses the JSON to extract layer definitions
        3. Splits the main function's body into separate functions for each layer
        4. Maintains connections between layers through function calls

        JSON file format:
        ```json
        [
          {
            "index": 0,
            "name": "layer1",
            "start_op": "op_name",
            "inputs": [
              {"shape": [1, 3, 224, 224], "type": "f32", "is_const": false},
              ...
            ],
            "output": {
              "shape": [1, 64, 112, 112],
              "type": "f32",
              "is_const": false
            }
          },
          ...
        ]
        ```

        Options:
        - `info-path`: Path to the JSON file containing layer information

        Example usage:
        ```bash
        dfg-opt --func-retrieve-graph-layers --info-path=layers.json input.mlir -o output.mlir
        ```

        This pass is typically used for:
        - Layer-based code generation (e.g., for neural networks)
        - Preparing IR for hardware targeting
        - Modularizing large functions

        Note: The pass assumes the main function contains the entire graph to be split.
    }];

    let constructor = "mlir::func::createFuncRetrieveGraphLayersPass()";

    let options = [
        Option<"infoPath", "info-path", "std::string", /*default=*/"",
            "Path to the JSON file containing layer information">
    ];
}

def LinalgSoftTranspose : Pass<"linalg-soft-transpose">
{
    let summary = "Transposes tensors using a software (non-hardware) implementation";
    let description = [{
        This pass processes `linalg.transpose` operations with constant tensor inputs
        and performs the transpose at compile time, replacing the operation with a new
        constant tensor containing the transposed data.

        The pass:
        1. Identifies `linalg.transpose` operations with constant inputs
        2. Extracts the input tensor data and permutation vector
        3. Computes the transposed tensor in memory
        4. Replaces the transpose operation with a new constant containing the result

        Currently supports:
        - Constant dense tensor inputs
        - Arbitrary permutation vectors
        - Any numeric element type (integer and floating point)

        Example:
        ```mlir
        // Before:
        %input = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
        %perm = arith.constant dense<[1, 0]> : tensor<2xi64>
        %result = linalg.transpose %input, %perm : tensor<2x2xi32>, tensor<2xi64>

        // After --linalg-soft-transpose:
        %result = arith.constant dense<[[1, 3], [2, 4]]> : tensor<2x2xi32>
        ```

        This pass is useful for:
        - Constant folding of transpose operations
        - Reducing runtime computation
        - Simplifying IR before hardware mapping

        Note: Only works with constant inputs. Dynamic transpose operations are
        left unchanged.
    }];

    let constructor = "mlir::linalg::createLinalgSoftTransposePass()";
}

#endif //DFG_MLIR_TRANSFORM_PASSES
