# Examples

## Typical MAC operator
This code snippet defines an SDF node named `mac` and instantiates it in the top region and connected with FIFO channels.

```
dfg.operator @mac inputs(%a: i32, %b: i32)
                  outputs(%c: i32)
                  iter_args(%sum: i32)
initialize {
    %0 = arith.constant 0 : i32
    dfg.yield %0 : i32
} {
    %0 = arith.muli %a, %b : i32
    %1 = arith.addi %0, %sum : i32
    dfg.output %1 : i32
    dfg.yield %1 : i32
}
dfg.region @top inputs(%arg0: i32, %arg1: i32)
                outputs(%arg2: i32)
{
    %0:2 = dfg.channel(16) : i32
    %1:2 = dfg.channel(16) : i32

    dfg.connect.input %0#0, %arg0 : i32
    dfg.connect.input %1#0, %arg1 : i32
    dfg.connect.output %2#1, %arg2 : i32

    dfg.instantiate @mac inputs(%0#1, %1#1) 
                         outputs(%2#0) : (i32, i32) -> i32
}
```
¡¡¡ This program is not supported in the converson and translation example below due to the `iter_args`. !!!

## FFT2D operator with tensors
This code snippet defines an SDF node named `fft` and instantiates it in the top region and connected with FIFO channels. Differently from the `MAC` operator above, this operator uses tensors.

```
dfg.operator @fft inputs(%arg0: tensor<1x4x8xf32>, %arg1 : tensor<1x4x8xf32>)
                        outputs(%arg2: tensor<1x4x8xf32>, %arg3: tensor<1x4x8xf32>)
{
    %0, %1 = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
    dfg.output %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

dfg.region @top inputs(%arg0: tensor<1x4x8xf32>, %arg1 : tensor<1x4x8xf32>)
                        outputs(%arg2: tensor<1x4x8xf32>, %arg3: tensor<1x4x8xf32>)
{
    %0:2 = dfg.channel(16) : tensor<1x4x8xf32>
    %1:2 = dfg.channel(16) : tensor<1x4x8xf32>
    %2:2 = dfg.channel(16) : tensor<1x4x8xf32>
    %3:2 = dfg.channel(16) : tensor<1x4x8xf32>

    dfg.connect.input %arg0, %0#0 : tensor<1x4x8xf32>
    dfg.connect.input %arg1, %1#0 : tensor<1x4x8xf32>
    dfg.connect.output %arg2, %2#1 : tensor<1x4x8xf32>
    dfg.connect.output %arg3, %3#1 : tensor<1x4x8xf32>

    dfg.instantiate @tosa_test inputs(%0#1, %1#1) outputs(%2#0, %3#0) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
}
```

## Convertion and Translation for FPGA backend
To get a working FPGA design we provide a conversion pass pipeline: `--convert-to-vitis`, which will lower up-to-tosa operations to scf level along with bufferizations of tensor values. Then operations in `arith`, `index`, `math`, `scf` and `dfg` will be converted to vitis equivalent operations. The result of this pipeline will be used in translation (i.e. `--vitis-generate-project`).

Let's assume the input file is named `dfg.mlir`, to get the bitstream file one only needs to execute these commands one by one:

1. Get the files needed for this project from current program
```
dfg-opt dfg.mlir --convert-to-vitis | dfg-translate --vitis-generate-project (--output-dir=/path/you/want/ --target-device="device-name")
```
2. Set up the environmental variables
```
export XILINX_PATH=/path/to/xilinx/install/
export XILINX_VERSION=version-number
```
3. Run through shell script
```
sh /path/to/run_design.sh
```
4. Check the outputs
```
tree /path/you/specified/driver
```
The driver for PYNQ is generated under directory `/path/you/specified/driver/driver`, which goal is to let user only take care of the input data.
To set up on PYNQ in JupyterNotebook or Python file, please use the following class and methods defined in the driver:

1. class `Accelerator`: this class' initializer accepts no arguments, one can instantiate one simply use `accl = Accelerator()`. Then there will be some important information printed out.
2. method `compute`: this method accepts a list, within which you should set up the correct number of inputs based on your own-defined `dfg.region` in MLIR.
3. method `get_execution_time`: this method will return a float number, which is the execution time in seconds roughly measured using `time` package.

In this work, [AMD Vivado Design Suite](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html) version 2024.2 is tested without any problems. If one need to use lower version of the tools, please modify the version of Xilinx IPs in [this](../lib/Target/GenerateVitisProject/GenerateVitisProject.cpp#L1170) translation respectively.


## **Conversion and Translation for the MDC Backend**

To generate a valid MDC project from MLIR, we provide a conversion pass pipeline: `--convert-to-mdc`. This pipeline transforms the input MLIR code into a set of files and directories compatible with the **Orcc Environment**. All generated files are organized within an `MDC/` directory, which includes the subdirectories `bin/`, `references/`, and `src/`.

For each MLIR file, the pipeline produces:

* **An XDF network** representing the dataflow graph.
* **Multiple CAL actor files**, each corresponding to a function or subroutine in the MLIR.
* **Two subdirectories under `src/`**:

  * `baseline/`: contains the `.xdf` file (network definition) and `.xdfdiag` (graphical metadata).
  * `custom/`: contains the generated CAL actor files for each function/operator.

The MLIR dialects supported by this pipeline include `arith`, `index`, `math`, `scf`, and `dfg`. Their operations are translated into equivalent CAL actors.

---

### **Usage Instructions**

Assuming your MLIR input file is named `dfg.mlir`, follow the steps below to generate and inspect the MDC project:

1. **Generate MDC files**

   ```bash
   dfg-opt dfg.mlir --prepare-for-mdc | dfg-translate --dfg-to-mdc --output-dir=/path/to/output
   ```

2. **Inspect the generated files**
   Navigate to the output directory:

   ```bash
   cd /path/to/output/MDC/
   ```

3. **Open the project in Orcc**
   Copy the entire `MDC/` folder into an Orcc project directory. You can then use the Orcc IDE to:

   * Visualize the graphical network (via `.xdf` and `.xdfdiag` files).
   * Inspect and simulate the generated CAL actors.

---

### **Example: Accumulate-and-Shifter Operator**

The following code snippet defines two simple SDF operators, `@accumulator` and `@lshifter`, each of which performs basic arithmetic operations and emits an output. They are instantiated and interconnected in the top-level region.

```mlir
dfg.operator @accumulator inputs(%in: i32) outputs(%out: i32) {
    %0 = arith.constant 1 : i32
    %1 = arith.addi %in, %0 : i32      
    dfg.output %1: i32 
}

dfg.operator @lshifter inputs(%in: i32) outputs(%out: i32) {
    %0 = arith.constant 2 : i32
    %1 = arith.muli %in, %0 : i32   
    dfg.output %1: i32  
}

dfg.region @top inputs(%arg0: i32) outputs(%arg1: i32) {
	%0:2 = dfg.channel(1) : i32
    dfg.instantiate @accumulator inputs(%arg0) outputs(%0#0) : (i32) -> i32
    dfg.instantiate @lshifter inputs(%0#1) outputs(%arg1) : (i32) -> i32
}
```
This structure enables a **modular and composable design** and can be **automatically translated into XDF networks and CAL actors** for deployment within the Orcc toolchain. The generated files include `top.xdf` and `top.xdfdiag` (stored in the `baseline/` directory and always named consistently), as well as `accumulator.cal` and `lshifter.cal` (stored in the `custom/` directory, named after the operators defined in the MLIR).

### **Merging MLIR in verilog by MDC Backend**

We aim to **merge the MLIR design** from the example above with the following simple hardware-oriented accumulator using the Orcc environment for hardware implementation:

```mlir
dfg.operator @accumulator inputs(%in: i32) outputs(%out: i32) {
    %0 = arith.constant 1 : i32
    %1 = arith.addi %in, %0 : i32      
    dfg.output %1: i32
    
}

dfg.region @top inputs(%arg0: i32, %arg1: i32) outputs(%arg3: i32, %arg4: i32) {
    	dfg.instantiate @accumulator inputs(%arg0) outputs(%arg3) : (i32) -> i32
    	dfg.instantiate @accumulator inputs(%arg1) outputs(%arg4) : (i32) -> i32
}
```

After generating MDC files following the the above instructions (find example files [here](https://github.com/fraratto/dfg-mlir/blob/dev-myrtus/test/Merging%20MDC/MDC)), the Verilog files can be obtained by applying the **conversion and translation** flow for the FPGA backend. This is achieved using the `dfg-translate` tool with the `--for-MDC=true` flag, as shown below:

```bash
dfg-opt dfg.mlir --convert-to-vitis | dfg-translate --vitis-generate-project \
    (--output-dir=/path/you/want/ --target-device="device-name") --for-MDC=true
```

Each operator results in a corresponding Verilog file. All Verilog files should be placed in the same folder, as shown in this [example directory](https://github.com/fraratto/dfg-mlir/blob/dev-myrtus/test/Merging%20MDC/MLIR%20verilog).

In the Orcc environment, you can import:

* [Verilog files](https://github.com/fraratto/dfg-mlir/blob/dev-myrtus/test/Merging%20MDC/MLIR%20verilog),
* [MDC files](https://github.com/fraratto/dfg-mlir/blob/dev-myrtus/test/Merging%20MDC/MDC),
* and the [Vivado protocol file](https://github.com/fraratto/dfg-mlir/blob/dev-myrtus/test/Merging%20MDC/protocol/protocol_VIVADO_us.xml).

Orcc will then generate the complete set of Verilog outputs, including the top module, submodules, and testbench, which are stored in the [`Merged verilog`](https://github.com/fraratto/dfg-mlir/blob/dev-myrtus/test/Merging%20MDC/Merged%20verilog%20) directory.
 