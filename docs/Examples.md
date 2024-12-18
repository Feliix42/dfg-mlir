# Examples

## Typical MAC operator example (FPGA backend)
Consider the given code in MLIR:

```
// dfg_operator.mlir
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

This code snippet defines an SDF node named `mac` and instantiates it in the top region and connected with FIFO channels. To get a working FPGA design first one need to convert all `operator` operations to `process` using

```
dfg-opt dfg_operator.mlir --dfg-operator-to-process -o dfg.mlir
```

This transformation pass will convert `operator` to the corresponding KPN `process` including the iteration arguments. Once you reach there, there are several things to do in order to get the hardware design from Vivado/Vitis.

1. Convert process operations to vitis functions
```
dfg-opt dfg.mlir --convert-dfg-to-vitis -o vitis.mlir
```
2. Translate vitis functions to C++ file
```
dfg-translate vitis.mlir --vitis-to-cpp -o vitis.cpp
```
3. Generate the vitis_hls script to run HLS
```
dfg-translate vitis.mlir --vitis-to-tcl -o run_hls.tcl
```
4. Generate the vivado script to generate the hardware
```
dfg-translate dfg.mlir --dfg-to-vivado-tcl -o run_vivado.tcl
```

The generated Tcl scripts can be direcly used with Xilinx executables with the mode tcl. For this example, you should get a `.xsa` file in the directory `/root-to-dfg_opeartor.mlir/vivado_project/top`, which contains the hardware (hwh, bit, etc...).
