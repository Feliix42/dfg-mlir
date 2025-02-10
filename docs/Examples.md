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
To get a working FPGA design we provide to conversion pass pipelines, which are
1. `--convert-to-vitis`
2. `--prepare-for-vivado`

The first pipeline will lower up-to-tosa operations to scf level along with bufferizations of tensor values. Then operations in `arith`, `index`, `math`, `scf` and `dfg` except for `dfg.region` will be converted to vitis equivalent operations. The result of this pipeline will be used in translations (i.e. `--vitis-to-tcl` and `vitis-to-cpp`).

The second pipeline doesn't contain any `to-vitis` passes, in which result the `dfg.region` operation will be translated later to a `tcl` script that manage the creation of FPGA design, Synthesis, Implementation and Generation of bitstreams.

Let's assume the input file is named `dfg.mlir`, to get the bitstream file one only needs to execute these commands one by one:

1. Get the HLS Cpp file from current program
```
dfg-opt dfg.mlir --convert-to-vitis | dfg-translate --vitis-to-cpp -o vitis.cpp
```
2. Get the TCL script to run High-Level Synthesis
```
dfg-opt dfg.mlir --convert-to-vitis | dfg-translate --vitis-to-tcl -o hls.tcl
```
3. Run HLS
```
/path/to/Xilinx/Vitis/2024.2/bin/vitis-run --mode hls --tcl hls.tcl
```
4. Get the TCL script to generate the full design in Vivado
```
dfg-opt dfg.mlir --prepare-for-vivado | dfg-translate --dfg-to-vivado-tcl -o vivado.tcl
```
5. Run Vivado
```
/path/to/Xilinx/Vivado/2024.2/bin/vivado -mode tcl -source vivado.tcl
```

In this work, [AMD Vivado Design Suite](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html) version 2024.2 is tested without any problems. If one need to use lower version of the tools, please modify the version of Xilinx IPs in [this](../lib/Target/VivadoTcl/TranslateToVivadoTcl.cpp) translation respectively.

