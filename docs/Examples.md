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

