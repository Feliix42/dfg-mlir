// RUN: dfg-opt --convert-to-vitis %s | dfg-opt | FileCheck %s

// CHECK: vitis.include "ap_int.h"
// CHECK: vitis.include "hls_stream.h"
// CHECK-LABEL: vitis.func @stream2mem_i32_12(

dfg.operator @matmul inputs(%mat0: tensor<4x5xi32>, %mat1: tensor<5x3xi32>)
                     outputs(%mat2: tensor<4x3xi32>)
{
    %0 = tensor.empty() : tensor<4x3xi32>
    %1 = linalg.matmul ins(%mat0, %mat1 : tensor<4x5xi32>, tensor<5x3xi32>)
                       outs(%0: tensor<4x3xi32>) -> tensor<4x3xi32>
    dfg.output %1 : tensor<4x3xi32>
}

dfg.region @top inputs(%in0: tensor<4x5xi32>, %in1: tensor<5x3xi32>)
                outputs(%out: tensor<4x3xi32>)
{
    %0:2 = dfg.channel(4) : tensor<4x5xi32>
    %1:2 = dfg.channel(4) : tensor<5x3xi32>
    %2:2 = dfg.channel(4) : tensor<4x3xi32>

    dfg.connect.input %in0, %0#0 : tensor<4x5xi32>
    dfg.connect.input %in1, %1#0 : tensor<5x3xi32>
    dfg.connect.output %out, %2#1 : tensor<4x3xi32>

    dfg.instantiate @matmul inputs(%0#1, %1#1) outputs(%2#0)
                            : (tensor<4x5xi32>, tensor<5x3xi32>) -> (tensor<4x3xi32>)

}
