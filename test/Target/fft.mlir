// RUN: dfg-opt %s | FileCheck %s

!type = tensor<1x4x8xf32>

// CHECK-LABEL: fft
dfg.operator @fft inputs(%arg0: !type, %arg1 : !type)
                        outputs(%arg2: !type, %arg3: !type)
{
    %0, %1 = tosa.fft2d %arg0, %arg1 {inverse = false} : (!type, !type) -> (!type, !type)
    dfg.output %0, %1 : !type, !type
}

// CHECK-LABEL: top
dfg.region @top inputs(%arg0: !type, %arg1 : !type)
                        outputs(%arg2: !type, %arg3: !type)
{
    %0:2 = dfg.channel(16) : !type
    %1:2 = dfg.channel(16) : !type
    %2:2 = dfg.channel(16) : !type
    %3:2 = dfg.channel(16) : !type

    dfg.connect.input %arg0, %0#0 : !type
    dfg.connect.input %arg1, %1#0 : !type
    dfg.connect.output %arg2, %2#1 : !type
    dfg.connect.output %arg3, %3#1 : !type

    dfg.instantiate @fft inputs(%0#1, %1#1) outputs(%2#0, %3#0) : (!type, !type) -> (!type, !type)
}
