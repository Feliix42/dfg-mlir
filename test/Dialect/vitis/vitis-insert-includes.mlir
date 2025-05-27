// RUN: dfg-opt --vitis-insert-includes %s | FileCheck %s

// CHECK: vitis.include "ap_int.h"
// CHECK-NEXT: vitis.include "ap_fixed.h"
// CHECK-NEXT: vitis.include "hls_stream.h"
// CHECK-NEXT: vitis.include "hls_math.h"

vitis.func @foo()
{ 
    %stream0 = vitis.variable as !vitis.stream<i32>

    %const0 = vitis.variable as f32 = 1.0
    %sin = vitis.math.sin %const0 : f32

    %fixed = vitis.variable as !vitis.ap_fixed<2,2>
}
