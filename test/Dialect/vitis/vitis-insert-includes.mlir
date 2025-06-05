// RUN: dfg-opt --emitHLS-insert-includes %s | FileCheck %s

// CHECK: emitHLS.include "ap_int.h"
// CHECK-NEXT: emitHLS.include "ap_fixed.h"
// CHECK-NEXT: emitHLS.include "hls_stream.h"
// CHECK-NEXT: emitHLS.include "hls_math.h"

emitHLS.func @foo()
{ 
    %stream0 = emitHLS.variable as !emitHLS.stream<i32>

    %const0 = emitHLS.variable as f32 = 1.0
    %sin = emitHLS.math.sin %const0 : f32

    %fixed = emitHLS.variable as !emitHLS.ap_fixed<2,2>
}
