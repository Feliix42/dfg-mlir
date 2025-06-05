// RUN: dfg-opt %s | FileCheck %s

// CHECK-LABEL: emitHLS.func @dummy
// CHECK-SAME: (%[[ARG0:.*]]: !emitHLS.stream<i32>, %[[ARG1:.*]]: !emitHLS.stream<i32>) {
emitHLS.func @dummy(%stream_in: !emitHLS.stream<i32>, %stream_out: !emitHLS.stream<i32>)
{
    // CHECK-NEXT: %[[ELEM0:.*]] = emitHLS.stream.read %[[ARG0]] : !emitHLS.stream<i32> -> i32
    %elem0 = emitHLS.stream.read %stream_in : !emitHLS.stream<i32> -> i32
    // CHECK-NEXT: emitHLS.stream.write(%[[ELEM0]]) %[[ARG1]] : i32 -> !emitHLS.stream<i32>
    emitHLS.stream.write (%elem0) %stream_out : i32 -> !emitHLS.stream<i32>
}

// CHECK-LABEL: emitHLS.func @loop
emitHLS.func @loop()
{
    // CHECK-NEXT: %[[CONST0:.*]] = emitHLS.variable as const i32 = 8
    %const0 = emitHLS.variable as const i32 = 8
    // CHECK-NEXT: %[[ARRAY0:.*]] = emitHLS.variable as !emitHLS.array<2xi32>
    %array0 = emitHLS.variable as !emitHLS.array<2xi32>
    // CHECK-NEXT: emitHLS.for %[[IDX0:.*]] = 0 to 2 step 1 {
    emitHLS.for %idx = 0 to 2 step 1 {
        // CHECK-NEXT: emitHLS.pragma.pipeline II=1 style=flp
        emitHLS.pragma.pipeline II=1 style=flp
        // CHECK-NEXT: emitHLS.array.write %[[CONST0]], %[[ARRAY0]][%[[IDX0]]] : i32 -> !emitHLS.array<2xi32>
        emitHLS.array.write %const0, %array0[%idx] : i32 -> !emitHLS.array<2xi32>
    }
}

// CHECK-LABEL: emitHLS.func @top
// CHECK-SAME: (%[[ARG0:.*]]: !emitHLS.ptr<i32>, %[[ARG1:.*]]: !emitHLS.ptr<f32>) {
emitHLS.func @top(%arg0: !emitHLS.ptr<i32>, %arg1: !emitHLS.ptr<f32>)
{
    // CHECK-NEXT: emitHLS.pragma.interface mode=m_axi port=%[[ARG0]](!emitHLS.ptr<i32>) offset=slave bundle=gmem_arg0
    emitHLS.pragma.interface mode=m_axi port=%arg0(!emitHLS.ptr<i32>) offset=slave bundle=gmem_arg0
    // CHECK-NEXT: emitHLS.pragma.interface mode=s_axilite port=%[[ARG0]](!emitHLS.ptr<i32>) bundle=control
    emitHLS.pragma.interface mode=s_axilite port=%arg0(!emitHLS.ptr<i32>) bundle=control
    // CHECK-NEXT: emitHLS.pragma.interface mode=m_axi port=%[[ARG1]](!emitHLS.ptr<f32>) offset=slave bundle=gmem_arg1
    emitHLS.pragma.interface mode=m_axi port=%arg1(!emitHLS.ptr<f32>) offset=slave bundle=gmem_arg1
    // CHECK-NEXT: emitHLS.pragma.interface mode=s_axilite port=%[[ARG1]](!emitHLS.ptr<f32>) bundle=control
    emitHLS.pragma.interface mode=s_axilite port=%arg1(!emitHLS.ptr<f32>) bundle=control
}
