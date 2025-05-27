// RUN: dfg-opt %s | FileCheck %s

// CHECK-LABEL: vitis.func @dummy
// CHECK-SAME: (%[[ARG0:.*]]: !vitis.stream<i32>, %[[ARG1:.*]]: !vitis.stream<i32>) {
vitis.func @dummy(%stream_in: !vitis.stream<i32>, %stream_out: !vitis.stream<i32>)
{
    // CHECK-NEXT: %[[ELEM0:.*]] = vitis.stream.read %[[ARG0]] : !vitis.stream<i32> -> i32
    %elem0 = vitis.stream.read %stream_in : !vitis.stream<i32> -> i32
    // CHECK-NEXT: vitis.stream.write(%[[ELEM0]]) %[[ARG1]] : i32 -> !vitis.stream<i32>
    vitis.stream.write (%elem0) %stream_out : i32 -> !vitis.stream<i32>
}

// CHECK-LABEL: vitis.func @loop
vitis.func @loop()
{
    // CHECK-NEXT: %[[CONST0:.*]] = vitis.variable as const i32 = 8
    %const0 = vitis.variable as const i32 = 8
    // CHECK-NEXT: %[[ARRAY0:.*]] = vitis.variable as !vitis.array<2xi32>
    %array0 = vitis.variable as !vitis.array<2xi32>
    // CHECK-NEXT: vitis.for %[[IDX0:.*]] = 0 to 2 step 1 {
    vitis.for %idx = 0 to 2 step 1 {
        // CHECK-NEXT: vitis.pragma.pipeline II=1 style=flp
        vitis.pragma.pipeline II=1 style=flp
        // CHECK-NEXT: vitis.array.write %[[CONST0]], %[[ARRAY0]][%[[IDX0]]] : i32 -> !vitis.array<2xi32>
        vitis.array.write %const0, %array0[%idx] : i32 -> !vitis.array<2xi32>
    }
}

// CHECK-LABEL: vitis.func @top
// CHECK-SAME: (%[[ARG0:.*]]: !vitis.ptr<i32>, %[[ARG1:.*]]: !vitis.ptr<f32>) {
vitis.func @top(%arg0: !vitis.ptr<i32>, %arg1: !vitis.ptr<f32>)
{
    // CHECK-NEXT: vitis.pragma.interface mode=m_axi port=%[[ARG0]](!vitis.ptr<i32>) offset=slave bundle=gmem_arg0
    vitis.pragma.interface mode=m_axi port=%arg0(!vitis.ptr<i32>) offset=slave bundle=gmem_arg0
    // CHECK-NEXT: vitis.pragma.interface mode=s_axilite port=%[[ARG0]](!vitis.ptr<i32>) bundle=control
    vitis.pragma.interface mode=s_axilite port=%arg0(!vitis.ptr<i32>) bundle=control
    // CHECK-NEXT: vitis.pragma.interface mode=m_axi port=%[[ARG1]](!vitis.ptr<f32>) offset=slave bundle=gmem_arg1
    vitis.pragma.interface mode=m_axi port=%arg1(!vitis.ptr<f32>) offset=slave bundle=gmem_arg1
    // CHECK-NEXT: vitis.pragma.interface mode=s_axilite port=%[[ARG1]](!vitis.ptr<f32>) bundle=control
    vitis.pragma.interface mode=s_axilite port=%arg1(!vitis.ptr<f32>) bundle=control
}
