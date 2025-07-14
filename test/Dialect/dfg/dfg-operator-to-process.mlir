// RUN: dfg-opt --dfg-operator-to-process %s | dfg-opt | FileCheck %s

// CHECK-LABEL: dfg.process @addition
// CHECK-SAME: inputs(%[[IN0:.*]] : i32, %[[IN1:.*]] : i32) outputs(%[[OUT0:.*]] : i32) {
// CHECK-NEXT: dfg.loop inputs (%[[IN0]] : i32, %[[IN1]] : i32) outputs (%[[OUT0]] : i32) {
// CHECK-NEXT: %[[PULL0:.*]] = dfg.pull %[[IN0]] : i32
// CHECK-NEXT: %[[PULL1:.*]] = dfg.pull %[[IN1]] : i32
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[PULL0]], %[[PULL1]] : i32
// CHECK-NEXT: dfg.push (%[[ADD]]) %[[OUT0]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: }
dfg.operator @addition inputs(%a: i32, %b: i32) outputs(%c: i32)
{
    %0 = arith.addi %a, %b : i32
    dfg.output %0 : i32
}

// CHECK-LABEL: dfg.process @tensor_addition
// CHECK-SAME: inputs(%[[IN0:.*]] : tensor<2xi32>, %[[IN1:.*]] : tensor<2xi32>) outputs(%[[OUT0:.*]] : tensor<2xi32>) {
// CHECK-NEXT: dfg.loop inputs (%[[IN0]] : tensor<2xi32>, %[[IN1]] : tensor<2xi32>) outputs (%[[OUT0]] : tensor<2xi32>) {
// CHECK-NEXT: %[[PULL0:.*]] = dfg.pull %[[IN0]] : tensor<2xi32>
// CHECK-NEXT: %[[PULL1:.*]] = dfg.pull %[[IN1]] : tensor<2xi32>
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[PULL0]], %[[PULL1]] : tensor<2xi32>
// CHECK-NEXT: dfg.push (%[[ADD]]) %[[OUT0]] : tensor<2xi32>
// CHECK-NEXT: }
// CHECK-NEXT: }
dfg.operator @tensor_addition inputs(%a: tensor<2xi32>, %b: tensor<2xi32>) outputs(%c: tensor<2xi32>)
{
    %0 = arith.addi %a, %b : tensor<2xi32>
    dfg.output %0 : tensor<2xi32>
}
