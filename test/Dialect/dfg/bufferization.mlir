// RUN: dfg-opt --one-shot-bufferize --reconcile-unrealized-casts %s | dfg-opt | FileCheck %s

!type = tensor<2xi32>

// CHECK-LABEL: dfg.process @addition
// CHECK-SAME: inputs(%[[IN0:.*]] : memref<2xi32>, %[[IN1:.*]] : memref<2xi32>) outputs(%[[OUT0:.*]] : memref<2xi32>) {
// CHECK-NEXT: %[[PULL0:.*]] = dfg.pull %[[IN0]] : memref<2xi32>
// CHECK-NEXT: %[[PULL1:.*]] = dfg.pull %[[IN1]] : memref<2xi32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi32>
// CHECK-NEXT: linalg.add ins(%[[PULL0]], %[[PULL1]] : memref<2xi32>, memref<2xi32>) outs(%[[ALLOC]] : memref<2xi32>)
// CHECK-NEXT: dfg.push (%[[ALLOC]]) %[[OUT0]] : memref<2xi32>
// CHECK-NEXT: }
dfg.process @addition inputs(%a: !type, %b: !type) outputs(%c: !type)
{
    %pull0 = dfg.pull %a : !type
    %pull1 = dfg.pull %b : !type
    %0 = tensor.empty() : !type
    %1 = linalg.add ins(%pull0, %pull1 : !type, !type) outs(%0 : !type) -> !type
    dfg.push (%1) %c : !type
}