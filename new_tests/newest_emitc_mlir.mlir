module {
  func.func private @source() -> (i64, i64, i64)
  func.func private @sum(i64, i64) -> i64
  func.func private @mul(i64, i64) -> i64
  func.func private @sink(i64)
  emitc.func @source_wrap(%arg0: !emitc.ptr<!emitc.opaque<"Application">>, %arg1: !emitc.opaque<"InputRTChannels<>">, %arg2: !emitc.opaque<"OutputRTChannels<>">) {
    %0 = emitc.call_opaque "std::get"(%arg2) {template_args = [0 : i32]} : (!emitc.opaque<"OutputRTChannels<>">) -> !emitc.opaque<"Channel">
    %1 = emitc.call_opaque "std::get"(%arg2) {template_args = [1 : i32]} : (!emitc.opaque<"OutputRTChannels<>">) -> !emitc.opaque<"Channel">
    %2 = emitc.call_opaque "std::get"(%arg2) {template_args = [2 : i32]} : (!emitc.opaque<"OutputRTChannels<>">) -> !emitc.opaque<"Channel">
    cf.br ^bb1(%0, %1, %2 : !emitc.opaque<"Channel">, !emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
  ^bb1(%3: !emitc.opaque<"Channel">, %4: !emitc.opaque<"Channel">, %5: !emitc.opaque<"Channel">):  // pred: ^bb0
    %6:3 = func.call @source() : () -> (i64, i64, i64)
    emitc.call_opaque "v3->Push"(%6#0) : (i64) -> ()
    emitc.call_opaque "v4->Push"(%6#1) : (i64) -> ()
    emitc.call_opaque "v5->Push"(%6#2) : (i64) -> ()
    emitc.return
  }
  emitc.func @sum_wrap(%arg0: !emitc.ptr<!emitc.opaque<"Application">>, %arg1: !emitc.opaque<"InputRTChannels<>">, %arg2: !emitc.opaque<"OutputRTChannels<>">) {
    %0 = emitc.call_opaque "std::get"(%arg1) {template_args = [0 : i32]} : (!emitc.opaque<"InputRTChannels<>">) -> !emitc.opaque<"Channel">
    %1 = emitc.call_opaque "std::get"(%arg1) {template_args = [1 : i32]} : (!emitc.opaque<"InputRTChannels<>">) -> !emitc.opaque<"Channel">
    %2 = emitc.call_opaque "std::get"(%arg2) {template_args = [0 : i32]} : (!emitc.opaque<"OutputRTChannels<>">) -> !emitc.opaque<"Channel">
    cf.br ^bb1(%0, %1, %2 : !emitc.opaque<"Channel">, !emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
  ^bb1(%3: !emitc.opaque<"Channel">, %4: !emitc.opaque<"Channel">, %5: !emitc.opaque<"Channel">):  // pred: ^bb0
    %6 = emitc.call_opaque "Pop"(%3) : (!emitc.opaque<"Channel">) -> i64
    %7 = emitc.call_opaque "Pop"(%4) : (!emitc.opaque<"Channel">) -> i64
    %8 = func.call @sum(%6, %7) : (i64, i64) -> i64
    emitc.call_opaque "v5->Push"(%8) : (i64) -> ()
    emitc.return
  }
  emitc.func @mul_wrap(%arg0: !emitc.ptr<!emitc.opaque<"Application">>, %arg1: !emitc.opaque<"InputRTChannels<>">, %arg2: !emitc.opaque<"OutputRTChannels<>">) {
    %0 = emitc.call_opaque "std::get"(%arg1) {template_args = [0 : i32]} : (!emitc.opaque<"InputRTChannels<>">) -> !emitc.opaque<"Channel">
    %1 = emitc.call_opaque "std::get"(%arg1) {template_args = [1 : i32]} : (!emitc.opaque<"InputRTChannels<>">) -> !emitc.opaque<"Channel">
    %2 = emitc.call_opaque "std::get"(%arg2) {template_args = [0 : i32]} : (!emitc.opaque<"OutputRTChannels<>">) -> !emitc.opaque<"Channel">
    cf.br ^bb1(%0, %1, %2 : !emitc.opaque<"Channel">, !emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
  ^bb1(%3: !emitc.opaque<"Channel">, %4: !emitc.opaque<"Channel">, %5: !emitc.opaque<"Channel">):  // pred: ^bb0
    %6 = emitc.call_opaque "Pop"(%3) : (!emitc.opaque<"Channel">) -> i64
    %7 = emitc.call_opaque "Pop"(%4) : (!emitc.opaque<"Channel">) -> i64
    %8 = func.call @mul(%6, %7) : (i64, i64) -> i64
    emitc.call_opaque "v5->Push"(%8) : (i64) -> ()
    emitc.return
  }
  emitc.func @sink_wrap(%arg0: !emitc.ptr<!emitc.opaque<"Application">>, %arg1: !emitc.opaque<"InputRTChannels<>">, %arg2: !emitc.opaque<"OutputRTChannels<>">) {
    %0 = emitc.call_opaque "std::get"(%arg1) {template_args = [0 : i32]} : (!emitc.opaque<"InputRTChannels<>">) -> !emitc.opaque<"Channel">
    cf.br ^bb1(%0 : !emitc.opaque<"Channel">)
  ^bb1(%1: !emitc.opaque<"Channel">):  // pred: ^bb0
    %2 = emitc.call_opaque "Pop"(%1) : (!emitc.opaque<"Channel">) -> i64
    func.call @sink(%2) : (i64) -> ()
    emitc.return
  }
  func.func @main() {
    %0:2 = emitc.call_opaque "mainRegion->AddChannel"() {template_args = [i64]} : () -> (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
    %1:2 = emitc.call_opaque "mainRegion->AddChannel"() {template_args = [i64]} : () -> (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
    %2:2 = emitc.call_opaque "mainRegion->AddChannel"() {template_args = [i64]} : () -> (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
    %3:2 = emitc.call_opaque "mainRegion->AddChannel"() {template_args = [i64]} : () -> (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
    %4:2 = emitc.call_opaque "mainRegion->AddChannel"() {template_args = [i64]} : () -> (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">)
    emitc.call_opaque "mainRegion->AddKpnProcess"(%0#0, %1#0, %3#0) : (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">, !emitc.opaque<"Channel">) -> ()
    emitc.call_opaque "mainRegion->AddKpnProcess"(%0#1, %1#1, %2#0) : (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">, !emitc.opaque<"Channel">) -> ()
    emitc.call_opaque "mainRegion->AddKpnProcess"(%2#1, %3#1, %4#0) : (!emitc.opaque<"Channel">, !emitc.opaque<"Channel">, !emitc.opaque<"Channel">) -> ()
    emitc.call_opaque "mainRegion->AddKpnProcess"(%4#1) : (!emitc.opaque<"Channel">) -> ()
    return
  }
}

