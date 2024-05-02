module {
  func.func private @source() -> (i64, i64, i64)
  func.func private @sum(i64, i64) -> i64
  func.func private @mul(i64, i64) -> i64
  func.func private @sink(i64)
  emitc.func @source_wrap(%arg0: !dfg.input<i64>, %arg1: !dfg.input<i64>, %arg2: !dfg.input<i64>) {
    %0:3 = func.call @source() : () -> (i64, i64, i64)
    emitc.call_opaque "channelName->Push"(%0#0) : (i64) -> ()
    emitc.call_opaque "channelName->Push"(%0#1) : (i64) -> ()
    emitc.call_opaque "channelName->Push"(%0#2) : (i64) -> ()
    emitc.return
  }
  emitc.func @sum_wrap(%arg0: !dfg.output<i64>, %arg1: !dfg.output<i64>, %arg2: !dfg.input<i64>) {
    %0 = emitc.call_opaque "channelName->Pop"() : () -> i64
    %1 = emitc.call_opaque "channelName->Pop"() : () -> i64
    %2 = func.call @sum(%0, %1) : (i64, i64) -> i64
    emitc.call_opaque "channelName->Push"(%2) : (i64) -> ()
    emitc.return
  }
  emitc.func @mul_wrap(%arg0: !dfg.output<i64>, %arg1: !dfg.output<i64>, %arg2: !dfg.input<i64>) {
    %0 = emitc.call_opaque "channelName->Pop"() : () -> i64
    %1 = emitc.call_opaque "channelName->Pop"() : () -> i64
    %2 = func.call @mul(%0, %1) : (i64, i64) -> i64
    emitc.call_opaque "channelName->Push"(%2) : (i64) -> ()
    emitc.return
  }
  emitc.func @sink_wrap(%arg0: !dfg.output<i64>) {
    %0 = emitc.call_opaque "channelName->Pop"() : () -> i64
    func.call @sink(%0) : (i64) -> ()
    emitc.return
  }
  func.func @main() {
    %0:2 = emitc.call_opaque "mainRegion->AddChannel"() : () -> (!dfg.input<i64>, !dfg.output<i64>)
    %1:2 = emitc.call_opaque "mainRegion->AddChannel"() : () -> (!dfg.input<i64>, !dfg.output<i64>)
    %2:2 = emitc.call_opaque "mainRegion->AddChannel"() : () -> (!dfg.input<i64>, !dfg.output<i64>)
    %3:2 = emitc.call_opaque "mainRegion->AddChannel"() : () -> (!dfg.input<i64>, !dfg.output<i64>)
    %4:2 = emitc.call_opaque "mainRegion->AddChannel"() : () -> (!dfg.input<i64>, !dfg.output<i64>)
    emitc.call_opaque "mainRegion->AddKpnProcess"() {args = ["hi", #emitc.opaque<"0">], template_args = []} : () -> ()
    emitc.call_opaque "mainRegion->AddKpnProcess"() {args = ["hi", #emitc.opaque<"0">], template_args = []} : () -> ()
    emitc.call_opaque "mainRegion->AddKpnProcess"() {args = ["hi", #emitc.opaque<"0">], template_args = []} : () -> ()
    emitc.call_opaque "mainRegion->AddKpnProcess"() {args = ["hi", #emitc.opaque<"0">], template_args = []} : () -> ()
    return
  }
}

