emitc.func private @source() -> (i64, i64, i64) 
                    attributes {specifiers = ["extern"]}
emitc.func private @sum(i64, i64) -> i64 
                    attributes {specifiers = ["extern"]}
emitc.func private @mul(i64, i64) -> i64
                    attributes {specifiers = ["extern"]}
emitc.func private @sink(i64) -> ()
                    attributes {specifiers = ["extern"]}

emitc.func @source_wrap(%arg0 : !emitc.ptr<!emitc.opaque<"Application">>, %arg1_packed : !emitc.opaque<"InputRTChannels<>", %arg2_packed : !emitc.opaque<"OutputRTChannels<int,int,int>">){
    %val_a = emitc.call_opaque "std::get<0>"(arg2_packed) : () -> !emitc.opaque<"IOChannel">
    %val_b = emitc.call_opaque "std::get<1>"(arg2_packed) : () -> !emitc.opaque<"IOChannel">
    %val_c = emitc.call_opaque "std::get<2>"(arg2_packed) : () -> !emitc.opaque<"IOChannel">
    %res1, %res2, %res3 = func.call @source() : () -> (i64, i64, i64)
    emitc.call_opaque "%val_a->Push" (%res1) : (i64) -> ()
    emitc.call_opaque "%val_b->Push" (%res2) : (i64) -> ()
    emitc.call_opaque "%val_c->Push" (%res3) : (i64) -> ()
}

emitc.func @sum_wrap(%arg0 : !emitc.ptr<!emitc.opaque<"Application">>, %arg1_packed : !emitc.opaque<"InputRTChannels<int,int>", %arg2_packed : !emitc.opaque<"OutputRTChannels<int>">){
    %a_in = emitc.call_opaque "std::get<0>"(arg1_packed) : () -> !emitc.opaque<"IOChannel">
    %b_in = emitc.call_opaque "std::get<1>"(arg1_packed) : () -> !emitc.opaque<"IOChannel">
    %res_out = emitc.call_opaque "std::get<0>"(arg2_packed) : () -> !emitc.opaque<"IOChannel">
    %a = emitc.call_opaque "%a_in->Pull" () : () -> i64
    %b = emitc.call_opaque "%b_in->Pull" () : () -> i64
    %res = func.call @sum(%a, %b) : (i64, i64) -> i64
    emitc.call_opaque "%res_out->Push" (%res) : (i64) -> ()
}

emitc.func @mul_wrap(%arg0 : !emitc.ptr<!emitc.opaque<"Application">>, %arg1_packed : !emitc.opaque<"InputRTChannels<int,int>", %arg2_packed : !emitc.opaque<"OutputRTChannels<int>">){
    %c_in = emitc.call_opaque "std::get<0>"(arg1_packed) : () -> !emitc.opaque<"IOChannel">
    %d_in = emitc.call_opaque "std::get<1>"(arg1_packed) : () -> !emitc.opaque<"IOChannel">
    %res_out = emitc.call_opaque "std::get<0>"(arg2_packed) : () -> !emitc.opaque<"IOChannel">
    %c = emitc.call_opaque "%c_in->Pull" () : () -> i64
    %d = emitc.call_opaque "%d_in->Pull" () : () -> i64
    %res = func.call @mul(%c, %d) : (i64, i64) -> i64
    emitc.call_opaque "%res_out->Push" (%res) : (i64) -> ()
}

emitc.func @sink_wrap(%arg0 : !emitc.ptr<!emitc.opaque<"Application">>, %arg1_packed : !emitc.opaque<"InputRTChannels<int>", %arg2_packed : !emitc.opaque<"OutputRTChannels<>">){
    %res_in = emitc.call_opaque "std::get<0>"(arg1_packed) : () -> !emitc.opaque<"IOChannel">
    %res = emitc.call_opaque "%res_in->Pull" () : () -> i64
    func.call @sink(%res) : (i64) -> ()
}

func.func @main() {
    %a_in, %a_out = dfg.channel() : i64
    %b_in, %b_out = dfg.channel() : i64
    %c_in, %c_out = dfg.channel() : i64
    %d_in, %d_out = dfg.channel() : i64
    %res_in, %res_out = dfg.channel() : i64
    dfg.instantiate @source_wrap inputs() outputs(%a_in, %b_in, %d_in) : () -> (i64, i64, i64)
    dfg.instantiate @sum_wrap inputs(%a_out, %b_out) outputs(%c_in) : (i64, i64) -> i64
    dfg.instantiate @mul_wrap inputs(%c_out, %d_out) outputs(%res_in) : (i64, i64) -> i64
    dfg.instantiate @sink_wrap inputs(%res_out) outputs() : (i64) -> ()

    emitc.call_opaque "mainRegion->AddKpnProcess" ("source_wrap", @source_wrap, {}, {%a_chan, %b_chan, %c_chan}) : (str, !emitc.ptr<!emitc.opaque<"KpnProcess">>, !emitc.opaque<"KpnProcess::Config">, !emitc.opaque<"std::tuple<IOChannel, IOChannel, IOChannel>">) -> ()

    return
}
