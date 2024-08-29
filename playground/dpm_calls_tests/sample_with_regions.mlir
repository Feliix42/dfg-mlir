func.func private @source() -> (i64, i64, i64)
func.func private @mul(i64, i64) -> i64
func.func private @sink(i64) -> ()

dfg.operator @multiply inputs(%a : i64, %b : i64) outputs(%c : i64) {
    %res = arith.muli %a, %b : i64
    dfg.output %res : i64
}

dfg.process @source_wrap inputs () outputs (%val_a: i64, %val_b: i64, %val_c: i64) {
    %res1, %res2, %res3 = func.call @source() : () -> (i64, i64, i64)
    dfg.push(%res1) %val_a : i64
    dfg.push(%res2) %val_b : i64
    dfg.push(%res3) %val_c : i64
}

dfg.process @sum inputs (%a_in: i64, %b_in: i64) outputs (%res_out: i64) {
    %a = dfg.pull %a_in : i64
    %b = dfg.pull %b_in : i64
    %res = arith.addi %a, %b : i64
    dfg.push(%res) %res_out : i64
}

dfg.process @sink_wrap inputs (%res_in: i64) outputs () {
    %res = dfg.pull %res_in : i64
    func.call @sink(%res) : (i64) -> ()
}

dfg.region @multiplyRegion inputs(%x_in : i64, %y_in : i64) outputs(%res_out : i64) {
    dfg.instantiate @multiply inputs(%x_in, %y_in) outputs(%res_out) : (i64, i64) -> i64
}

dfg.region @mainRegion inputs() outputs() {
    %a_in, %a_out = dfg.channel() : i64
    %b_in, %b_out = dfg.channel() : i64
    %c_in, %c_out = dfg.channel() : i64
    %d_in, %d_out = dfg.channel() : i64
    %res_in, %res_out = dfg.channel() : i64

    dfg.instantiate @source_wrap inputs() outputs(%a_in, %b_in, %d_in) : () -> (i64, i64, i64)
    dfg.instantiate @sum inputs(%a_out, %b_out) outputs(%c_in) : (i64, i64) -> i64

    dfg.embed @multiplyRegion inputs(%c_out, %d_out) outputs(%res_in) : (i64, i64) -> i64

    dfg.instantiate @sink_wrap inputs(%res_out) outputs() : (i64) -> ()
}
