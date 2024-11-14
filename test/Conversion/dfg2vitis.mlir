dfg.process @mac inputs (%a_in: i32, %b_in : i32) outputs (%c_out: i32) {
    %sum = arith.constant 0 : i32
    dfg.loop inputs (%a_in: i32, %b_in : i32) iter_args(%sum: i32) {
        %a = dfg.pull %a_in : i32
        %b = dfg.pull %b_in : i32
        %c = arith.muli %a, %b : i32
        %d = arith.addi %c, %sum : i32
        dfg.push(%d) %c_out : i32
        dfg.yield %d : i32
    }
}
dfg.region @mac_region inputs(%a_in: i32, %b_in : i32) outputs(%c_out: i32) {
    dfg.instantiate @mac inputs (%a_in, %b_in) outputs (%c_out) : (i32, i32) -> i32
}
dfg.operator @foo inputs(%in: i32) outputs(%out: i32)
{
    dfg.output %in : i32
}
dfg.region @foo_region inputs(%in: i32) outputs(%out: i32)
{
    %0:2 = dfg.channel(4) : i32
    dfg.instantiate @foo inputs(%in) outputs(%0#0) : (i32) -> i32
    dfg.instantiate @foo inputs(%0#1) outputs(%out) : (i32) -> i32
}
dfg.region @top inputs(%x: i32, %y: i32) outputs(%z: i32) {
    %x_sx, %x_rx = dfg.channel(4) : i32
    %y_sx, %y_rx = dfg.channel(4) : i32
    %z_sx, %z_rx = dfg.channel(4) : i32
    dfg.connect.input %x, %x_sx : i32
    dfg.connect.input %y, %y_sx : i32
    %inter:2 = dfg.channel(4) : i32
    dfg.embed @foo_region inputs(%x_rx) outputs(%inter#0) : (i32) -> i32
    dfg.embed @mac_region inputs (%inter#1, %y_rx) outputs (%z_sx) : (i32, i32) -> i32
    dfg.connect.output %z, %z_rx : i32
}
