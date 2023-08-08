dfg.operator @adder
    inputs(%in1: i32, %in2: i32)
    outputs(%out: i32)
{
    %data1 = dfg.pull %in1 : i32
    %data2 = dfg.pull %in2 : i32
    %0 = arith.addi %data1, %data2 : i32
    dfg.push(%0) %out : i32
}

dfg.operator @multiplier
    inputs(%in1: i32, %in2: i32)
    outputs(%out: i32)
{
    %data1 = dfg.pull %in1 : i32
    %data2 = dfg.pull %in2 : i32
    %0 = arith.muli %data1, %data2 : i32
    dfg.push(%0) %out : i32
}

func.func @top(%in1: i32, %in2: i32, %in3: i32) -> i32
{
    %q1_in, %q1_out = dfg.channel(i32, 4)
    %q2_in, %q2_out = dfg.channel(i32, 4)
    %q3_in, %q3_out = dfg.channel(i32, 4)
    %q4_in, %q4_out = dfg.channel(i32, 4)
    %q5_in, %q5_out = dfg.channel(i32, 4)

    dfg.push(%in1) %q1_in : i32
    dfg.push(%in2) %q2_in : i32
    dfg.push(%in3) %q3_in : i32


    dfg.instantiate @adder inputs(%q1_out, %q2_out) outputs(%q4_in) : (i32, i32) -> i32
    dfg.instantiate @multiplier inputs(%q4_out, %q3_out) outputs(%q5_in) : (i32, i32) -> i32

    %0 = dfg.pull %q5_out : i32
    return %0 : i32
}

hw.module @test() -> (o: i8)
{
    %c1_i8 = hw.constant 1 : i8
    %0 = arith.addi %c1_i8, %c1_i8 : i8
    hw.output %0 : i8
}
