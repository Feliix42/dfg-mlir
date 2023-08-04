dfg.operator @test
    inputs(%in: i32)
    outputs(%out: i32)
{
    %0 = dfg.pull %in : i32
    dfg.push(%0) %out : i32
}