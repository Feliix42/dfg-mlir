dfg.operator @test
    inputs(%in: !dfg.output<i32>)
    outputs(%out: !dfg.input<i32>)
    { looped=false }
{
    %0 = dfg.pull %in : (!dfg.output<i32>) -> (i32)
    dfg.push(%0) %out : (i32) -> (!dfg.input<i32>)
}