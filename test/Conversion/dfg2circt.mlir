dfg.operator @test
    inputs(%in: !dfg.output<ui32>)
    outputs(%out: !dfg.input<ui32>)
{
    %0 = dfg.pull %in : ui32 from !dfg.output<ui32>
    dfg.push(%0) %out : ui32 to !dfg.input<ui32>
}