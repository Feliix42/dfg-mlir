dfg.operator @sum
    inputs (%op_a: !dfg.output<ui32>, %op_b: !dfg.output<ui32>)
    outputs (%a: !dfg.input<ui32>)
    { looped = true }
{
    %inp1 = dfg.pull %op_a : (!dfg.output<ui32>) -> (ui32)
    %inp2 = dfg.pull %op_b : (!dfg.output<ui32>) -> (ui32)

    // NOTE(feliix42): This would actually be an external function called here.
    %cast1 = builtin.unrealized_conversion_cast %inp1 : ui32 to i32
    %cast2 = builtin.unrealized_conversion_cast %inp2 : ui32 to i32
    %addition = arith.addi %cast1, %cast2 : i32
    %result = builtin.unrealized_conversion_cast %addition : i32 to ui32

    dfg.push(%result) %a : (ui32) -> (!dfg.input<ui32>)

    // in Ops.td add NoTerminator to solve the terminator error
}

func.func @return_a_value() -> ui32
{
    %0 = arith.constant 0 : i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
    return %1 : ui32
}

dfg.operator @get_op
    inputs()
    outputs(%op_b: !dfg.input<ui32>)
    { looped = false }
{
    %b = func.call @return_a_value() : () -> ui32
    dfg.push(%b) %op_b : (ui32) -> (!dfg.input<ui32>)
}

// Aren't the inputs and outputs arrays?
func.func @run_dfg(%op_a: ui32) -> ui32 {
    %op_a_in, %op_a_out = dfg.channel<ui32,2>
    %op_b_in, %op_b_out = dfg.channel<ui32,2>
    %res_in, %res_out = dfg.channel<ui32>

    // inputs
    dfg.push(%op_a) %op_a_in : (ui32) -> (!dfg.input<ui32>)

    // connect
    dfg.instantiate @sum inputs(%op_a_out, %op_b_out) outputs(%res_in) : (!dfg.output<ui32>, !dfg.output<ui32>) -> (!dfg.input<ui32>)
    dfg.instantiate @get_op inputs() outputs(%op_b_in) : () -> (!dfg.input<ui32>)
    // What about automatically verify the type of the inputs and outputs ports
    // and drop the signature here. Need to add verify function in the op. But
    // in the print part we print everything, just a prettification idea.

    %res = dfg.pull %res_out : (!dfg.output<ui32>) -> (ui32)

    return %res : ui32
}