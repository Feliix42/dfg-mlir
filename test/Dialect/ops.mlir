dfg.operator @sum
    inputs (%op_a: ui32, %op_b: ui32)
    outputs (%a: ui32)
{
    dfg.loop inputs(%op_a: ui32) {
        %inp1 = dfg.pull %op_a : ui32
        %inp2 = dfg.pull %op_b : ui32

        // NOTE(feliix42): This would actually be an external function called here.
        %cast1 = builtin.unrealized_conversion_cast %inp1 : ui32 to i32
        %cast2 = builtin.unrealized_conversion_cast %inp2 : ui32 to i32
        %addition = arith.addi %cast1, %cast2 : i32
        %result = builtin.unrealized_conversion_cast %addition : i32 to ui32

        dfg.push(%result) %a : ui32
    }
}

func.func @return_a_value() -> ui32
{
    %0 = arith.constant 0 : i32
    %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
    return %1 : ui32
}

// dfg.operator @some_foreign_kernel
//     inputs(%foo: i32, %bar: i32)
//     outputs(%baz: i64)

dfg.operator @get_op
    inputs()
    outputs(%op_b: ui32)
{
    %b = func.call @return_a_value() : () -> ui32
    dfg.push(%b) %op_b : ui32
}

func.func @do_computations(%some_input: memref<32xi32>) -> memref<32xi32>
{
    return %some_input : memref<32xi32>
}

dfg.operator @op_with_attributes
    inputs(%something: memref<32xi32>)
    outputs(%some_result: memref<32xi32>)
    attributes { dfg.location = "src/somefile.cpp" }
{
    %some_input = dfg.pull %something : memref<32xi32>
    %res = func.call @do_computations(%some_input): (memref<32xi32>) -> memref<32xi32>
    dfg.push(%res) %some_result : memref<32xi32>
}

// Aren't the inputs and outputs arrays?
func.func @run_dfg(%op_a: ui32) -> ui32 {
    %op_a_in, %op_a_out = dfg.channel(ui32,2)
    %op_b_in, %op_b_out = dfg.channel(ui32,2)
    %res_in, %res_out = dfg.channel(ui32)

    // inputs
    dfg.push(%op_a) %op_a_in : ui32

    // connect
    dfg.instantiate @sum inputs(%op_a_out, %op_b_out) outputs(%res_in) : (ui32, ui32) -> (ui32)
    dfg.instantiate @get_op inputs() outputs(%op_b_in) : () -> (ui32)

    %res = dfg.pull %res_out : ui32

    return %res : ui32
}