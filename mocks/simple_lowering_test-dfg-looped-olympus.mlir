llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
llvm.mlir.global internal constant @str0("num recvd: %d\0A\00")


dfg.operator @produce_value
    outputs (%val: i32)
{
    dfg.loop outputs(%val: i32) {
        %val1 = arith.constant 323729 : i32

        dfg.push(%val1) %val : i32
    }
}

dfg.operator @consume
    inputs (%number: i32)
{
    %recvd = dfg.pull %number : i32

    %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<15 x i8>>
    %4 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.getelementptr %5[%4, %4] : (!llvm.ptr<array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %21 = llvm.call @printf(%6, %recvd) : (!llvm.ptr<i8>, i32) -> i32
}

dfg.operator @kernel_test
    inputs (%num1: i32)
    outputs (%num2: i32)
    attributes { evp.path = "../../something.cpp", multiplicity = array<i64: 1, 1> }

func.func @algo() {
    %op_in, %op_out = dfg.channel() : i32
    %a_in, %a_out = dfg.channel() : i32

    dfg.instantiate @produce_value outputs(%op_in) : () -> i32
    dfg.instantiate offloaded @kernel_test inputs(%op_out) outputs(%a_in) : (i32) -> i32
    dfg.instantiate @consume inputs(%a_out) : (i32) -> ()

    return
}
