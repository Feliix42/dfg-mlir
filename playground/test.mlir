llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
llvm.mlir.global internal constant @str0("num recvd: %d\0A\00")

func.func private @sum(i64, i64) -> i64
func.func private @mul(i64, i64) -> i64

dfg.operator @sum_wrap
    inputs (%a_in: i64, %b_in: i64)
    outputs (%res_out: i64)
{
    %a = dfg.pull %a_in : i64
    %b = dfg.pull %b_in : i64

    %res = func.call @sum(%a, %b) : (i64, i64) -> i64

    dfg.push(%res) %res_out : i64
}

dfg.operator @mul_wrap
    inputs (%c_in: i64, %d_in: i64)
    outputs (%res_out: i64)
{
    %c = dfg.pull %c_in : i64
    %d = dfg.pull %d_in : i64

    %res = func.call @mul(%c, %d) : (i64, i64) -> i64

    dfg.push(%res) %res_out : i64
}

func.func @algo() {
    // inputs
    %a = arith.constant 2 : i64
    %b = arith.constant 3 : i64
    %d = arith.constant 4 : i64

    %a_in, %a_out = dfg.channel() : i64
    %b_in, %b_out = dfg.channel() : i64
    %c_in, %c_out = dfg.channel() : i64
    %d_in, %d_out = dfg.channel() : i64
    %res_in, %res_out = dfg.channel() : i64

    // initial inputs
    dfg.push(%a) %a_in : i64
    dfg.push(%b) %b_in : i64
    dfg.push(%d) %d_in : i64

    dfg.instantiate @sum_wrap inputs(%a_out, %b_out) outputs(%c_in) : (i64, i64) -> i64
    dfg.instantiate @mul_wrap inputs(%c_out, %d_out) outputs(%res_in) : (i64, i64) -> i64

    %recvd = dfg.pull %res_out : i64

    %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<15 x i8>>
    %4 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.getelementptr %5[%4, %4] : (!llvm.ptr<array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %21 = llvm.call @printf(%6, %recvd) : (!llvm.ptr<i8>, i64) -> i32

    return
}
