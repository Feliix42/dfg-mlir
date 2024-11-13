vitis.func @mac(%a: !vitis.stream<!vitis.ap_axiu<32, 0, 0, 0, 1>>) {
    %c_true = vitis.constant true
    %iter_arg = vitis.constant 0 : i32
    vitis.return
}