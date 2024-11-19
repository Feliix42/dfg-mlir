// vitis.typedef @data_type : !vitis.ap_uint<32>
// vitis.typedef @stream_data_type : !vitis.ap_axiu<32, 0, 0, 0, 1>
// vitis.typedef @stream_type : !vitis.stream<!vitis.alias<"stream_data_type">>
!data_type = !vitis.ap_uint<32>
!stream_type = !vitis.ap_axiu<32, 0, 0, 0, 1>
vitis.func @foo(%in: !vitis.stream<!stream_type>, %out: !vitis.stream<!stream_type>)
{
    %cst_true = vitis.constant true
    vitis.while %cst_true {
        %0 = vitis.stream.read %in : !vitis.stream<!stream_type> -> !stream_type
        vitis.stream.write(%0) %out : !stream_type -> !vitis.stream<!stream_type>
        %last = vitis.stream.get_last %0 : !stream_type -> i1
        vitis.if_break %last
    }
    vitis.return
}
// vitis.func @mac(%a: !vitis.stream<!stream_type>, %b: !vitis.stream<!stream_type>, %c: !vitis.stream<!stream_type>)
// {
//     %cst_true = vitis.constant true
//     %iter_arg = vitis.constant 0 : !data_type
//     vitis.while %cst_true {
//         %0 = vitis.stream.read %a : !vitis.stream<!stream_type> -> !stream_type
//         %1 = vitis.stream.read %b : !vitis.stream<!stream_type> -> !stream_type

//         %data_a = vitis.stream.get_data %0 : !stream_type -> !data_type
//         %data_b = vitis.stream.get_data %1 : !stream_type -> !data_type
//         %product = vitis.arith.mul %data_a, %data_b : !data_type
//         %sum = vitis.arith.add %data_a, %data_b : !data_type
        
//         %result = vitis.define : !stream_type
//         vitis.stream.set_data(%sum) %result : !data_type -> !stream_type

//         %last_a = vitis.stream.get_last %0 : !stream_type -> i1
//         %last_b = vitis.stream.get_last %0 : !stream_type -> i1
//         %last = vitis.arith.or %last_a, %last_b : i1
//         vitis.stream.set_last(%last) %result : i1 -> !stream_type

//         vitis.if_break %last
//     }
//     vitis.return
// }
