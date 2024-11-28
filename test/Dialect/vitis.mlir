!data_type = i32
!stream_type = !vitis.ap_axis<32, 0, 0, 0, 1>
vitis.func @foo(%a: !vitis.stream<!stream_type>, %b: !vitis.stream<!stream_type>)
{
    vitis.while_true {
        %0 = vitis.stream.read %a : !vitis.stream<!stream_type> -> !stream_type
        vitis.stream.write(%0) %b : !stream_type -> !vitis.stream<!stream_type>

        %last = vitis.stream.get_last %0 : !stream_type -> i1
        vitis.if_break %last
    }
    vitis.return
}
vitis.func @mac(%a: !vitis.stream<!stream_type>, %b: !vitis.stream<!stream_type>, %c: !vitis.stream<!stream_type>)
{
    %iter_arg = vitis.constant 0 : !data_type
    vitis.while_true {
        %0 = vitis.stream.read %a : !vitis.stream<!stream_type> -> !stream_type
        %1 = vitis.stream.read %b : !vitis.stream<!stream_type> -> !stream_type

        %data_a = vitis.stream.get_data %0 : !stream_type -> !data_type
        %data_b = vitis.stream.get_data %1 : !stream_type -> !data_type
        %product = vitis.arith.mul %data_a, %data_b : !data_type
        %sum = vitis.arith.add %product, %iter_arg : !data_type
        vitis.update %iter_arg, %sum : !data_type
        
        %result = vitis.define : !stream_type
        vitis.stream.set_data(%sum) %result : !data_type -> !stream_type

        %last_a = vitis.stream.get_last %0 : !stream_type -> i1
        %last_b = vitis.stream.get_last %1 : !stream_type -> i1
        %last = vitis.arith.or %last_a, %last_b : i1
        vitis.stream.set_last(%last) %result : i1 -> !stream_type

        vitis.stream.write(%result) %c : !stream_type -> !vitis.stream<!stream_type>

        vitis.if_break %last
    }
    vitis.return
}
