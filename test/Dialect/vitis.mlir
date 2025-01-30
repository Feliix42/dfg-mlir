!data_type = !vitis.ap_fixed<32,16>
!stream_type = !vitis.hls_axis<!data_type, 0, 0, 0, 1>
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
    %c0_i32 = vitis.constant 0 : i32
    %iter_arg = vitis.variable init %c0_i32 : !data_type
    vitis.while_true {
        %0 = vitis.stream.read %a : !vitis.stream<!stream_type> -> !stream_type
        %1 = vitis.stream.read %b : !vitis.stream<!stream_type> -> !stream_type

        %data_a = vitis.stream.get_data %0 : !stream_type -> !data_type
        %data_b = vitis.stream.get_data %1 : !stream_type -> !data_type
        %product = vitis.arith.mul %data_a, %data_b : !data_type
        %sum = vitis.arith.add %product, %iter_arg : !data_type
        vitis.update %iter_arg, %sum : !data_type
        
        %result = vitis.variable : !stream_type
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

vitis.func @test()
{
    %c0_f32 = vitis.constant 0.0 : f32
    %array0 = vitis.variable init %c0_f32 : !vitis.array<4 x f32>
    %array1 = vitis.variable init %c0_f32 : !vitis.array<4 x f32>
    %c0 = vitis.constant 0 : index
    %c1 = vitis.constant 1 : index
    %c4 = vitis.constant 4 : index

    vitis.for %i0 = %c0 to %c4 step %c1 {
        vitis.for %i1 = %c0 to %c4 step %c1 {
            %elem = vitis.array.read %array0[%i1] : !vitis.array<4 x f32>
            vitis.array.write %elem, %array1[%i1] : !vitis.array<4 x f32>
        }
        %elem = vitis.array.read %array0[%i0] : !vitis.array<4 x f32>
        vitis.array.write %elem, %array1[%i0] : !vitis.array<4 x f32>
    }

    vitis.return
}
