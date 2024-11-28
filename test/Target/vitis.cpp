#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

extern "C" {
void foo(
    hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v1,
    hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v2)
{
#pragma HLS INTERFACE mode = axis port = v1
#pragma HLS INTERFACE mode = axis port = v2
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
    while (true) {
#pragma HLS PIPELINE II = 1
        ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v3 = v1.read();
        v2.write(v3);
        ap_int<1> v4 = v3.last;
        if (v4) break;
    }
}
void mac(
    hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v1,
    hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v2,
    hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v3)
{
#pragma HLS INTERFACE mode = axis port = v1
#pragma HLS INTERFACE mode = axis port = v2
#pragma HLS INTERFACE mode = axis port = v3
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
    ap_int<32> v4 = 0;
    while (true) {
#pragma HLS PIPELINE II = 1
        ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v5 = v1.read();
        ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v6 = v2.read();
        ap_int<32> v7 = v5.data;
        ap_int<32> v8 = v6.data;
        ap_int<32> v9 = v7 * v8;
        ap_int<32> v10 = v9 + v4;
        v4 = v10;
        ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v11;

        v11.data = v10;
        ap_int<1> v12 = v5.last;
        ap_int<1> v13 = v6.last;
        ap_int<1> v14 = v12 | v13;
        v11.last = v14;
        v3.write(v11);
        if (v14) break;
    }
}
}
