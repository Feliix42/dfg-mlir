#include"ap_axi_sdata.h"
#include"ap_int.h"
#include"hls_stream.h"

extern "C" {
void foo(hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v1, hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v2) {
#pragma HLS INTERFACE mode=axis port=v1
#pragma HLS INTERFACE mode=axis port=v2
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
  ap_int<1> v3 = true;
  while(v3) {
  #pragma HLS PIPELINE II=1
    ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v4 = v1.read();
    v2.write(v4);
    ap_int<1> v5 = v4.last;
    if (v5) break;
  }

}
void mac(hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v1, hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v2, hls::stream<ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v3) {
#pragma HLS INTERFACE mode=axis port=v1
#pragma HLS INTERFACE mode=axis port=v2
#pragma HLS INTERFACE mode=axis port=v3
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
  ap_int<1> v4 = true;
  ap_int<32> v5 = 0;
  while(v4) {
  #pragma HLS PIPELINE II=1
    ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v6 = v1.read();
    ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v7 = v2.read();
    ap_int<32> v8 = v6.data;
    ap_int<32> v9 = v7.data;
    ap_int<32> v10 = v8 * v9;
    ap_int<32> v11 = v10 + v5;
    v5 = v11;
    ap_axis<32, 0, 0, 0, AXIS_ENABLE_LAST> v12;

    v12.data = v11;
    ap_int<1> v13 = v6.last;
    ap_int<1> v14 = v7.last;
    ap_int<1> v15 = v13 | v14;
    v12.last = v15;
    if (v15) break;
  }

}
}
