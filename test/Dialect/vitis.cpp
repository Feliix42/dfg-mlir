#include"ap_axi_sdata.h"
#include"ap_int.h"
#include"hls_stream.h"

extern "C" {
void foo(hls::stream<ap_axiu<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v1, hls::stream<ap_axiu<32, 0, 0, 0, AXIS_ENABLE_LAST>> &v2) {
#pragma HLS INTERFACE mode=axis port=v1
#pragma HLS INTERFACE mode=axis port=v2
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
  bool v3 = true;
  while(v3) {
  #pragma HLS PIPELINE II=1
    ap_axiu<32, 0, 0, 0, AXIS_ENABLE_LAST> v4 = v1.read();
    v2.write(v4);
    bool v4 = v5.last;
    if (v4) break;
  }

}
}
