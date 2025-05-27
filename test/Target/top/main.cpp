// Generated HLS code from MLIR Vitis Dialect
#include "hls_stream.h"
#include "hls_math.h"
void stream2mem_float_32(hls::stream<float> &arg0, float *arg1)
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 32; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    float data_tmp0 = arg0.read();
    arg1[idx0] = data_tmp0;
  }
}
void mem2stream_float_32(float *arg0, hls::stream<float> &arg1)
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 32; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    float elem_tmp0 = arg0[idx0];
    arg1.write(elem_tmp0);
  }
}
void fft(hls::stream<float> &arg0, hls::stream<float> &arg1, hls::stream<float> &arg2, hls::stream<float> &arg3)
{
  #pragma HLS INLINE off
  const float cst0 = 8.000000000e+00;
  const float cst1 = 4.000000000e+00;
  const float cst2 = 6.283185480e+00;
  const size_t cst3 = 8;
  const size_t cst4 = 4;
  const float cst5 = 0.0e+00;
  float array0[4][8];
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      float data_tmp0 = arg0.read();
      array0[idx0][idx1] = data_tmp0;
    }
  }
  float array1[4][8];
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      float data_tmp0 = arg1.read();
      array1[idx0][idx1] = data_tmp0;
    }
  }
  float array2[4][8];
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      array2[idx0][idx1] = cst5;
    }
  }
  float array3[4][8];
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      array3[idx0][idx1] = cst5;
    }
  }
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      for (size_t idx2 = 0; idx2 < 4; idx2 += 1) {
        for (size_t idx3 = 0; idx3 < 8; idx3 += 1) {
          #pragma HLS PIPELINE II=1 style=flp
          float elem_tmp0 = array0[idx2][idx3];
          float elem_tmp1 = array1[idx2][idx3];
          float elem_tmp2 = array2[idx0][idx1];
          float elem_tmp3 = array3[idx0][idx1];
          size_t prod_tmp0 = idx2 * idx0;
          size_t prod_tmp1 = idx3 * idx1;
          size_t rem_tmp0 = prod_tmp0 % cst4;
          size_t rem_tmp1 = prod_tmp1 % cst3;
          float cast_tmp0 = (float)rem_tmp0;
          float cast_tmp1 = (float)rem_tmp1;
          float qout_tmp0 = cast_tmp0 / cst1;
          float qout_tmp1 = cast_tmp1 / cst0;
          float sum_tmp0 = qout_tmp0 + qout_tmp1;
          float prod_tmp2 = sum_tmp0 * cst2;
          float cos_tmp0 = hls::cos(prod_tmp2);
          float sin_tmp0 = hls::sin(prod_tmp2);
          float prod_tmp3 = elem_tmp0 * cos_tmp0;
          float prod_tmp4 = elem_tmp1 * sin_tmp0;
          float sum_tmp1 = prod_tmp3 + prod_tmp4;
          float prod_tmp5 = elem_tmp1 * cos_tmp0;
          float prod_tmp6 = elem_tmp0 * sin_tmp0;
          float diff_tmp0 = prod_tmp5 - prod_tmp6;
          float sum_tmp2 = elem_tmp2 + sum_tmp1;
          float sum_tmp3 = elem_tmp3 + diff_tmp0;
          array2[idx0][idx1] = sum_tmp2;
          array3[idx0][idx1] = sum_tmp3;
        }
      }
    }
  }
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      float elem_tmp0 = array2[idx0][idx1];
      arg2.write(elem_tmp0);
    }
  }
  for (size_t idx0 = 0; idx0 < 4; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      float elem_tmp0 = array3[idx0][idx1];
      arg3.write(elem_tmp0);
    }
  }
}
void top(float *arg0, float *arg1, float *arg2, float *arg3)
{
  #pragma HLS INTERFACE mode=m_axi port=arg0 offset=slave bundle=gmem_arg0
  #pragma HLS INTERFACE mode=s_axilite port=arg0 bundle=control
  #pragma HLS INTERFACE mode=m_axi port=arg1 offset=slave bundle=gmem_arg1
  #pragma HLS INTERFACE mode=s_axilite port=arg1 bundle=control
  #pragma HLS INTERFACE mode=m_axi port=arg2 offset=slave bundle=gmem_arg2
  #pragma HLS INTERFACE mode=s_axilite port=arg2 bundle=control
  #pragma HLS INTERFACE mode=m_axi port=arg3 offset=slave bundle=gmem_arg3
  #pragma HLS INTERFACE mode=s_axilite port=arg3 bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

  hls::stream<float> stream0;
  #pragma HLS STREAM variable=stream0 depth=16
  #pragma HLS BIND_STORAGE variable=stream0 type=fifo impl=srl
  hls::stream<float> stream1;
  #pragma HLS STREAM variable=stream1 depth=16
  #pragma HLS BIND_STORAGE variable=stream1 type=fifo impl=srl
  hls::stream<float> stream2;
  #pragma HLS STREAM variable=stream2 depth=16
  #pragma HLS BIND_STORAGE variable=stream2 type=fifo impl=srl
  hls::stream<float> stream3;
  #pragma HLS STREAM variable=stream3 depth=16
  #pragma HLS BIND_STORAGE variable=stream3 type=fifo impl=srl

  #pragma HLS DATAFLOW
  mem2stream_float_32(arg0, stream0);
  mem2stream_float_32(arg1, stream1);
  fft(stream0, stream1, stream2, stream3);
  stream2mem_float_32(stream2, arg2);
  stream2mem_float_32(stream3, arg3);
}
