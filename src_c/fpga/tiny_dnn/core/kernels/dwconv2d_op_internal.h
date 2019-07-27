/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

extern volatile int *dnn_addr;
extern volatile int *dma_addr;
extern volatile int16_t *src_addr;
extern volatile float *dst_addr;
extern unsigned long src_phys;
extern unsigned long dst_phys;

#pragma once

#include <chrono>    // for high_resolution_clock, NOLINT

//void dma_reset(){
//  dma_addr[0x30/4] = 4;
//  dma_addr[0x00/4] = 4;
//  while (dma_addr[0x00/4] & 0x4);
//}

union{
  struct{
    int16_t j;
    int16_t i;
  };
  float f;
} conv16d;

namespace tiny_dnn {
namespace kernels {

inline void dwconv2d_op_internal(const tensor_t &in_data,
                                 const vec_t &W,
                                 const vec_t &bias,
                                 tensor_t &out_data,
                                 const core::dwconv_params &params,
                                 const bool parallelize) {

  size_t out_area    = params.out.area();
  size_t iw          = params.in_padded.width_;
  size_t ih          = params.in_padded.height_;
  size_t id          = params.in.depth_;
  size_t ow          = params.out.width_;
  size_t oh          = params.out.height_;
  size_t od          = params.in.depth_;
  size_t kw          = params.weight.width_;
  size_t kh          = params.weight.height_;
  size_t w_dilation  = params.w_dilation;
  size_t h_dilation  = params.h_dilation;
  size_t elem_stride = params.w_stride;
  size_t line_stride = iw * params.h_stride;

  size_t ss          = (iw*ih*id+3)/4;
  size_t ds          = (ow*oh*od+1)/2;

  if(in_data.size()>1){
    dst = std::chrono::high_resolution_clock::now();

    dnn_addr[ 5] = ss-1;       //ss
    dnn_addr[ 6] = 1-1;        //id//////
    dnn_addr[ 7] = iw*ih;      //is
    dnn_addr[ 8] = ih-1;       //ih
    dnn_addr[ 9] = iw-1;       //iw

    dnn_addr[10] = ds-1;       //ds
    dnn_addr[11] = od-1;       //od
    dnn_addr[12] = ow*oh;      //os
    dnn_addr[13] = oh-1;       //oh
    dnn_addr[14] = ow-1;       //ow

    dnn_addr[15] = 0;          //dd

    dnn_addr[ 1] = kw*kh*1-1;  //fs//////
    dnn_addr[ 2] = kw*kh-1;    //ks
    dnn_addr[ 3] = kh-1;       //kh
    dnn_addr[ 4] = kw-1;       //kw

    /////////////////////////////////////////////////////////
    // Weight transfer

    // DMA Buffer
    for(size_t o=0;o<od;o++){
      size_t ptr  =  o   *kh*kw;
      size_t ptr_ = (o/4)*kh*kw*4 + (o%4);
      for(size_t i=0;i<kh*kw;i++){
        conv16d.f = W[ptr+i];
        src_addr[ptr_+i*4] = conv16d.i;
      }
    }
    __asm__("DSB 15");

    dnn_addr[0] = 0; // init
    dnn_addr[0] = 2|256; // wwrite|dwconv

    // AXI DMA transfer tx
    dma_reset();
    dma_addr[0x00/4] = 1;
    dma_addr[0x18/4] = src_phys;
    dma_addr[0x28/4] = ((od+3)/4)*4*kh*kw*2;

    while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish

    /////////////////////////////////////////////////////////
    // Bias transfer
    // DMA Buffer
    for (size_t o = 0; o < od; o++) {
      if (params.has_bias) {
        conv16d.f = bias[o];
        src_addr[o] = conv16d.i;
      }else{
        src_addr[o] = 0;
      }
    }
    __asm__("DSB 15");

    dnn_addr[0] = 0; // init
    dnn_addr[0] = 1|256; // bwrite|dwconv

    // AXI DMA transfer tx
    dma_reset();
    dma_addr[0x00/4] = 1;
    dma_addr[0x18/4] = src_phys;
    dma_addr[0x28/4] = od*2;

    while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish

    /////////////////////////////////////////////////////////
    // Run
    dnn_addr[ 1] = iw*ih*1-1;  //fs//////
    dnn_addr[0] = 0;   // init
    dnn_addr[0] = 2|4|8|256; // wwrite|run|enbias|dwconv


    // DMA Buffer
    for(size_t o=0;o<id;o++){
      size_t ptr  =  o   *ih*iw;
      size_t ptr_ = (o/4)*ih*iw*4 + (o%4);
      for(size_t i=0;i<ih*iw;i++){
        conv16d.f = in_data[0][ptr+i];
        src_addr[ptr_+i*4] = conv16d.i;
      }
    }
    __asm__("DSB 15");

    // AXI DMA transfer tx
    dma_reset();
    dma_addr[0x00/4] = 1;
    dma_addr[0x18/4] = src_phys;
    dma_addr[0x28/4] = ((id+3)/4)*4*ih*iw*2;

    // Wait for the tx to finish
    while ((dma_addr[0x04/4] & 0x1000)!=0x1000);

    for (size_t sample = 0; sample < in_data.size(); sample++) {

      if(sample!=0){
        __asm__("DSB 15");
        for(size_t i=0;i<od*oh*ow;i++){
          out_data[sample-1][i] = dst_addr[i];
        }
      }

      // AXI DMA transfer rx
      dma_reset();
      dma_addr[0x30/4] = 1;
      dma_addr[0x48/4] = dst_phys;
      dma_addr[0x58/4] = ow*oh*od*4;

      if(sample+1<in_data.size()){
        for(size_t o=0;o<id;o++){
          size_t ptr  =  o   *ih*iw;
          size_t ptr_ = (o/4)*ih*iw*4 + (o%4);
          for(size_t i=0;i<ih*iw;i++){
            conv16d.f = in_data[sample+1][ptr+i];
            src_addr[ptr_+i*4] = conv16d.i;
          }
        }
        __asm__("DSB 15");

        // AXI DMA transfer tx
        dma_addr[0x00/4] = 1;
        dma_addr[0x18/4] = src_phys;
        dma_addr[0x28/4] = ((id+3)/4)*4*ih*iw*2;

        // Wait for the tx to finish
        while ((dma_addr[0x04/4] & 0x1000)!=0x1000);
      }else{
        dnn_addr[0] = 2|4|8|64|256; // wwrite|run|enbias|last|dwconv
      }

      // Wait for the rx to finish
      while ((dma_addr[0x34/4] & 0x1000)!=0x1000) ;
      dma_reset();

    }

    __asm__("DSB 15");
    for(size_t i=0;i<od*oh*ow;i++){
      out_data[in_data.size()-1][i] = dst_addr[i];
    }

    dnn_addr[0] = 0; // idle

    dft += std::chrono::high_resolution_clock::now() - dst;

  }else{
    for (size_t sample = 0; sample < in_data.size(); sample++) {
      const vec_t &in = in_data[sample];
      vec_t &a        = out_data[sample];
      for (size_t o = 0; o < od; o++) {
        size_t inc = o;

        float_t *pa = &a[params.out.get_index(0, 0, o)];
        size_t idx;
        idx                = params.weight.get_index(0, 0, inc);
        const float_t *pw  = &W[idx];
        idx                = params.in_padded.get_index(0, 0, inc);
        const float_t *pin = &in[idx];
        float_t *pout      = pa;
        for (size_t y = 0; y < oh; y++) {
          const float_t *pin_line = pin;
          for (size_t x = 0; x < ow; x++) {
            const float_t *pin_element = pin_line;
            const float_t *pw_element  = pw;
            float_t sum{0};
            // should be optimized for small kernel(3x3,5x5)
            for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
              for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                sum += pw_element[wx] * pin_element[wx * w_dilation];
              }
              pw_element += kw;
              pin_element += iw * h_dilation;
            }
            pout[x] += sum;
            pin_line += elem_stride;
          }
          pout += ow;
          pin += line_stride;
        }
        if (params.has_bias) {
          vectorize::add(bias[o], out_area, pa);
        }
      }
    }
  }
}

/******************************************************************/

template <typename tensor_t, typename vec_t>
void dwconv2d_op_internal(const tensor_t &prev_out,
                          const vec_t &W,
                          tensor_t &dW,
                          tensor_t &db,
                          tensor_t &curr_delta,
                          tensor_t &prev_delta,
                          const core::dwconv_params &params,
                          const bool parallelize) {
  typedef typename vec_t::value_type float_t;

  dst = std::chrono::high_resolution_clock::now();

  for_i(parallelize, prev_out.size(), [&](size_t sample) {
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      size_t outc = inc;

      size_t idx        = 0;
      idx               = params.weight.get_index(0, 0, inc);
      const float_t *pw = &W[idx];

      idx                       = params.out.get_index(0, 0, outc);
      const float_t *pdelta_src = &curr_delta[sample][idx];

      idx = params.in_padded.get_index(0, 0, inc);
      // float_t* pdelta_dst = &(*prev_delta)[sample][idx];
      float_t *pdelta_dst = &prev_delta[sample][idx];

      for (size_t y = 0; y < params.out.height_; y++) {
        for (size_t x = 0; x < params.out.width_; x++) {
          const float_t *ppw = pw;

          idx                       = y * params.out.width_ + x;
          const float_t ppdelta_src = pdelta_src[idx];

          float_t *ppdelta_dst =
            pdelta_dst + y * params.h_stride * params.in_padded.width_ +
            x * params.w_stride;

          for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
            for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
              idx = wy * params.in_padded.width_ + wx;
              ppdelta_dst[idx] += *ppw++ * ppdelta_src;
            }
          }
        }
      }
    }
  });

  dbt += std::chrono::high_resolution_clock::now() - dst;

  dst = std::chrono::high_resolution_clock::now();
  for_i(parallelize, prev_out.size(), [&](size_t sample) {
    // accumulate dw
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      size_t outc = inc;

      for (size_t wy = 0; wy < params.weight.height_; wy++) {
        for (size_t wx = 0; wx < params.weight.width_; wx++) {
          float_t dst{0};

          size_t idx           = 0;
          idx                  = params.in_padded.get_index(wx, wy, inc);
          const float_t *prevo = &prev_out[sample][idx];

          idx                  = params.out.get_index(0, 0, outc);
          const float_t *delta = &curr_delta[sample][idx];

          if (params.w_stride > 1) {
            for (size_t y = 0; y < params.out.height_; y++) {
              size_t prevo_idx =
                y * params.in_padded.width_ * params.h_stride;
              size_t delta_idx = y * params.out.width_;

              for (size_t x = 0; x < params.out.width_; x++) {
                dst += prevo[prevo_idx + x * params.w_stride] *
                  delta[delta_idx + x];
              }
            }
          } else {
            for (size_t y = 0; y < params.out.height_; y++) {
              dst += vectorize::dot(
                                    prevo + y * params.in_padded.width_ * params.h_stride,
                                    delta + y * params.out.width_, params.out.width_);
            }
          }

          idx = inc;
          if(sample==0){
            dW[0][params.weight.get_index(wx, wy, idx)] = dst;
          }else{
            dW[0][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const float_t *delta  = &curr_delta[sample][idx];
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        if(sample==0){
          db[0][outc] = std::accumulate(delta, deltaa, float_t{0});
        }else{
          db[0][outc] += std::accumulate(delta, deltaa, float_t{0});
        }
      }
    }
  });
  ddt += std::chrono::high_resolution_clock::now() - dst;
}

}  // namespace kernels
}  // namespace tiny_dnn
