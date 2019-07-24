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

#include <limits>
#include <vector>

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
} conv16p;

union{
  int i;
  float f;
} convp;

namespace tiny_dnn {
namespace kernels {

inline void maxpool_op_internal(const tensor_t &in_data,
                                tensor_t &out_data,
                                core::maxpool_params &params,
                                const bool layer_parallelize) {
  pst = std::chrono::high_resolution_clock::now();

  if((params.pool_size_x==2)&&(params.pool_size_y==2)&&
     (params.stride_x   ==2)&&(params.stride_y   ==2)&&
     (in_data.size() > 1)  ){

    size_t ow          = params.out.width_;
    size_t oh          = params.out.height_;
    size_t od          = params.out.depth_;

    dnn_addr[10] = (ow*oh*od)/2-1; //ds
    dnn_addr[11] = 0;              //od
    dnn_addr[12] = ow*oh*od;       //os
    dnn_addr[13] = oh;             //oh
    dnn_addr[14] = ow;             //ow

    dnn_addr[0] = 0;   // init
    dnn_addr[0] = 128; // pool

    for(size_t sample = 0; sample < in_data.size(); sample++){
      const vec_t &in          = in_data[sample];
      vec_t &out               = out_data[sample];
      std::vector<size_t> &max = params.out2inmax[sample];

      size_t idxs = 0;
      size_t idxd = 0;
      for(size_t cy = 0; cy < od*oh*2; cy++){
        for(size_t x = 0; x < ow*2; x++){
          conv16p.f = in[idxs+x];
          src_addr[idxd+x*2] = conv16p.i;
        }
        idxs += ow*2;
        if(cy%2){
          idxd += ow*2*2-1;
        }else{
          idxd ++;
        }
      }
      __asm__("DSB 15");

      // AXI DMA transfer tx rx
      dma_reset();
      dma_addr[0x30/4] = 1;
      dma_addr[0x48/4] = dst_phys;
      dma_addr[0x58/4] = ow*oh*od*4;
      dma_addr[0x00/4] = 1;
      dma_addr[0x18/4] = src_phys;
      dma_addr[0x28/4] = ow*2*oh*2*od*2;

      // Wait for the tx to finish
      while ((dma_addr[0x04/4] & 0x1000)!=0x1000);

      // Wait for the rx to finish
      while ((dma_addr[0x34/4] & 0x1000)!=0x1000) ;

      __asm__("DSB 15");
      for(int i=0; i<ow*oh*od;){
        convp.f = dst_addr[i];
        max[i]  = convp.i & 0x0000ffff;
        convp.i = convp.i & 0xffff0000;
        out[i] = convp.f;
        i++;

        convp.f = dst_addr[i];
        max[i]  = convp.i & 0x0000ffff;
        convp.i = convp.i & 0xffff0000;
        out[i] = convp.f;
        i++;
      }
    }

    dnn_addr[0] = 0;   // idle

  }else{
    for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
        const vec_t &in          = in_data[sample];
        vec_t &out               = out_data[sample];
        std::vector<size_t> &max = params.out2inmax[sample];

        for (size_t i = 0; i < params.out2in.size(); i++) {
          const auto &in_index = params.out2in[i];
          float_t max_value    = std::numeric_limits<float_t>::lowest();
          size_t idx           = 0;
          for (auto j : in_index) {
            if (in[j] > max_value) {
              max_value = in[j];
              idx       = j;
            }
          }
          max[i] = idx;
          out[i] = max_value;
        }
      });
  }
  pft += std::chrono::high_resolution_clock::now() - pst;
}

inline void maxpool_grad_op_internal(tensor_t &prev_delta,
                                     const tensor_t &curr_delta,
                                     std::vector<std::vector<size_t>> &max_idx,
                                     const std::vector<size_t> &in2out,
                                     const bool layer_parallelize) {
  pst = std::chrono::high_resolution_clock::now();
  for_i(layer_parallelize, prev_delta.size(), [&](size_t sample) {
    vec_t &prev                    = prev_delta[sample];
    const vec_t &curr              = curr_delta[sample];
    const std::vector<size_t> &max = max_idx[sample];

    prev.assign(prev.size(), float_t{0});
    for (size_t i = 0; i < max.size(); i++) {
      prev[max[i]] = curr[i];
    }
  });
  pbt += std::chrono::high_resolution_clock::now() - pst;
}

}  // namespace kernels
}  // namespace tiny_dnn
