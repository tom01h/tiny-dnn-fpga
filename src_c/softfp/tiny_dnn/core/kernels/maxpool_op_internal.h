/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <limits>
#include <vector>

#include <chrono>    // for high_resolution_clock, NOLINT

union{
  float i;
  float f;
} conv16p;

namespace tiny_dnn {
namespace kernels {

inline void maxpool_op_internal(const tensor_t &in_data,
                                tensor_t &out_data,
                                core::maxpool_params &params,
                                const bool layer_parallelize) {
  pst = std::chrono::high_resolution_clock::now();
  psc = main_time;

  if((params.pool_size_x==2)&&(params.pool_size_y==2)&&
     (params.stride_x   ==2)&&(params.stride_y   ==2)&&
     (in_data.size() != 1)){
     //     (in_data.size() != 1)&&0){

    size_t ow          = params.out.width_;
    size_t oh          = params.out.height_;
    size_t od          = params.out.depth_;

    eval();
    verilator_top->od =  0;
    verilator_top->ow =  ow;
    verilator_top->os =  ow*oh*od;      //even number
    verilator_top->ds = (ow*oh*od)/2-1;
    verilator_top->pool = 1;
    verilator_top->dst_ready = 1;
    eval();
    eval();

    for(size_t sample = 0; sample < in_data.size(); sample++){
      const vec_t &in          = in_data[sample];
      vec_t &out               = out_data[sample];
      std::vector<size_t> &max = params.out2inmax[sample];

      verilator_top->src_valid = 1;
      for(size_t c = 0; c < od; c++){
        for(size_t y = 0; y < oh; y++){
          for(size_t x = 0; x < ow; x++){
            float_t max_value;
            size_t idx;
            size_t idx0 = c*oh*ow*4 + (y*2+0)*ow*2 + x*2;
            size_t idx1 = c*oh*ow*4 + (y*2+1)*ow*2 + x*2;
            conv16p.f = in[idx0+0];
            verilator_top->src_data0 = conv16p.i;
            conv16p.f = in[idx1+0];
            verilator_top->src_data1 = conv16p.i;
            conv16p.f = in[idx0+1];
            verilator_top->src_data2 = conv16p.i;
            conv16p.f = in[idx1+1];
            verilator_top->src_data3 = conv16p.i;
            /*
            if(in[idx1+0]>in[idx0+0]){
              max_value = in[idx1+0];
              idx = idx1+0;
            }else{
              max_value = in[idx0];
              idx = idx0;
            }
            if(in[idx0+1]>max_value){
              max_value = in[idx0+1];
              idx = idx0+1;
            }
            if(in[idx1+1]>max_value){
              max_value = in[idx1+1];
              idx = idx1+1;
            }
            max[c*oh*ow + y*ow + x] = idx;
            out[c*oh*ow + y*ow + x] = max_value;
            */
            eval();
          }
        }
      }
      verilator_top->src_valid = 0;
      eval();
      while(!verilator_top->dst_valid){
        eval();
      }
      for(int i=0; verilator_top->dst_valid;){
        conv16p.i = verilator_top->dst_ptr0;
        max[i] = conv16p.f;
        out[i++] = verilator_top->dst_data0;

        conv16p.i = verilator_top->dst_ptr1;
        max[i] = conv16p.f;
        out[i++] = verilator_top->dst_data1;

        eval();
      }
    }
    verilator_top->pool = 0;
    verilator_top->dst_ready = 0;
    eval();
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
  pfc += main_time - psc;
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
