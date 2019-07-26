/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include "Vtiny_dnn_top.h"
#include "systemc.h"

extern Vtiny_dnn_top* verilator_top;

extern sc_signal <bool>      backprop;
extern sc_signal <bool>      enbias;
extern sc_signal <bool>      run;
extern sc_signal <bool>      wwrite;
extern sc_signal <bool>      bwrite;

extern sc_signal <sc_uint<4> >  vdd;
extern sc_signal <sc_uint<12> > vss;
extern sc_signal <sc_uint<4> >  vid;
extern sc_signal <sc_uint<10> > vis;
extern sc_signal <sc_uint<5> >  vih;
extern sc_signal <sc_uint<5> >  viw;
extern sc_signal <sc_uint<12> > vds;
extern sc_signal <sc_uint<4> >  vod;
extern sc_signal <sc_uint<10> > vos;
extern sc_signal <sc_uint<5> >  voh;
extern sc_signal <sc_uint<5> >  vow;
extern sc_signal <sc_uint<10> > vfs;
extern sc_signal <sc_uint<10> > vks;
extern sc_signal <sc_uint<5> >  vkh;
extern sc_signal <sc_uint<5> >  vkw;

extern sc_signal <bool>         s_init;
extern sc_signal <bool>         s_fin;
extern sc_signal <bool>         k_init;
extern sc_signal <bool>         k_fin;
extern sc_signal <bool>         exec;
extern sc_signal <sc_uint<12> > ia;
extern sc_signal <sc_uint<10> > wa;

extern void eval();

#pragma once

#include <chrono>    // for high_resolution_clock, NOLINT

union{
  float i;
  float f;
} conv16d;

union{
  float i;
  float f;
} convd;

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
    dsc = main_time;

    //if(0){
    verilator_top->dd = 0;
    verilator_top->ss = ss-1;
    verilator_top->id = 1-1; ////////////
    verilator_top->is = iw*ih;
    verilator_top->ih = ih-1;
    verilator_top->iw = iw-1;
    verilator_top->ds = ds-1;
    verilator_top->od = od-1;
    verilator_top->os = ow*oh;
    verilator_top->oh = oh-1;
    verilator_top->ow = ow-1;
    verilator_top->fs = kw*kh*1-1; /////////
    verilator_top->ks = kw*kh-1;
    verilator_top->kh = kh-1;
    verilator_top->kw = kw-1;

    verilator_top->backprop = 0;
    verilator_top->deltaw = 0;
    verilator_top->enbias = 1;
    verilator_top->wwrite = 0;
    verilator_top->bwrite = 0;
    verilator_top->run = 0;
    verilator_top->last = 0;
    verilator_top->src_valid = 0;
    verilator_top->dst_ready = 1;

    eval();

    verilator_top->dwconv = 1;
    verilator_top->wwrite = 1;

    eval();
    verilator_top->src_valid = 1;
    for(size_t o=0;o<(od+3)/4;o++){
      for(size_t i=0;i<kh*kw;i++){
        if((o*4+0)<od){
          conv16d.f = W[(o*4+0)*kh*kw+i];
          verilator_top->src_data0 = conv16d.i;
        }
        if((o*4+1)<od){
          conv16d.f = W[(o*4+1)*kh*kw+i];
          verilator_top->src_data1 = conv16d.i;
        }
        if((o*4+2)<od){
          conv16d.f = W[(o*4+2)*kh*kw+i];
          verilator_top->src_data2 = conv16d.i;
        }
        if((o*4+3)<od){
          conv16d.f = W[(o*4+3)*kh*kw+i];
          verilator_top->src_data3 = conv16d.i;
        }
        eval();
      }
    }
    verilator_top->src_valid = 0;
    eval();
    verilator_top->wwrite = 0;
    eval();

    verilator_top->bwrite = 1;
    eval();
    verilator_top->src_valid = 1;
    for (size_t o = 0; o < od; o++) {
      if (params.has_bias) {
        conv16d.f = bias[o];
      }else{
        conv16d.f = 0;
      }
      switch(o%4){
      case 0 : verilator_top->src_data0 = conv16d.i;break;
      case 1 : verilator_top->src_data1 = conv16d.i;break;
      case 2 : verilator_top->src_data2 = conv16d.i;break;
      case 3 : verilator_top->src_data3 = conv16d.i;eval();break;
      }
    }
    if(od%4){
      eval();
    }
    verilator_top->src_valid = 0;
    eval();
    verilator_top->bwrite = 0;
    bwrite = 0;
    eval();

    verilator_top->fs = iw*ih*1-1; /////////

    verilator_top->run = 1;
    verilator_top->wwrite = 1;
    eval();
    verilator_top->src_valid = 1;
    for(size_t o=0;o<(od+3)/4;o++){
      for(size_t i=0;i<iw*ih;){
        if(verilator_top->src_ready){
          if((o*4+0)<od){
            conv16d.f = in_data[0][(o*4+0)*ih*iw+i];
            verilator_top->src_data0 = conv16d.i;
          }
          if((o*4+1)<od){
            conv16d.f = in_data[0][(o*4+1)*ih*iw+i];
            verilator_top->src_data1 = conv16d.i;
          }
          if((o*4+2)<od){
            conv16d.f = in_data[0][(o*4+2)*ih*iw+i];
            verilator_top->src_data2 = conv16d.i;
          }
          if((o*4+3)<od){
            conv16d.f = in_data[0][(o*4+3)*ih*iw+i];
            verilator_top->src_data3 = conv16d.i;
          }
          i++;
          eval();
        }else{
          eval();
        }
      }
    }
    verilator_top->src_valid = 0;
    eval();

    for (size_t sample = 0; sample < in_data.size(); sample++) {
      const vec_t &in = in_data[sample+1];
      vec_t &a        = out_data[sample-1];

      if(sample!=0){
        while(!verilator_top->dst_valid) {
          eval();
        }
        for(size_t outa = 0; outa < ow*oh*od; ){
          switch(outa%2){
          case 0: convd.i = verilator_top->dst_data0;break;
          case 1: convd.i = verilator_top->dst_data1;break;
          }
          a[outa] = convd.f;
          if((outa%2)==1){
            eval();
          }
          outa++;
        }
        if((ow*oh*od)%2){
          eval();
        }
      }

      for(int i = 0; i < 10; i++){
        eval();
      }

      if(sample+1<in_data.size()){
        verilator_top->src_valid = 1;
        for(size_t o=0;o<(od+3)/4;o++){
          for(size_t i=0;i<iw*ih;){
            if(verilator_top->src_ready){
              if((o*4+0)<od){
                conv16d.f = in[(o*4+0)*ih*iw+i];
                verilator_top->src_data0 = conv16d.i;
              }
              if((o*4+1)<od){
                conv16d.f = in[(o*4+1)*ih*iw+i];
                verilator_top->src_data1 = conv16d.i;
              }
              if((o*4+2)<od){
                conv16d.f = in[(o*4+2)*ih*iw+i];
                verilator_top->src_data2 = conv16d.i;
              }
              if((o*4+3)<od){
                conv16d.f = in[(o*4+3)*ih*iw+i];
                verilator_top->src_data3 = conv16d.i;
              }
              i++;
              eval();
            }else{
              eval();
            }
          }
        }
        verilator_top->src_valid = 0;
        eval();
      }else{
        verilator_top->last = 1;
      }
    }

    while(!verilator_top->dst_valid) {
      eval();
    }
    for(size_t outa = 0; outa < ow*oh*od; ){
      switch(outa%2){
      case 0: convd.i = verilator_top->dst_data0;break;
      case 1: convd.i = verilator_top->dst_data1;break;
      }
      out_data[in_data.size()-1][outa] = convd.f;
      if((outa%2)==1){
        eval();
      }
      outa++;
    }
    if((ow*oh*od)%2){
      eval();
    }

    verilator_top->enbias = 0;
    verilator_top->wwrite = 0;
    verilator_top->dwconv = 0;
    verilator_top->last = 0;
    verilator_top->run = 0;

    eval();

    dft += std::chrono::high_resolution_clock::now() - dst;
    dfc += main_time - dsc;

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
