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
} conv16;

union{
  float i;
  float f;
} conv;

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_internal(const tensor_t &in_data,
                               const vec_t &W,
                               const vec_t &bias,
                               tensor_t &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {
  cst = std::chrono::high_resolution_clock::now();
  //  csc = main_time;

  size_t out_area    = params.out.area();
  size_t iw          = params.in_padded.width_;
  size_t ih          = params.in_padded.height_;
  size_t id          = params.in.depth_;
  size_t ow          = params.out.width_;
  size_t oh          = params.out.height_;
  size_t od          = params.out.depth_;
  size_t kw          = params.weight.width_;
  size_t kh          = params.weight.height_;
  size_t elem_stride = params.w_stride;
  size_t line_stride = iw * params.h_stride;
  size_t ss          = (iw*ih*id+3)/4;
  size_t ds          = (ow*oh*od+1)/2;

  // NOT supported parametor
  // params.tbl.is_connected
  // params.w_stride
  // params.h_stride
  if(in_data.size()>1){
    //if(0){
    verilator_top->dd = 0;
    verilator_top->ss = ss-1;
    verilator_top->id = id-1;
    verilator_top->is = iw*ih;
    verilator_top->ih = ih-1;
    verilator_top->iw = iw-1;
    verilator_top->ds = ds-1;
    verilator_top->od = od-1;
    verilator_top->os = ow*oh;
    verilator_top->oh = oh-1;
    verilator_top->ow = ow-1;
    verilator_top->fs = kw*kh*id-1;
    verilator_top->ks = kw*kh-1;
    verilator_top->kh = kh-1;
    verilator_top->kw = kw-1;

    vdd = 0;
    vss = ss-1;
    vid = id-1;
    vis = iw*ih;
    vih = ih-1;
    viw = iw-1;
    vds = ds-1;
    vod = od-1;
    vos = ow*oh;
    voh = oh-1;
    vow = ow-1;
    vfs = kw*kh*id-1;
    vks = kw*kh-1;
    vkh = kh-1;
    vkw = kw-1;

    verilator_top->backprop = 0;
    verilator_top->deltaw = 0;
    verilator_top->enbias = 1;
    verilator_top->wwrite = 0;
    verilator_top->bwrite = 0;
    verilator_top->run = 0;
    verilator_top->last = 0;
    verilator_top->src_valid = 0;
    verilator_top->dst_ready = 1;

    backprop = 0;
    enbias = 1;
    wwrite = 0;
    bwrite = 0;
    run = 0;

    eval();

    verilator_top->wwrite = 1;
    wwrite = 1;
    eval();
    verilator_top->src_valid = 1;
    for(size_t o=0;o<(od+3)/4;o++){
      for(size_t i=0;i<id*kh*kw;i++){
        if((o*4+0)<od){
          conv16.f = W[(o*4+0)*id*kh*kw+i];
          verilator_top->src_data0 = conv16.i;
        }
        if((o*4+1)<od){
          conv16.f = W[(o*4+1)*id*kh*kw+i];
          verilator_top->src_data1 = conv16.i;
        }
        if((o*4+2)<od){
          conv16.f = W[(o*4+2)*id*kh*kw+i];
          verilator_top->src_data2 = conv16.i;
        }
        if((o*4+3)<od){
          conv16.f = W[(o*4+3)*id*kh*kw+i];
          verilator_top->src_data3 = conv16.i;
        }
        eval();
      }
    }
    verilator_top->src_valid = 0;
    eval();
    verilator_top->wwrite = 0;
    wwrite = 0;
    eval();

    verilator_top->bwrite = 1;
    bwrite = 1;
    eval();
    verilator_top->src_valid = 1;
    for (size_t o = 0; o < od; o++) {
      if (params.has_bias) {
        conv16.f = bias[o];
      }else{
        conv16.f = 0;
      }
      switch(o%4){
      case 0 : verilator_top->src_data0 = conv16.i;break;
      case 1 : verilator_top->src_data1 = conv16.i;break;
      case 2 : verilator_top->src_data2 = conv16.i;break;
      case 3 : verilator_top->src_data3 = conv16.i;eval();break;
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

    verilator_top->run = 1;
    run = 1;
    eval();
    verilator_top->src_valid = 1;

    size_t ina, outa;

    for(ina = 0; ina < iw*ih*id; ){
      if(verilator_top->src_ready){
        conv16.f = in_data[0][ina];
        switch(ina%4){
        case 0 : verilator_top->src_data0 = conv16.i;break;
        case 1 : verilator_top->src_data1 = conv16.i;break;
        case 2 : verilator_top->src_data2 = conv16.i;break;
        case 3 : verilator_top->src_data3 = conv16.i;eval();break;
        }
        ina++;
      }else{
        eval();
      }
    }
    if((iw*ih*id)%4){
      eval();
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
        for(outa = 0; outa < ow*oh*od; ){
          switch(outa%2){
          case 0: conv.i = verilator_top->dst_data0;break;
          case 1: conv.i = verilator_top->dst_data1;break;
          }
          a[outa] = conv.f;
          if((outa%2)==1){
            eval();
          }
          outa++;
        }
      }
      if((ow*oh*od)%2){
        eval();
      }

      for(int i = 0; i < 10; i++){
        eval();
      }

      if(sample+1<in_data.size()){
        verilator_top->src_valid = 1;
        for(ina = 0; ina < iw*ih*id; ){
          if(verilator_top->src_ready){
            conv16.f = in[ina];
            switch(ina%4){
            case 0 : verilator_top->src_data0 = conv16.i;break;
            case 1 : verilator_top->src_data1 = conv16.i;break;
            case 2 : verilator_top->src_data2 = conv16.i;break;
            case 3 : verilator_top->src_data3 = conv16.i;eval();break;
            }
            ina++;
          }else{
            eval();
          }
        }
        if((iw*ih*id)%4){
          eval();
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
    for(outa = 0; outa < ow*oh*od; ){
      switch(outa%2){
      case 0: conv.i = verilator_top->dst_data0;break;
      case 1: conv.i = verilator_top->dst_data1;break;
      }
      out_data[in_data.size()-1][outa] = conv.f;
      if((outa%2)==1){
        eval();
      }
      outa++;
    }
    if((ow*oh*od)%2){
      eval();
    }

    verilator_top->enbias = 0;
    verilator_top->run = 0;
    verilator_top->last = 0;

    enbias = 0;
    run = 0;
    eval();
  }else{
    for (size_t sample = 0; sample < in_data.size(); sample++) {
      const vec_t &in = in_data[sample];
      vec_t &a        = out_data[sample];
      for (size_t o = 0; o < od; o++) {
        float_t *pa = &a[params.out.get_index(0, 0, o)];
        for (size_t inc = 0; inc < id; inc++) {
          if (!params.tbl.is_connected(o, inc)) continue;
          size_t idx;
          idx                = params.weight.get_index(0, 0, id * o + inc);
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
                  sum += pw_element[wx] * pin_element[wx];
                }
                pw_element += kw;
                pin_element += iw;
              }
              pout[x] += sum;
              pin_line += elem_stride;
            }
            pout += ow;
            pin += line_stride;
          }
        }
        if (params.has_bias) {
          vectorize::add(bias[o], out_area, pa);
        }
      }
    }
  }

  cft += std::chrono::high_resolution_clock::now() - cst;
  //  cfc += main_time - csc;
}

/******************************************************************/

template <typename tensor_t, typename vec_t>
void conv2d_op_internal(const tensor_t &prev_out,
                        const vec_t &W,
                        tensor_t &dW,
                        tensor_t &db,
                        tensor_t &curr_delta,
                        tensor_t &prev_delta,
                        const core::conv_params &params,
                        const bool parallelize) {
  typedef typename vec_t::value_type float_t;

  size_t iw          = params.in_padded.width_;
  size_t ih          = params.in_padded.height_;
  size_t id          = params.in.depth_;
  size_t ow          = params.out.width_;
  size_t oh          = params.out.height_;
  size_t od          = params.out.depth_;
  size_t kw          = params.weight.width_;
  size_t kh          = params.weight.height_;
  size_t ss          = (ow*oh*od+3)/4;
  size_t ds          = (iw*ih*id+1)/2;

  cst = std::chrono::high_resolution_clock::now();
  //  csc = main_time;

  size_t ina, outa;

  if(id!=1){ // because input delta NOT USED

  verilator_top->dd = 0;
  verilator_top->ss = ss-1;
  verilator_top->id = od-1;
  verilator_top->is = ow*oh;
  verilator_top->ih = oh-1;
  verilator_top->iw = ow-1;
  verilator_top->ds = ds-1;
  verilator_top->od = id-1;
  verilator_top->os = iw*ih;
  verilator_top->oh = ih-1;
  verilator_top->ow = iw-1;
  verilator_top->fs = kw*kh*od-1;
  verilator_top->ks = kw*kh-1;
  verilator_top->kh = kh-1;
  verilator_top->kw = kw-1;

  vdd = 0;
  vss = ss-1;
  vid = od-1;
  vis = ow*oh;
  vih = oh-1;
  viw = ow-1;
  vds = ds-1;
  vod = id-1;
  vos = iw*ih;
  voh = ih-1;
  vow = iw-1;
  vfs = kw*kh*od-1;
  vks = kw*kh-1;
  vkh = kh-1;
  vkw = kw-1;

  verilator_top->backprop = 1;
  verilator_top->deltaw = 0;
  verilator_top->enbias = 0;
  verilator_top->wwrite = 0;
  verilator_top->bwrite = 0;
  verilator_top->run = 0;
  verilator_top->last = 0;
  verilator_top->src_valid = 0;
  verilator_top->dst_ready = 1;

  backprop = 1;
  enbias = 0;
  wwrite = 0;
  bwrite = 0;
  run = 0;
  eval();

  verilator_top->wwrite = 1;
  wwrite = 1;
  eval();
  verilator_top->src_valid = 1;
  for(size_t ii=0;ii<od;ii++){        //od-1=veri->id
    for(size_t o=0;o<(id+3)/4;o++){   //id-1=veri->od
      for(size_t i=0;i<kh*kw;i++){
        if((o*4+0)<id){
          conv16.f = W[(o*4+0)*kh*kw+i+ii*kh*kw*id];
          verilator_top->src_data0 = conv16.i;
        }
        if((o*4+1)<id){
          conv16.f = W[(o*4+1)*kh*kw+i+ii*kh*kw*id];
          verilator_top->src_data1 = conv16.i;
        }
        if((o*4+2)<id){
          conv16.f = W[(o*4+2)*kh*kw+i+ii*kh*kw*id];
          verilator_top->src_data2 = conv16.i;
        }
        if((o*4+3)<id){
          conv16.f = W[(o*4+3)*kh*kw+i+ii*kh*kw*id];
          verilator_top->src_data3 = conv16.i;
        }
        eval();
      }
    }
  }
  verilator_top->src_valid = 0;
  eval();
  verilator_top->wwrite = 0;
  wwrite = 0;
  eval();

  verilator_top->run = 1;
  run = 1;
  eval();
  verilator_top->src_valid = 1;

  for(ina = 0; ina < ow*oh*od; ){
    if(verilator_top->src_ready){
      conv16.f = curr_delta[0][ina];
      switch(ina%4){
      case 0 : verilator_top->src_data0 = conv16.i;break;
      case 1 : verilator_top->src_data1 = conv16.i;break;
      case 2 : verilator_top->src_data2 = conv16.i;break;
      case 3 : verilator_top->src_data3 = conv16.i;eval();break;
      }
      ina++;
    }else{
      eval();
    }
  }
  if((ow*oh*od)%4){
    eval();
  }

  verilator_top->src_valid = 0;
  eval();

  // NOT supported parametor
  // params.tbl.is_connected
  // params.w_stride
  // params.h_stride
  for (size_t sample = 0; sample < prev_out.size(); sample++){
    // propagate delta to previous layer

    const vec_t &in = curr_delta[sample+1];
    vec_t &a        = prev_delta[sample-1];

    if(sample!=0){
      while(!verilator_top->dst_valid){
        eval();
      }
      for(outa = 0; outa < iw*ih*id; ){
        switch(outa%2){
        case 0: conv.i = verilator_top->dst_data0;break;
        case 1: conv.i = verilator_top->dst_data1;break;
        }
        a[outa] = conv.f;
        if((outa%2)==1){
          eval();
        }
        outa++;
      }
      if((iw*ih*id)%2){
        eval();
      }
    }

    for(int i = 0; i < 10; i++){
      eval();
    }

    if(sample+1<prev_out.size()){
      verilator_top->src_valid = 1;
      for(ina = 0; ina < ow*oh*od; ){
        if(verilator_top->src_ready){
          conv16.f = in[ina];
          switch(ina%4){
          case 0 : verilator_top->src_data0 = conv16.i;break;
          case 1 : verilator_top->src_data1 = conv16.i;break;
          case 2 : verilator_top->src_data2 = conv16.i;break;
          case 3 : verilator_top->src_data3 = conv16.i;eval();break;
          }
          ina++;
        }else{
          eval();
        }
      }
      if((ow*oh*od)%4){
        eval();
      }

      verilator_top->src_valid = 0;
      eval();
    }else{
      verilator_top->last = 1;
    }

    //if(1){
    if(0){
      for (size_t inc = 0; inc < id; inc++) {
        for (size_t y = 0; y < ih; y++) {
          for (size_t x = 0; x < iw; x++) {
            int yy = (y-kh+1);
            int xx = (x-kw+1);

            float_t sum{0};
            for (size_t outc = 0; outc < od; outc++) {
              for (int wy = 0; wy < kh; wy++) {   // NOLINT
                if((yy+wy)<0){wy=-yy;}
                if((yy+wy)==oh){break;}
                for (int wx = 0; wx < kw; wx++) {  // NOLINT
                  if((xx+wx)<0){wx=-xx;}
                  if((xx+wx)==ow){break;}
                  sum +=
                    W[id*outc*kh*kw + inc*kh*kw + (kh-1-wy)*kw + (kw-1-wx)] *
                    curr_delta[sample][outc*oh*ow + (yy+wy)*ow + (xx+wx)];
                }
              }
            }
            prev_delta[sample][inc*ih*iw + y*iw + x] = sum;
          }
        }
      }
    }
  }

  while(!verilator_top->dst_valid) {
    eval();
  }
  for(outa = 0; outa < iw*ih*id; ){
    switch(outa%2){
    case 0: conv.i = verilator_top->dst_data0;break;
    case 1: conv.i = verilator_top->dst_data1;break;
    }
    prev_delta[prev_out.size()-1][outa] = conv.f;
    if((outa%2)==1){
      eval();
    }
    outa++;
  }
  if((iw*ih*id)%2){
    eval();
  }

  verilator_top->backprop = 0;
  verilator_top->run = 0;
  verilator_top->last = 0;

  backprop = 0;
  run = 0;
  eval();

  }//if(id!=1)

  cbt += std::chrono::high_resolution_clock::now() - cst;
  //  cbc += main_time - csc;

  cst = std::chrono::high_resolution_clock::now();
  //  csc = main_time;

  ss          = (iw*ih*id+3)/4;
  ds          = (kw*kh*id*od+1)/2;

  verilator_top->ss = ss-1;
  verilator_top->dd = id-1;
  verilator_top->id = 0;
  verilator_top->is = iw*ih;
  verilator_top->ih = ih-1;
  verilator_top->iw = iw-1;
  verilator_top->ds = ds-1;
  verilator_top->od = od-1;
  verilator_top->os = kw*kh*id;
  verilator_top->oh = kh-1;
  verilator_top->ow = kw-1;
  verilator_top->fs = ow*oh-1;
  verilator_top->ks = ow*oh-1;
  verilator_top->kh = oh-1;
  verilator_top->kw = ow-1;

  vdd = id-1;
  vss = ss-1;
  vid = 0;
  vis = iw*ih;
  vih = ih-1;
  viw = iw-1;
  vds = ds-1;
  vod = od-1;
  vos = kw*kh*id;
  voh = kh-1;
  vow = kw-1;
  vfs = ow*oh-1;
  vks = ow*oh-1;
  vkh = oh-1;
  vkw = ow-1;

  verilator_top->backprop = 0;
  verilator_top->deltaw = 1;
  verilator_top->enbias = 0;
  verilator_top->wwrite = 0;
  verilator_top->bwrite = 0;
  verilator_top->run = 0;
  verilator_top->last = 0;
  verilator_top->src_valid = 0;
  verilator_top->dst_ready = 1;

  backprop = 0;
  enbias = 0;
  wwrite = 0;
  bwrite = 0;
  run = 0;
  eval();

  verilator_top->run = 1;
  run = 1;
  verilator_top->wwrite = 1;
  wwrite = 1;
  eval();
  verilator_top->src_valid = 1;
  for(size_t o=0;o<(od+3)/4;o++){
    for(size_t i=0;i<ow*oh;){
      if(verilator_top->src_ready){
        if((o*4+0)<od){
          conv16.f = curr_delta[0][(o*4+0)*oh*ow+i];
          verilator_top->src_data0 = conv16.i;
        }
        if((o*4+1)<od){
          conv16.f = curr_delta[0][(o*4+1)*oh*ow+i];
          verilator_top->src_data1 = conv16.i;
        }
        if((o*4+2)<od){
          conv16.f = curr_delta[0][(o*4+2)*oh*ow+i];
          verilator_top->src_data2 = conv16.i;
        }
        if((o*4+3)<od){
          conv16.f = curr_delta[0][(o*4+3)*oh*ow+i];
          verilator_top->src_data3 = conv16.i;
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
  verilator_top->wwrite = 0;
  wwrite = 0;
  eval();

  verilator_top->src_valid = 1;
  for(ina = 0; ina < iw*ih*id; ){
    if(verilator_top->src_ready){
      conv16.f = prev_out[0][ina];
      switch(ina%4){
      case 0 : verilator_top->src_data0 = conv16.i;break;
      case 1 : verilator_top->src_data1 = conv16.i;break;
      case 2 : verilator_top->src_data2 = conv16.i;break;
      case 3 : verilator_top->src_data3 = conv16.i;eval();break;
      }
      ina++;
    }else{
      eval();
    }
  }
  if((iw*ih*id)%4){
    eval();
  }
  verilator_top->src_valid = 0;
  eval();

  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    // accumulate dw

    const vec_t &delta = curr_delta[sample+1];
    const vec_t &prevo = prev_out[sample+1];

    if(sample+1<prev_out.size()){
      verilator_top->wwrite = 1;
      wwrite = 1;
      eval();
      verilator_top->src_valid = 1;
      for(size_t o=0;o<(od+3)/4;o++){
        for(size_t i=0;i<ow*oh;){
          if(verilator_top->src_ready){
            if((o*4+0)<od){
              conv16.f = delta[(o*4+0)*oh*ow+i];
              verilator_top->src_data0 = conv16.i;
            }
            if((o*4+1)<od){
              conv16.f = delta[(o*4+1)*oh*ow+i];
              verilator_top->src_data1 = conv16.i;
            }
            if((o*4+2)<od){
              conv16.f = delta[(o*4+2)*oh*ow+i];
              verilator_top->src_data2 = conv16.i;
            }
            if((o*4+3)<od){
              conv16.f = delta[(o*4+3)*oh*ow+i];
              verilator_top->src_data3 = conv16.i;
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
      verilator_top->wwrite = 0;
      wwrite = 0;
      eval();

      verilator_top->run = 1;
      run = 1;
      eval();
      verilator_top->src_valid = 1;
      for(ina = 0; ina < iw*ih*id; ){
        if(verilator_top->src_ready){
          conv16.f = prevo[ina];
          switch(ina%4){
          case 0 : verilator_top->src_data0 = conv16.i;break;
          case 1 : verilator_top->src_data1 = conv16.i;break;
          case 2 : verilator_top->src_data2 = conv16.i;break;
          case 3 : verilator_top->src_data3 = conv16.i;eval();break;
          }
          ina++;
        }else{
          eval();
        }
      }
      if((iw*ih*id)%4){
        eval();
      }
      verilator_top->src_valid = 0;
      eval();
    }else{
      verilator_top->last = 1;
    }

    //if(1){
    if(0){
      for (size_t outc = 0; outc < od; outc++) {
        for (size_t inc = 0; inc < id; inc++) {

          const float_t *delta = &curr_delta[sample][outc*ow*oh];

          for (size_t wy = 0; wy < kh; wy++) {
            for (size_t wx = 0; wx < kw; wx++) {
              float_t dst{0};

              const float_t *prevo = &prev_out[sample][wx + wy*iw + inc*iw*ih];

              for (size_t y = 0; y < oh; y++) {
                for (size_t x = 0; x < ow; x++) {
                  dst += prevo[y*iw + x] * delta[y*ow + x];
                }
              }

              dW[0][wx + wy*kw + inc*kw*kh + outc*kw*kh*id] += dst;
            }
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (size_t outc = 0; outc < od; outc++) {
        const float_t *delta = &curr_delta[sample][outc*ow*oh];
        float_t dst{0};

        for (size_t y = 0; y < oh; y++) {
          for (size_t x = 0; x < ow; x++) {
            dst += delta[y*ow + x];
          }
        }

        db[0][outc] += dst;
      }
    }

    while(!verilator_top->src_ready){
      eval();
    }
  }//for(sample)

  while(!verilator_top->dst_valid) {
    eval();
  }
  for(outa = 0; outa < kw*kh*id*od; ){
    switch(outa%2){
    case 0: conv.i = verilator_top->dst_data0;break;
    case 1: conv.i = verilator_top->dst_data1;break;
    }
    dW[0][outa] = conv.f;
    if((outa%2)==1){
      eval();
    }
    outa++;
  }
  if((kw*kh*id*od)%2){
    eval();
  }

  verilator_top->run = 0;
  verilator_top->last = 0;
  verilator_top->deltaw = 0;
  run = 0;
  eval();

  cdt += std::chrono::high_resolution_clock::now() - cst;
  //  cdc += main_time - csc;
}

}  // namespace kernels
}  // namespace tiny_dnn
