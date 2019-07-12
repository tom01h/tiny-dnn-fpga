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

void dma_reset(){
  dma_addr[0x30/4] = 4;
  dma_addr[0x00/4] = 4;
  while (dma_addr[0x00/4] & 0x4);
}

union{
  struct{
    int16_t j;
    int16_t i;
  };
  float f;
} conv16;

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_internal(const tensor_t &in_data,
                               const vec_t &W,
                               const vec_t &bias,
                               tensor_t &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {
  cst = std::chrono::high_resolution_clock::now();


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

    dnn_addr[ 5] = ss-1;       //ss
    dnn_addr[ 6] = id-1;       //id
    dnn_addr[ 7] = iw*ih;      //is
    dnn_addr[ 8] = ih-1;       //ih
    dnn_addr[ 9] = iw-1;       //iw

    dnn_addr[10] = ds-1;       //ds
    dnn_addr[11] = od-1;       //od
    dnn_addr[12] = ow*oh;      //os
    dnn_addr[13] = oh-1;       //oh
    dnn_addr[14] = ow-1;       //ow

    dnn_addr[15] = 0;          //dd

    dnn_addr[ 1] = kw*kh*id-1; //fs
    dnn_addr[ 2] = kw*kh-1;    //ks
    dnn_addr[ 3] = kh-1;       //kh
    dnn_addr[ 4] = kw-1;       //kw


    /////////////////////////////////////////////////////////
    // Weight transfer

    // DMA Buffer
    size_t ptr=0;
    for(size_t o=0;o<(od+3)/4;o++){
      for(size_t i=0;i<id*kh*kw;i++){
        if((o*4+0)<od){
          conv16.f = W[(o*4+0)*id*kh*kw+i];
          src_addr[ptr+0] = conv16.i;
        }
        if((o*4+1)<od){
          conv16.f = W[(o*4+1)*id*kh*kw+i];
          src_addr[ptr+1] = conv16.i;
        }
        if((o*4+2)<od){
          conv16.f = W[(o*4+2)*id*kh*kw+i];
          src_addr[ptr+2] = conv16.i;
        }
        if((o*4+3)<od){
          conv16.f = W[(o*4+3)*id*kh*kw+i];
          src_addr[ptr+3] = conv16.i;
        }
        ptr+=4;
      }
    }
    __asm__("DSB 15");

    dnn_addr[0] = 0; // init
    dnn_addr[0] = 2; // wwrite

    // AXI DMA transfer tx
    dma_reset();
    dma_addr[0x00/4] = 1;
    dma_addr[0x18/4] = src_phys;
    dma_addr[0x28/4] = ((od+3)/4)*4*id*kh*kw*2;

    while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish

    /////////////////////////////////////////////////////////
    // Bias transfer
    // DMA Buffer
    for (size_t o = 0; o < od; o++) {
      if (params.has_bias) {
        conv16.f = bias[o];
        src_addr[o] = conv16.i;
      }else{
        src_addr[o] = 0;
      }
    }
    __asm__("DSB 15");

    dnn_addr[0] = 0; // init
    dnn_addr[0] = 1; // bwrite

    // AXI DMA transfer tx
    dma_reset();
    dma_addr[0x00/4] = 1;
    dma_addr[0x18/4] = src_phys;
    dma_addr[0x28/4] = od*2;

    while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish

    /////////////////////////////////////////////////////////
    // Run
    dnn_addr[0] = 0;   // init
    dnn_addr[0] = 4|8; // run|enbias

    for(size_t i=0;i<iw*ih*id;i++){
      conv16.f = in_data[0][i];
      src_addr[i] = conv16.i;
    }
    __asm__("DSB 15");

    // AXI DMA transfer tx
    dma_reset();
    dma_addr[0x00/4] = 1;
    dma_addr[0x18/4] = src_phys;
    dma_addr[0x28/4] = iw*ih*id*2;

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
        for(size_t i=0;i<iw*ih*id;i++){
          conv16.f = in_data[sample+1][i];
          src_addr[i] = conv16.i;
        }
        __asm__("DSB 15");

        // AXI DMA transfer tx
        dma_addr[0x00/4] = 1;
        dma_addr[0x18/4] = src_phys;
        dma_addr[0x28/4] = iw*ih*id*2;

        // Wait for the tx to finish
        while ((dma_addr[0x04/4] & 0x1000)!=0x1000);
      }else{
        dnn_addr[0] = 4|8|64; // run|enbias|last
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

  cst = std::chrono::high_resolution_clock::now();

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

  if(id!=1){ // because input delta NOT USED

  dnn_addr[ 5] = ss-1;       //ss
  dnn_addr[ 6] = od-1;       //id
  dnn_addr[ 7] = ow*oh;      //is
  dnn_addr[ 8] = oh-1;       //ih
  dnn_addr[ 9] = ow-1;       //iw

  dnn_addr[10] = ds-1;       //ds
  dnn_addr[11] = id-1;       //od
  dnn_addr[12] = iw*ih;      //os
  dnn_addr[13] = ih-1;       //oh
  dnn_addr[14] = iw-1;       //ow

  dnn_addr[15] = 0;          //dd

  dnn_addr[ 1] = kw*kh*od-1; //fs
  dnn_addr[ 2] = kw*kh-1;    //ks
  dnn_addr[ 3] = kh-1;       //kh
  dnn_addr[ 4] = kw-1;       //kw

  /////////////////////////////////////////////////////////
  // Weight transfer
  // DMA Buffer
  size_t ptr=0;
  for(size_t ii=0;ii<od;ii++){        //od-1=veri->id
    for(size_t o=0;o<(id+3)/4;o++){   //id-1=veri->od
      for(size_t i=0;i<kh*kw;i++){
        if((o*4+0)<id){
          conv16.f = W[(o*4+0)*kh*kw+i+ii*kh*kw*id];
          src_addr[ptr+0] = conv16.i;
        }
        if((o*4+1)<id){
          conv16.f = W[(o*4+1)*kh*kw+i+ii*kh*kw*id];
          src_addr[ptr+1] = conv16.i;
        }
        if((o*4+2)<id){
          conv16.f = W[(o*4+2)*kh*kw+i+ii*kh*kw*id];
          src_addr[ptr+2] = conv16.i;
        }
        if((o*4+3)<id){
          conv16.f = W[(o*4+3)*kh*kw+i+ii*kh*kw*id];
          src_addr[ptr+3] = conv16.i;
        }
        ptr+=4;
      }
    }
  }
  __asm__("DSB 15");

  dnn_addr[0] = 0|16; // init|backprop
  dnn_addr[0] = 2|16; // wwrite|backprop

  // AXI DMA transfer tx
  dma_reset();
  dma_addr[0x00/4] = 1;
  dma_addr[0x18/4] = src_phys;
  dma_addr[0x28/4] = ((id+3)/4)*4*od*kh*kw*2;

  while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish

  /////////////////////////////////////////////////////////
  // Run
  for(size_t i=0;i<ow*oh*od;i++){
    conv16.f = curr_delta[0][i];
    src_addr[i] = conv16.i;
  }
  __asm__("DSB 15");

  dnn_addr[0] = 0|16; // init|backprop
  dnn_addr[0] = 4|16; // run|backprop

  // AXI DMA transfer tx
  dma_reset();
  dma_addr[0x00/4] = 1;
  dma_addr[0x18/4] = src_phys;
  dma_addr[0x28/4] = ow*oh*od*2;

  // Wait for the tx to finish
  while ((dma_addr[0x04/4] & 0x1000)!=0x1000);

  for (size_t sample = 0; sample < prev_out.size(); sample++) {

    if(sample!=0){
      __asm__("DSB 15");
      for(size_t i=0;i<id*ih*iw;i++){
        prev_delta[sample-1][i] = dst_addr[i];
      }
    }

    // AXI DMA transfer rx
    dma_reset();
    dma_addr[0x30/4] = 1;
    dma_addr[0x48/4] = dst_phys;
    dma_addr[0x58/4] = iw*ih*id*4;

    if(sample+1<prev_out.size()){
      for(size_t i=0;i<ow*oh*od;i++){
        conv16.f = curr_delta[sample+1][i];
        src_addr[i] = conv16.i;
      }
      __asm__("DSB 15");

      // AXI DMA transfer tx
      dma_addr[0x00/4] = 1;
      dma_addr[0x18/4] = src_phys;
      dma_addr[0x28/4] = ow*oh*od*2;

      // Wait for the tx to finish
      while ((dma_addr[0x04/4] & 0x1000)!=0x1000);
    }else{
      dnn_addr[0] = 4|16|64; // run|backprop|last
    }

    // Wait for the rx to finish
    while ((dma_addr[0x34/4] & 0x1000)!=0x1000) ;
    dma_reset();

  }

  __asm__("DSB 15");
  for(size_t i=0;i<id*ih*iw;i++){
    prev_delta[prev_out.size()-1][i] = dst_addr[i];
  }

  dnn_addr[ 0] = 0; // idle

  }

  cbt += std::chrono::high_resolution_clock::now() - cst;

  ss          = (iw*ih*id+3)/4;
  ds          = (kw*kh*id*od+1)/2;

  cst = std::chrono::high_resolution_clock::now();

  dnn_addr[ 5] = ss-1;       //ss
  dnn_addr[ 6] = 0;          //id
  dnn_addr[ 7] = iw*ih;      //is
  dnn_addr[ 8] = ih-1;       //ih
  dnn_addr[ 9] = iw-1;       //iw

  dnn_addr[10] = ds-1;       //ds
  dnn_addr[11] = od-1;       //od
  dnn_addr[12] = kw*kh*id;   //os
  dnn_addr[13] = kh-1;       //oh
  dnn_addr[14] = kw-1;       //ow

  dnn_addr[15] = id-1;       //dd

  dnn_addr[ 1] = ow*oh-1;    //fs
  dnn_addr[ 2] = ow*oh-1;    //ks
  dnn_addr[ 3] = oh-1;       //kh
  dnn_addr[ 4] = ow-1;       //kw

  /////////////////////////////////////////////////////////
  // current delta transfer

  // DMA Buffer
  size_t ptr=0;

  for(size_t o=0;o<od;o++){
    db[0][o] = 0;
  }
  for(size_t o=0;o<(od+3)/4;o++){
    for(size_t i=0;i<ow*oh;i++){
      if((o*4+0)<od){
        conv16.f = curr_delta[0][(o*4+0)*oh*ow+i];
        src_addr[ptr+0] = conv16.i;
        db[0][o*4+0] += conv16.f;
      }
      if((o*4+1)<od){
        conv16.f = curr_delta[0][(o*4+1)*oh*ow+i];
        src_addr[ptr+1] = conv16.i;
        db[0][o*4+1] += conv16.f;
      }
      if((o*4+2)<od){
        conv16.f = curr_delta[0][(o*4+2)*oh*ow+i];
        src_addr[ptr+2] = conv16.i;
        db[0][o*4+2] += conv16.f;
      }
      if((o*4+3)<od){
        conv16.f = curr_delta[0][(o*4+3)*oh*ow+i];
        src_addr[ptr+3] = conv16.i;
        db[0][o*4+3] += conv16.f;
      }
      ptr+=4;
    }
  }
  __asm__("DSB 15");

  dnn_addr[0] = 0; // init
  dnn_addr[0] = 2; // wwrite

  // AXI DMA transfer tx
  dma_reset();
  dma_addr[0x00/4] = 1;
  dma_addr[0x18/4] = src_phys;
  dma_addr[0x28/4] = ((od+3)/4)*4*ow*oh*2;

  while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish

  /////////////////////////////////////////////////////////
  // in data
  for(size_t i=0;i<iw*ih*id;i++){
    conv16.f = prev_out[0][i];
    src_addr[i] = conv16.i;
  }
  __asm__("DSB 15");

  dma_reset();

  dnn_addr[0] = 0; // init
  dnn_addr[0] = 4|32; // run|deltaw

  // AXI DMA transfer tx
  dma_addr[0x00/4] = 1;
  dma_addr[0x18/4] = src_phys;
  dma_addr[0x28/4] = iw*ih*id*2;

  // Wait for the tx to finish
  while ((dma_addr[0x04/4] & 0x1000)!=0x1000);

  for (size_t sample = 0; sample < prev_out.size(); sample++) {

    if(sample+1<prev_out.size()){
      dnn_addr[0] = 2|4|32; // wwrite|run|deltaw

      ptr=0;
      for(size_t o=0;o<(od+3)/4;o++){
        for(size_t i=0;i<ow*oh;i++){
          if((o*4+0)<od){
            conv16.f = curr_delta[sample+1][(o*4+0)*oh*ow+i];
            src_addr[ptr+0] = conv16.i;
            db[0][o*4+0] += conv16.f;
          }
          if((o*4+1)<od){
            conv16.f = curr_delta[sample+1][(o*4+1)*oh*ow+i];
            src_addr[ptr+1] = conv16.i;
            db[0][o*4+1] += conv16.f;
          }
          if((o*4+2)<od){
            conv16.f = curr_delta[sample+1][(o*4+2)*oh*ow+i];
            src_addr[ptr+2] = conv16.i;
            db[0][o*4+2] += conv16.f;
          }
          if((o*4+3)<od){
            conv16.f = curr_delta[sample+1][(o*4+3)*oh*ow+i];
            src_addr[ptr+3] = conv16.i;
            db[0][o*4+3] += conv16.f;
          }
          ptr+=4;
        }
      }
      __asm__("DSB 15");

      // AXI DMA transfer tx
      dma_reset();
      dma_addr[0x00/4] = 1;
      dma_addr[0x18/4] = src_phys;
      dma_addr[0x28/4] = ((od+3)/4)*4*ow*oh*2;

      while ((dma_addr[0x04/4] & 0x1000)!=0x1000); // Wait for the tx to finish
    }

    if(sample+1<prev_out.size()){
      dnn_addr[0] = 4|32; // run|deltaw

      for(size_t i=0;i<iw*ih*id;i++){
        conv16.f = prev_out[sample+1][i];
        src_addr[i] = conv16.i;
      }
      __asm__("DSB 15");

      // AXI DMA transfer tx
      dma_reset();
      dma_addr[0x00/4] = 1;
      dma_addr[0x18/4] = src_phys;
      dma_addr[0x28/4] = iw*ih*id*2;

      // Wait for the tx to finish
      while ((dma_addr[0x04/4] & 0x1000)!=0x1000);

    }else{
      // AXI DMA transfer rx
      dma_reset();
      dma_addr[0x30/4] = 1;
      dma_addr[0x48/4] = dst_phys;
      dma_addr[0x58/4] = kw*kh*id*od*4;

      dnn_addr[0] = 4|32|64; // run|deltaw|last
    }

    // Wait for sample finish
    while ((dnn_addr[0] & 0x80000000)!=0x80000000) ;

  }

  // Wait for the rx to finish
  while ((dma_addr[0x34/4] & 0x1000)!=0x1000) ;
  dma_reset();

  __asm__("DSB 15");
  for(size_t i=0;i<kw*kh*id*od;i++){
    dW[0][i] = dst_addr[i];
  }

  dnn_addr[0] = 0; // idle

  cdt += std::chrono::high_resolution_clock::now() - cst;
}

}  // namespace kernels
}  // namespace tiny_dnn
