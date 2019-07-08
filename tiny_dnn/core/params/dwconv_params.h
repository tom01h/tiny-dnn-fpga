/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <deque>
#include <vector>

#include "tiny_dnn/core/params/params.h"

namespace tiny_dnn {
namespace core {

struct dwconv_layer_worker_specific_storage {
  std::vector<const vec_t *> prev_out_padded_;
  std::vector<vec_t> prev_out_buf_;
  std::vector<vec_t> prev_delta_padded_;
};

class dwconv_params : public Params {
 public:
  connection_table tbl;
  index3d<size_t> in;
  index3d<size_t> in_padded;
  index3d<size_t> out;
  index3d<size_t> weight;
  bool has_bias;
  padding pad_type;
  size_t w_stride;
  size_t h_stride;
  size_t w_dilation;
  size_t h_dilation;

  friend std::ostream &operator<<(std::ostream &o,
                                  const core::dwconv_params &param) {
    o << "in:        " << param.in << "\n";
    o << "out:       " << param.out << "\n";
    o << "in_padded: " << param.in_padded << "\n";
    o << "weight:    " << param.weight << "\n";
    o << "has_bias:  " << param.has_bias << "\n";
    o << "w_stride:  " << param.w_stride << "\n";
    o << "h_stride:  " << param.h_stride << "\n";
    o << "w_dilation:  " << param.w_dilation << "\n";
    o << "h_dilation:  " << param.h_dilation << "\n";
    return o;
  }
};

inline dwconv_params &Params::dwconv() {
  return *(static_cast<dwconv_params *>(this));
}

class DWConv2dPadding {
 public:
  DWConv2dPadding() {}
  explicit DWConv2dPadding(const dwconv_params &params) : params_(params) {}

  /* Applies padding to an input tensor given the convolution parameters
   *
   * @param in The input tensor
   * @param out The output tensor with padding applied
   */
  void copy_and_pad_input(const tensor_t &in, tensor_t &out) {
    if (params_.pad_type == padding::valid) {
      return;
    }

    tensor_t buf(in.size());

    for_i(true, buf.size(), [&](size_t sample) {
      // alloc temporary buffer.
      buf[sample].resize(params_.in_padded.size());

      // make padded version in order to avoid corner-case in fprop/bprop
      for (size_t c = 0; c < params_.in.depth_; c++) {
        float_t *pimg = &buf[sample][params_.in_padded.get_index(
          params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
        const float_t *pin = &in[sample][params_.in.get_index(0, 0, c)];

        for (size_t y = 0; y < params_.in.height_; y++) {
          std::copy(pin, pin + params_.in.width_, pimg);
          pin += params_.in.width_;
          pimg += params_.in_padded.width_;
        }
      }
    });

    // shrink buffer to output
    out = buf;
  }

  /* Applies unpadding to an input tensor given the convolution parameters
   *
   * @param in The input tensor
   * @param out The output tensor with unpadding applied
   */
  void copy_and_unpad_delta(const tensor_t &delta, tensor_t &delta_unpadded) {
    if (params_.pad_type == padding::valid) {
      return;
    }

    tensor_t buf(delta.size());

    for_i(true, buf.size(), [&](size_t sample) {
      // alloc temporary buffer.
      buf[sample].resize(params_.in.size());

      for (size_t c = 0; c < params_.in.depth_; c++) {
        const float_t *pin = &delta[sample][params_.in_padded.get_index(
          params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
        float_t *pdst = &buf[sample][params_.in.get_index(0, 0, c)];

        for (size_t y = 0; y < params_.in.height_; y++) {
          std::copy(pin, pin + params_.in.width_, pdst);
          pdst += params_.in.width_;
          pin += params_.in_padded.width_;
        }
      }
    });

    // shrink buffer to output
    delta_unpadded = buf;
  }

 private:
  dwconv_params params_;
};

}  // namespace core
}  // namespace tiny_dnn
