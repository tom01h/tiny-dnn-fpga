/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <utility>

#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

#include <chrono>    // for high_resolution_clock, NOLINT

int af=0;
int ab=0;

namespace tiny_dnn {

class relu_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "relu-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    ast = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::max(float_t(0), x[j]);
    }
    aft += std::chrono::high_resolution_clock::now() - ast;
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    ast = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of relu)
      dx[j] = dy[j] * (y[j] > float_t(0) ? float_t(1) : float_t(0));
    }
    abt += std::chrono::high_resolution_clock::now() - ast;
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
