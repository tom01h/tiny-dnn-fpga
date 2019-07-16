/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// verilator
#include "unistd.h"
#include "getopt.h"
#include "systemc.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include "tiny_dnn_sc_ctl.h"
#include "Vtiny_dnn_top.h"

sc_clock clk ("clk", 10, SC_NS);

sc_signal <bool>      backprop;
sc_signal <bool>      enbias;
sc_signal <bool>      run;
sc_signal <bool>      wwrite;
sc_signal <bool>      bwrite;

sc_signal <sc_uint<12> > vss;
sc_signal <sc_uint<12> > vds;

sc_signal <sc_uint<4> >  vdd;
sc_signal <sc_uint<4> >  vid;
sc_signal <sc_uint<10> > vis;
sc_signal <sc_uint<5> >  vih;
sc_signal <sc_uint<5> >  viw;
sc_signal <sc_uint<4> >  vod;
sc_signal <sc_uint<10> > vos;
sc_signal <sc_uint<5> >  voh;
sc_signal <sc_uint<5> >  vow;
sc_signal <sc_uint<10> > vfs;
sc_signal <sc_uint<10> > vks;
sc_signal <sc_uint<5> >  vkh;
sc_signal <sc_uint<5> >  vkw;

sc_signal <bool>         s_init;
sc_signal <bool>         out_busy;
sc_signal <bool>         outrf;
sc_signal <bool>         s_fin;
sc_signal <bool>         k_init;
sc_signal <bool>         k_fin;
sc_signal <bool>         exec;
sc_signal <sc_uint<12> > ia;
sc_signal <sc_uint<10> > wa;

vluint64_t main_time = 0;
vluint64_t vcdstart = 0;
//vluint64_t vcdstart = 500000;
vluint64_t vcdend = vcdstart + 500000;

VerilatedVcdC* tfp;
Vtiny_dnn_top* verilator_top;
// verilator

void eval()
{
  // negedge clk /////////////////////////////
  verilator_top->clk = !clk;

  verilator_top->eval();
  sc_start(5, SC_NS);

  if((main_time>=vcdstart)&((main_time<vcdend)|(vcdend==0)))
    tfp->dump(main_time);
  main_time += 5;

  // posegedge clk /////////////////////////////
  verilator_top->clk = !clk;

  verilator_top->eval();
  sc_start(5, SC_NS);
  //          verilog -> SystemC
  s_init = verilator_top->sc_s_init;
  out_busy = verilator_top->sc_out_busy;
  outrf = verilator_top->sc_outrf;
  //          SystemC -> verilog

  verilator_top->sc_s_fin = s_fin;
  verilator_top->sc_k_init = k_init;
  verilator_top->sc_k_fin = k_fin;
  verilator_top->sc_exec = exec;
  verilator_top->sc_ia = ia.read();
  verilator_top->sc_wa = wa.read();

  if((main_time>=vcdstart)&((main_time<vcdend)|(vcdend==0)))
    tfp->dump(main_time);
  main_time += 5;

  return;
}
// verilator

#include <iostream>

#include <chrono>    // for high_resolution_clock, NOLINT
std::chrono::high_resolution_clock::time_point ast;
std::chrono::high_resolution_clock::duration aft, abt;
std::chrono::high_resolution_clock::time_point pst;
std::chrono::high_resolution_clock::duration pft, pbt;
std::chrono::high_resolution_clock::time_point cst;
std::chrono::high_resolution_clock::duration cft, cbt, cdt;
std::chrono::high_resolution_clock::time_point dst;
std::chrono::high_resolution_clock::duration dft, dbt, ddt;
//vluint64_t csc;
//vluint64_t cfc, cbc, cdc;


#include "tiny_dnn/tiny_dnn.h"

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {

  //using fc       = tiny_dnn::fully_connected_layer;
  using conv     = tiny_dnn::convolutional_layer;
  using dwconv   = tiny_dnn::dwconvolutional_layer;
  using max_pool = tiny_dnn::max_pooling_layer;
  using relu     = tiny_dnn::relu_layer;
  using softmax  = tiny_dnn::softmax_layer;
  //using avg_pool = tiny_dnn::average_pooling_layer;

  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;

  nn << conv(28, 28, 5, 1, 6, padding::valid, true, 1, 1, 1, 1, backend_type)
     << max_pool(24, 24, 6, 2)
     << relu()
     << dwconv(12, 12, 3, 6, padding::valid, true, 1, 1, 1, 1, backend_type)
     << relu()
     << conv(10, 10, 1, 6, 10, padding::valid, true, 1, 1, 1, 1, backend_type)
     << relu()
     << dwconv(10, 10, 3, 10, padding::valid, true, 1, 1, 1, 1, backend_type)
     << relu()
     << conv(8, 8, 1, 10, 16, padding::valid, true, 1, 1, 1, 1, backend_type)
     << max_pool(8, 8, 16, 2)
     << relu()
     << conv(4, 4, 4, 16, 10, padding::valid, true, 1, 1, 1, 1, backend_type)
     << softmax(10);
}

static void train_net(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch,
                        tiny_dnn::core::backend_t backend_type) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adagrad optimizer;

  construct_net(nn, backend_type);

  std::cout << "load models..." << std::endl;

  // load MNIST dataset
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;

  tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               &train_images, -1.0, 1.0, 0, 0);
  tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, -1.0, 1.0, 0, 0);

  train_labels.resize(2400);
  train_images.resize(2400);
  //train_labels.resize(20000);
  //train_images.resize(20000);
  //train_labels.resize(32);
  //train_images.resize(32);

  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    //tiny_dnn::result res = nn.test(test_images, test_labels);
    //std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  std::cout << "activ forward "
            << std::chrono::duration_cast<std::chrono::milliseconds>(aft).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "activ back "
            << std::chrono::duration_cast<std::chrono::milliseconds>(abt).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "pool forward "
            << std::chrono::duration_cast<std::chrono::milliseconds>(pft).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "pool back "
            << std::chrono::duration_cast<std::chrono::milliseconds>(pbt).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "cov forward "
            << std::chrono::duration_cast<std::chrono::milliseconds>(cft).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "cov back "
            << std::chrono::duration_cast<std::chrono::milliseconds>(cbt).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "cov d param "
            << std::chrono::duration_cast<std::chrono::milliseconds>(cdt).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "dwcov forward "
            << std::chrono::duration_cast<std::chrono::milliseconds>(dft).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "dwcov back "
            << std::chrono::duration_cast<std::chrono::milliseconds>(dbt).count()
            << " ms elapsed"
            << std::endl;
  std::cout << "dwcov d param "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ddt).count()
            << " ms elapsed"
            << std::endl;

  //  std::cout << "cov forward "
  //            << (cfc / 1000)
  //            << " us elapsed"
  //            << std::endl;
  //  std::cout << "cov back "
  //            << (cbc / 1000)
  //            << " us elapsed"
  //            << std::endl;
  //  std::cout << "cov d param "
  //            << (cdc / 1000)
  //            << " us elapsed"
  //            << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
  // save network model & trained weights
  nn.save("Net-model");
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {{
    "internal", "nnpack", "libdnn", "avx", "opencl",
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epochs 1"
            << " --minibatch_size 16"
            << " --backend_type internal" << std::endl;
}

int sc_main(int argc, char **argv) {
  double learning_rate                   = 1;
  int epochs                             = 1;
  std::string data_path                  = "";
  int minibatch_size                     = 16;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }

// verilator
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);
  tfp = new VerilatedVcdC;
  verilator_top = new Vtiny_dnn_top;
  verilator_top->trace(tfp, 99); // requires explicit max levels param
  tfp->open("tmp.vcd");
  main_time = 0;

  tiny_dnn_sc_ctl U_tiny_dnn_sc_ctl("U_tiny_dnn_sc_ctl");
  U_tiny_dnn_sc_ctl.clk(clk);
  U_tiny_dnn_sc_ctl.backprop(backprop);
  U_tiny_dnn_sc_ctl.run(run);
  U_tiny_dnn_sc_ctl.wwrite(wwrite);
  U_tiny_dnn_sc_ctl.bwrite(bwrite);
  U_tiny_dnn_sc_ctl.s_init(s_init);
  U_tiny_dnn_sc_ctl.out_busy(out_busy);
  U_tiny_dnn_sc_ctl.outrf(outrf);
  U_tiny_dnn_sc_ctl.s_fin(s_fin);
  U_tiny_dnn_sc_ctl.k_init(k_init);
  U_tiny_dnn_sc_ctl.k_fin(k_fin);
  U_tiny_dnn_sc_ctl.exec(exec);
  U_tiny_dnn_sc_ctl.ia(ia);
  U_tiny_dnn_sc_ctl.wa(wa);

  U_tiny_dnn_sc_ctl.dd(vdd);
  U_tiny_dnn_sc_ctl.id(vid);
  U_tiny_dnn_sc_ctl.is(vis);
  U_tiny_dnn_sc_ctl.ih(vih);
  U_tiny_dnn_sc_ctl.iw(viw);
  U_tiny_dnn_sc_ctl.od(vod);
  U_tiny_dnn_sc_ctl.os(vos);
  U_tiny_dnn_sc_ctl.oh(voh);
  U_tiny_dnn_sc_ctl.ow(vow);
  U_tiny_dnn_sc_ctl.fs(vfs);
  U_tiny_dnn_sc_ctl.ks(vks);
  U_tiny_dnn_sc_ctl.kh(vkh);
  U_tiny_dnn_sc_ctl.kw(vkw);

  verilator_top->clk = 1;
  verilator_top->eval();
  sc_start(5, SC_NS);
  main_time += 5;
// verilator

  if (data_path == "") {
    std::cerr << "Data path not specified." << std::endl;
    usage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 20000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (20000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train_net(data_path, learning_rate, epochs, minibatch_size, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  delete verilator_top;
  tfp->close();
  return 0;
}
