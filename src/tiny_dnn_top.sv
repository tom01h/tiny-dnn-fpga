module tiny_dnn_top
  (
   input wire        clk,

   input wire        backprop,
   input wire        deltaw,
   input wire        enbias,
   input wire        run,
   input wire        wwrite,
   input wire        bwrite,
   input wire        last,

   output wire       sc_s_init,
   output wire       sc_out_busy,
   output wire       sc_outrf,
   input wire        sc_s_fin,
   input wire        sc_k_init,
   input wire        sc_k_fin,
   input wire        sc_exec,
   input wire [11:0] sc_ia,
   input wire [9:0]  sc_wa,

   input wire        src_valid,
   input real        src_data0,
   input real        src_data1,
   input real        src_data2,
   input real        src_data3,
   input wire        src_last,
   output wire       src_ready,

   output wire       dst_valid,
   output real       dst_data0,
   output real       dst_data1,
   output wire       dst_last,
   input wire        dst_ready,

   input wire [11:0] ss,
   input wire [3:0]  dd,
   input wire [3:0]  id,
   input wire [9:0]  is,
   input wire [4:0]  ih,
   input wire [4:0]  iw,
   input wire [11:0] ds,
   input wire [3:0]  od,
   input wire [9:0]  os,
   input wire [4:0]  oh,
   input wire [4:0]  ow,
   input wire [9:0]  fs,
   input wire [9:0]  ks,
   input wire [4:0]  kh,
   input wire [4:0]  kw
   );

   parameter f_num  = 16;

   // batch control <-> sample control
   wire               s_init;
   wire               s_fin;
   wire               out_busy;

   // sample control -> core
   wire               k_init;
   wire               k_fin;
   wire [9:0]         wa;

   // sample control -> core, src buffer
   wire               exec;
   wire [11:0]        ia;
   wire               execp;
   wire               inp;
   // out control -> core, dst buffer
   wire               outr;
   wire               outrf;
   wire               accr;
   wire [11:0]        oa;
   wire               sum_update;
   wire               outp;

   // batch control -> weight buffer
   wire [3:0]         prm_v;
   wire [9:0]         prm_a;
   // batch control -> src buffer
   wire               src_v;
   wire [11:0]        src_a;
   // batch control -> dst buffer
   wire               dst_v;
   wire [11:0]        dst_a;
   wire               dst_acc;

   // core <-> src,dst buffer
   real               d;
   real               sum [0:f_num];

   batch_ctrl batch_ctrl
     (
      .clk(clk),
      .s_init(s_init),
      .s_fin(s_fin),
      .backprop(backprop),
      .deltaw(deltaw),
      .run(run),
      .wwrite(wwrite),
      .bwrite(bwrite),
      .last(last),

      .src_valid(src_valid),
      .src_last(src_last),
      .src_ready(src_ready),
      .dst_valid(dst_valid),
      .dst_ready(dst_ready),

      .prm_v(prm_v[3:0]),
      .prm_a(prm_a[9:0]),
      .src_v(src_v),
      .src_a(src_a[11:0]),
      .dst_v(dst_v),
      .dst_a(dst_a[11:0]),
      .dst_acc(dst_acc),

      .execp(execp),
      .inp(inp),
      .outp(outp),

      .ss(ss[11:0]),
      .ds(ds[11:0]),
      .id(id[3:0]),
      .od(od[3:0]),
      .fs(fs[9:0]),
      .ks(ks[9:0])
      );

   src_buf src_buf
     (
      .clk(clk),
      .src_v(src_v),
      .src_a({inp,src_a[11:0]}),
      .src_d0(src_data0),
      .src_d1(src_data1),
      .src_d2(src_data2),
      .src_d3(src_data3),
      .exec(exec|k_init),
      .ia({execp,ia[11:0]}),
      .d(d)
      );

   dst_buf dst_buf
     (
      .clk(clk),
      .dst_v(dst_v),
      .dst_a({outp,dst_a[11:0]}),
      .dst_d0(dst_data0),
      .dst_d1(dst_data1),
      .outr(outr),
      .accr(accr),
      .oa({execp,oa[11:0]}),
      .sum(sum[0])
      );

   out_ctrl out_ctrl
     (
      .clk(clk),
      .rst(~run),
      .dst_acc(dst_acc),
      .s_init(s_init),
      .k_init(k_init),
      .k_fin(k_fin),
      .out_busy(out_busy),
      .od(od[3:0]),
      .os(os[9:0]),
      .outr(outr),
      .outrf(outrf),
      .accr(accr),
      .oa(oa[11:0]),
      .update(sum_update)
      );

/*
   assign sc_s_init = s_init;
   assign sc_out_busy = out_busy;
   assign sc_outrf = outrf;
   assign s_fin = sc_s_fin;
   assign k_init = sc_k_init;
   assign k_fin = sc_k_fin;
   assign exec = sc_exec;
   assign ia = sc_ia;
   assign wa = sc_wa;
/**/

   tiny_dnn_ex_ctl tiny_dnn_ex_ctl
     (
      .clk(clk),
      .backprop(backprop),
      .run(run),
      .wwrite(wwrite),
      .bwrite(bwrite),
      .s_init(s_init),
      .out_busy(out_busy),
      .outrf(outrf),
/**/
      .s_fin(s_fin),
      .k_init(k_init),
      .k_fin(k_fin),
      .exec(exec),
      .ia(ia),
      .wa(wa),
/**/
      .dd(dd),
      .id(id),
      .is(is),
      .ih(ih),
      .iw(iw),
      .od(od),
      .os(os),
      .oh(oh),
      .ow(ow),
      .fs(fs),
      .ks(ks),
      .kh(kh),
      .kw(kw),
      .rst(~run)
      );

   assign sum[f_num] = 0;

   real write_data [0:f_num-1];
   generate
      genvar j;
      for (j = 0; j < f_num/4; j = j + 1) begin
         assign write_data[j*4  ] = src_data0;
         assign write_data[j*4+1] = src_data1;
         assign write_data[j*4+2] = src_data2;
         assign write_data[j*4+3] = src_data3;
      end
   endgenerate
   generate
      genvar i;
      for (i = 0; i < f_num; i = i + 1) begin
         tiny_dnn_core tiny_dnn_core
               (
                .clk(clk),
                .init(k_init),
                .write((wwrite|bwrite)&(prm_v[3:0] == (i/4)) & src_valid & src_ready),
                .bwrite(bwrite),
                .exec(exec),
                .outr(outr),
                .update(sum_update),
                .bias(k_fin&enbias),
                .ra({deltaw&execp,wa[9:0]}),
                .wa({deltaw&inp,  prm_a[9:0]}),
                .d(d),
                .wd(write_data[i]),
                .sum_in(sum[i+1]),
                .sum(sum[i])
                );
      end
   endgenerate

endmodule
