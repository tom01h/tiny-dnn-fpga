module tiny_dnn_top
  (
   input wire         S_AXI_ACLK,
   input wire         S_AXI_ARESETN,

   ////////////////////////////////////////////////////////////////////////////
   // AXI Lite Slave Interface
   input wire [31:0]  S_AXI_AWADDR,
   input wire         S_AXI_AWVALID,
   output wire        S_AXI_AWREADY,
   input wire [31:0]  S_AXI_WDATA,
   input wire [3:0]   S_AXI_WSTRB,
   input wire         S_AXI_WVALID,
   output wire        S_AXI_WREADY,
   output wire [1:0]  S_AXI_BRESP,
   output wire        S_AXI_BVALID,
   input wire         S_AXI_BREADY,

   input wire [31:0]  S_AXI_ARADDR,
   input wire         S_AXI_ARVALID,
   output wire        S_AXI_ARREADY,
   output wire [31:0] S_AXI_RDATA,
   output wire [1:0]  S_AXI_RRESP,
   output wire        S_AXI_RVALID,
   input wire         S_AXI_RREADY,

   input wire         AXIS_ACLK,
   input wire         AXIS_ARESETN,

   ////////////////////////////////////////////////////////////////////////////
   // AXI Stream Master Interface
   output wire        M_AXIS_TVALID,
   output wire [63:0] M_AXIS_TDATA,
   output wire [7:0]  M_AXIS_TSTRB,
   output wire        M_AXIS_TLAST,
   input wire         M_AXIS_TREADY,

   ////////////////////////////////////////////////////////////////////////////
   // AXI Stream Slave Interface
   output wire        S_AXIS_TREADY,
   input wire [63:0]  S_AXIS_TDATA,
   input wire [7:0]   S_AXIS_TSTRB,
   input wire         S_AXIS_TLAST,
   input wire         S_AXIS_TVALID
   );

   parameter f_num  = 16;

   wire               clk = AXIS_ACLK;

   wire               src_ready;
   wire [31:0]        dst_data0;
   wire [31:0]        dst_data1;
   wire               dst_valid;

   wire               src_valid = S_AXIS_TVALID;
   wire [15:0]        src_data0 = S_AXIS_TDATA[15:0];
   wire [15:0]        src_data1 = S_AXIS_TDATA[31:16];
   wire [15:0]        src_data2 = S_AXIS_TDATA[47:32];
   wire [15:0]        src_data3 = S_AXIS_TDATA[63:48];
   wire               src_last  = S_AXIS_TLAST;
   assign             S_AXIS_TREADY = src_ready;

   assign             M_AXIS_TVALID = dst_valid;
   assign             M_AXIS_TDATA[31:0]  = dst_data0;
   assign             M_AXIS_TDATA[63:32] = dst_data1;
   assign             M_AXIS_TSTRB  = {8{dst_valid}};
// assign             M_AXIS_TLAST  = dst_last;
   assign             M_AXIS_TLAST  = 1'b0;
   wire               dst_ready = M_AXIS_TREADY;


   wire               backprop;
   wire               deltaw;
   wire               enbias;
   wire               run;
   wire               wwrite;
   wire               bwrite;
   wire               last;
   wire               pool;

   wire [11:0]        ss;
   wire [3:0]         id;
   wire [9:0]         is;
   wire [4:0]         ih;
   wire [4:0]         iw;
   wire [11:0]        ds;
   wire [3:0]         od;
   wire [9:0]         os;
   wire [4:0]         oh;
   wire [4:0]         ow;
   wire [9:0]         fs;
   wire [9:0]         ks;
   wire [4:0]         kh;
   wire [4:0]         kw;
   wire [3:0]         dd;


   tiny_dnn_reg tiny_dnn_reg
     (
      .S_AXI_ACLK(S_AXI_ACLK),
      .S_AXI_ARESETN(S_AXI_ARESETN),

      .S_AXI_AWADDR(S_AXI_AWADDR),
      .S_AXI_AWVALID(S_AXI_AWVALID),
      .S_AXI_AWREADY(S_AXI_AWREADY),
      .S_AXI_WDATA(S_AXI_WDATA),
      .S_AXI_WSTRB(S_AXI_WSTRB),
      .S_AXI_WVALID(S_AXI_WVALID),
      .S_AXI_WREADY(S_AXI_WREADY),
      .S_AXI_BRESP(S_AXI_BRESP),
      .S_AXI_BVALID(S_AXI_BVALID),
      .S_AXI_BREADY(S_AXI_BREADY),

      .S_AXI_ARADDR(S_AXI_ARADDR),
      .S_AXI_ARVALID(S_AXI_ARVALID),
      .S_AXI_ARREADY(S_AXI_ARREADY),
      .S_AXI_RDATA(S_AXI_RDATA),
      .S_AXI_RRESP(S_AXI_RRESP),
      .S_AXI_RVALID(S_AXI_RVALID),
      .S_AXI_RREADY(S_AXI_RREADY),

      .src_ready(src_ready),

      .backprop(backprop), .deltaw(deltaw), .enbias(enbias),
      .run(run), .wwrite(wwrite), .bwrite(bwrite), .last(last), .pool(pool),
      .ss(ss), .id(id), .is(is), .ih(ih), .iw(iw),
      .ds(ds), .od(od), .os(os), .oh(oh), .ow(ow),
      .fs(fs),          .ks(ks), .kh(kh), .kw(kw), .dd(dd)
      );

   // batch control <-> sample control
   wire               s_init;
   wire               s_fin;
   wire               out_busy;
   wire               p_fin;
   wire               pool_busy;

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
   wire [15:0]        d;
   wire               signo [0:f_num];
   wire signed [9:0]  expo [0:f_num];
   wire signed [31:0] addo [0:f_num];
   wire [15:0]        pool_out;
   wire [15:0]        pool_ptr;

   batch_ctrl batch_ctrl
     (
      .clk(clk),
      .s_init(s_init),
      .s_fin(s_fin),
      .p_fin(p_fin),
      .backprop(backprop),
      .deltaw(deltaw),
      .run(run),
      .pool(pool),
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
      .pool(pool),
      .outr(outr),
      .accr(accr),
      .oa({execp,oa[11:0]}),
      .signo(signo[0]),
      .expo(expo[0]),
      .addo(addo[0]),
      .po(pool_out),
      .pp(pool_ptr)
      );

   out_ctrl out_ctrl
     (
      .clk(clk),
      .rst(~(run|pool)),
      .dst_acc(dst_acc),
      .s_init(s_init),
      .k_init(k_init),
      .k_fin(k_fin),
      .p_fin(p_fin),
      .pool(pool),
      .pool_busy(pool_busy),
      .src_valid(src_valid),
      .out_busy(out_busy),
      .od(od[3:0]),
      .os(os[9:0]),
      .outr(outr),
      .outrf(outrf),
      .accr(accr),
      .oa(oa[11:0]),
      .update(sum_update)
      );

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
      .s_fin(s_fin),
      .k_init(k_init),
      .k_fin(k_fin),
      .exec(exec),
      .ia(ia),
      .wa(wa),
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

   tiny_dnn_pool tiny_dnn_pool
     (
      .clk(clk),
      .pool(pool),
      .p_fin(p_fin),
      .en(pool&src_valid),
      .ow(ow),
      .d0(src_data0),
      .d1(src_data1),
      .d2(src_data2),
      .d3(src_data3),
      .pool_busy(pool_busy),
      .po(pool_out),
      .pp(pool_ptr)
      );

   assign signo[f_num] = 0;
   assign expo[f_num] = 0;
   assign addo[f_num] = 0;

   wire [15:0]        write_data [0:f_num-1];
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
                .signi(signo[i+1]),
                .expi(expo[i+1]),
                .addi(addo[i+1]),
                .signo(signo[i]),
                .expo(expo[i]),
                .addo(addo[i])
                );
      end
   endgenerate

endmodule
