module tiny_dnn_ex_ctl
  (
   input wire         clk,
   input wire         backprop,
   input wire         run,
   input wire         wwrite,
   input wire         bwrite,
   input wire         s_init,
   input wire         out_busy,
   input wire         outrf,
   output wire        s_fin,
   output wire        k_init,
   output wire        k_fin,
   output wire        exec,
   output wire [11:0] ia,
   output wire [9:0]  wa,
   input wire [3:0]   dd,
   input wire [3:0]   id,
   input wire [9:0]   is,
   input wire [4:0]   ih,
   input wire [4:0]   iw,
   input wire [3:0]   od,
   input wire [9:0]   os,
   input wire [4:0]   oh,
   input wire [4:0]   ow,
   input wire [9:0]   fs,
   input wire [9:0]   ks,
   input wire [4:0]   kh,
   input wire [4:0]   kw,
   input wire         rst
   );
   
   wire               last_dc, last_iy, last_ix, last_ic, last_fy, last_fx;
   wire               next_dc, next_iy, next_ix, next_ic, next_fy, next_fx;
   wire [3:0]         dc,                             ic;
   wire [4:0]                  iy     , ix     ,          fy     , fx;


   loop1 #(.W(4)) l_dc(.ini(4'd0), .fin(dd),  .data(dc), .start(s_init),  .last(last_dc),
                       .clk(clk),  .rst(rst),             .next(next_dc),   .en(last_iy)  );

   loop1 #(.W(5)) l_iy(.ini(5'd0), .fin(oh),  .data(iy), .start(next_dc), .last(last_iy),
                       .clk(clk),  .rst(rst),             .next(next_iy),   .en(last_ix)  );

   loop1 #(.W(5)) l_ix(.ini(5'd0), .fin(ow),  .data(ix), .start(next_iy), .last(last_ix),
                       .clk(clk),  .rst(rst),             .next(next_ix),   .en(last_ic)  );

   wire signed [5:0]  yy = (backprop) ? iy-kh                  : iy;
   wire [4:0]         sy = (backprop) ? ((yy<0)  ? -yy   : 0)  : 0;
   wire [4:0]         ey = (backprop) ? ((iy>ih) ? ih-yy : kh) : kh;

   wire signed [5:0]  xx = (backprop) ? ix-kw                  : ix;
   wire [4:0]         sx = (backprop) ? ((xx<0)  ? -xx   : 0)  : 0;
   wire [4:0]         ex = (backprop) ? ((ix>iw) ? iw-xx : kw) : kw;

   wire               s_init0, k_init0, start;
   assign k_init = s_init0 | k_init0&!out_busy;

   dff #(.W(1)) d_s_init0(.in(s_init), .data(s_init0), .clk(clk), .rst(rst), .en(1'b1));
   dff #(.W(1)) d_exec   (.in(k_init|exec&!last_ic), .data(exec), .clk(clk), .rst(rst), .en(1'b1));
   dff #(.W(1)) d_start  (.in(k_init), .data(start), .clk(clk), .rst(rst), .en(1'b1));

   loop1 #(.W(4)) l_ic(.ini(4'd0), .fin(id),  .data(ic), .start(start),   .last(last_ic),
                       .clk(clk),  .rst(rst),             .next(next_ic),   .en(last_fy)  );

   loop1 #(.W(5)) l_fy(.ini(sy),   .fin(ey),  .data(fy), .start(next_ic), .last(last_fy),
                       .clk(clk),  .rst(rst|k_init),      .next(next_fy),   .en(last_fx)  );

   loop1 #(.W(5)) l_fx(.ini(sx),   .fin(ex),  .data(fx), .start(next_fy), .last(last_fx),
                       .clk(clk),  .rst(rst|k_init),      .next(next_fx),   .en(1'b1)  );

   wire [4:0]         y0 = yy+fy;
   wire [4:0]         x0 = xx+fx;
   
   assign ia = dc*is+ic*is+y0*(iw+1)+x0;
   assign wa = ic*(ks+1)+fy*(kw+1)+fx;

// fx loop end
// fy loop end
// ic loop end

   dff #(.W(1)) d_k_fin (.in(last_ic), .data(k_fin), .clk(clk), .rst(rst), .en(1'b1));
   dff #(.W(1)) d_k_init0 (.in(next_ix&!s_init), .data(k_init0), .clk(clk),
                           .rst(rst), .en(!out_busy|next_ix));

// ix loop end
// iy loop end
// dc loop end

   wire               s_fin0;

   dff #(.W(1)) d_s_fin0 (.in(last_dc), .data(s_fin0), .clk(clk), .rst(rst), .en(last_dc|outrf));
   dff #(.W(1)) d_s_fin (.in(s_fin0&outrf), .data(s_fin), .clk(clk), .rst(rst), .en(1'b1));

endmodule
