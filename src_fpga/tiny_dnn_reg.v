module tiny_dnn_reg
  (
   input wire        S_AXI_ACLK,
   input wire        S_AXI_ARESETN,

   ////////////////////////////////////////////////////////////////////////////
   // AXI Lite Slave Interface
   input wire [31:0] S_AXI_AWADDR,
   input wire        S_AXI_AWVALID,
   output wire       S_AXI_AWREADY,
   input wire [31:0] S_AXI_WDATA,
   input wire [3:0]  S_AXI_WSTRB,
   input wire        S_AXI_WVALID,
   output wire       S_AXI_WREADY,
   output wire [1:0] S_AXI_BRESP,
   output wire       S_AXI_BVALID,
   input wire        S_AXI_BREADY,

   input wire [31:0] S_AXI_ARADDR,
   input wire        S_AXI_ARVALID,
   output wire       S_AXI_ARREADY,
   output reg [31:0] S_AXI_RDATA,
   output wire [1:0] S_AXI_RRESP,
   output wire       S_AXI_RVALID,
   input wire        S_AXI_RREADY,

   ////////////////////////////////////////////////////////////////////////////
   // Control signal
   input wire        src_ready,

   output reg        backprop,
   output reg        deltaw,
   output reg        enbias,
   output reg        run,
   output reg        wwrite,
   output reg        bwrite,
   output reg        last,

   output reg [11:0] ss,
   output reg [3:0]  id,
   output reg [9:0]  is,
   output reg [4:0]  ih,
   output reg [4:0]  iw,
   output reg [11:0] ds,
   output reg [3:0]  od,
   output reg [9:0]  os,
   output reg [4:0]  oh,
   output reg [4:0]  ow,
   output reg [9:0]  fs,
   output reg [9:0]  ks,
   output reg [4:0]  kh,
   output reg [4:0]  kw,
   output reg [3:0]  dd
   );

   reg [3:0]          axist;
   reg [5:2]          wb_adr_i;
   reg [31:0]         wb_dat_i;

   assign S_AXI_BRESP = 2'b00;
   assign S_AXI_RRESP = 2'b00;
   assign S_AXI_AWREADY = (axist == 4'b0000)|(axist == 4'b0010);
   assign S_AXI_WREADY  = (axist == 4'b0000)|(axist == 4'b0001);
   assign S_AXI_ARREADY = (axist == 4'b0000);
   assign S_AXI_BVALID  = (axist == 4'b0011);
   assign S_AXI_RVALID  = (axist == 4'b0100);

   always @(posedge S_AXI_ACLK)begin
      if(~S_AXI_ARESETN)begin
         axist<=4'b0000;

         wb_adr_i<=0;
         wb_dat_i<=0;
      end else if(axist==4'b000)begin
         if(S_AXI_AWVALID & S_AXI_WVALID)begin
            axist<=4'b00011;
            wb_adr_i[5:2]<=S_AXI_AWADDR[5:2];
            wb_dat_i<=S_AXI_WDATA;
         end else if(S_AXI_AWVALID)begin
            axist<=4'b0001;
            wb_adr_i[5:2]<=S_AXI_AWADDR[5:2];
         end else if(S_AXI_WVALID)begin
            axist<=4'b0010;
            wb_dat_i<=S_AXI_WDATA;
         end else if(S_AXI_ARVALID)begin
            axist<=4'b0100;
         end
      end else if(axist==4'b0001)begin
         if(S_AXI_WVALID)begin
            axist<=4'b0011;
            wb_dat_i<=S_AXI_WDATA;
         end
      end else if(axist==4'b0010)begin
         if(S_AXI_AWVALID)begin
            axist<=4'b0011;
            wb_adr_i[5:2]<=S_AXI_AWADDR[5:2];
         end
      end else if(axist==4'b0011)begin
         if(S_AXI_BREADY)
           axist<=4'b0000;
      end else if(axist==4'b0100)begin
         if(S_AXI_RREADY)
           axist<=4'b0000;
      end
   end

   wire        read  = S_AXI_ARVALID & S_AXI_ARREADY;
   wire        write = (axist==4'b0011) & S_AXI_BREADY;
   
   always @(posedge S_AXI_ACLK)begin
      if(~S_AXI_ARESETN)begin
         S_AXI_RDATA <= 32'h0;
      end else if(read)begin
         case(S_AXI_ARADDR[5:2])
           4'd0 : S_AXI_RDATA <= {src_ready, 24'h0,
                                  last, deltaw, backprop, enbias, run, wwrite, bwrite};
           4'd1 : S_AXI_RDATA <= {22'h0,fs[9:0]};
           4'd2 : S_AXI_RDATA <= {22'h0,ks[9:0]};
           4'd3 : S_AXI_RDATA <= {27'h0,kh[4:0]};
           4'd4 : S_AXI_RDATA <= {27'h0,kw[4:0]};

           4'd5 : S_AXI_RDATA <= {20'h0,ss[11:0]};
           4'd6 : S_AXI_RDATA <= {28'h0,id[3:0]};
           4'd7 : S_AXI_RDATA <= {22'h0,is[9:0]};
           4'd8 : S_AXI_RDATA <= {27'h0,ih[4:0]};
           4'd9 : S_AXI_RDATA <= {27'h0,iw[4:0]};

           4'd10: S_AXI_RDATA <= {20'h0,ds[11:0]};
           4'd11: S_AXI_RDATA <= {28'h0,od[3:0]};
           4'd12: S_AXI_RDATA <= {22'h0,os[9:0]};
           4'd13: S_AXI_RDATA <= {27'h0,oh[4:0]};
           4'd14: S_AXI_RDATA <= {27'h0,ow[4:0]};

           4'd15: S_AXI_RDATA <= {28'h0,dd[3:0]};
           default: S_AXI_RDATA <= {32'h0};
         endcase
      end
   end

   always @(posedge S_AXI_ACLK)begin
      if(~S_AXI_ARESETN)begin
           backprop <= 0;
           deltaw <= 0;
           enbias <= 0;
           run <= 0;
           wwrite <= 0;
           bwrite <= 0;
           last <= 0;
           fs[9:0] <= 0;
           ks[9:0] <= 0;
           kh[4:0] <= 0;
           kw[4:0] <= 0;

           ss[11:0] <= 0;
           id[3:0] <= 0;
           is[9:0] <= 0;
           ih[4:0] <= 0;
           iw[4:0] <= 0;

           ds[11:0] <= 0;
           od[3:0] <= 0;
           os[9:0] <= 0;
           oh[4:0] <= 0;
           ow[4:0] <= 0;

           dd[3:0] <= 0;
      end else if(write)begin
         case(wb_adr_i[5:2])
           4'd0 : {last, deltaw, backprop, enbias, run, wwrite, bwrite} <= wb_dat_i[6:0];
           4'd1 : fs[9:0] <= wb_dat_i[9:0];
           4'd2 : ks[9:0] <= wb_dat_i[9:0];
           4'd3 : kh[4:0] <= wb_dat_i[4:0];
           4'd4 : kw[4:0] <= wb_dat_i[4:0];

           4'd5 : ss[11:0] <= wb_dat_i[11:0];
           4'd6 : id[3:0] <= wb_dat_i[3:0];
           4'd7 : is[9:0] <= wb_dat_i[9:0];
           4'd8 : ih[4:0] <= wb_dat_i[4:0];
           4'd9 : iw[4:0] <= wb_dat_i[4:0];

           4'd10: ds[11:0] <= wb_dat_i[11:0];
           4'd11: od[3:0] <= wb_dat_i[3:0];
           4'd12: os[9:0] <= wb_dat_i[9:0];
           4'd13: oh[4:0] <= wb_dat_i[4:0];
           4'd14: ow[4:0] <= wb_dat_i[4:0];

           4'd15: dd[3:0] <= wb_dat_i[3:0];
         endcase
      end
   end
endmodule
