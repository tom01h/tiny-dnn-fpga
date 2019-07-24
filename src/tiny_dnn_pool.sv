module tiny_dnn_pool
  (
   input wire        clk,
   input wire        pool,
   input wire        p_fin,
   input wire        en,
   input wire [4:0]  ow,
   input real        d0,
   input real        d1,
   input real        d2,
   input real        d3,
   output wire       pool_busy,
   output real       po,
   output reg [15:0] pp
   );

   reg               en1, en2, en3, en4;
   reg [11:0]        pa1, pa2, pa3, pa4;
   real              o10, o11, o12, o13;
   real              o20, o21, o3, o4;
   reg [1:0]         p20, p21, p3, p4;

   assign pool_busy = en1| en2| en3| en4;

   reg [4:0]         px;
   reg [15:1]        pa;

   always_ff @(posedge clk)begin
      if(~pool|p_fin)begin
         pa <= 15'd0;
         px <= 5'd0;
      end else if(en)begin
         if((px+1) == ow)begin
            pa <= pa + 1 + ow;
            px <= 0;
         end else begin
            pa <= pa + 1;
            px <= px + 1;
         end
      end
   end

   always_ff @(posedge clk)begin
      if(~pool)begin
         en1 <= 1'b0;
         en2 <= 1'b0;
         en3 <= 1'b0;
         en4 <= 1'b0;
      end else begin
         en1 <= en;
         en2 <= en1;
         en3 <= en2;
         en4 <= en3;
      end
      if(en)begin
         o10 <= d0;
         o11 <= d1;
         o12 <= d2;
         o13 <= d3;
         pa1 <= pa;
      end
      if(en1)begin
         pa2 <= pa1;
         if(o10>=o11)begin
            o20 <= o10;
            p20 <= 2'd0;
         end else begin
            o20 <= o11;
            p20 <= 2'd2;
         end
         if(o12>=o13)begin
            o21 <= o12;
            p21 <= 2'd1;
         end else begin
            o21 <= o13;
            p21 <= 2'd3;
         end
      end
      if(en2)begin
         pa3 <= pa2;
         if(o20>=o21)begin
            o3 <= o20;
            p3 <= p20;
         end else begin
            o3 <= o21;
            p3 <= p21;
         end
      end
      if(en3)begin
         pa4 <= pa3;
         o4 <= o3;
         p4 <= p3;
      end
      if(en4)begin
         po <= o4;
         if(p4[1])
           pp <= {pa4+ow,p4[0]};
         else
           pp <= {pa4,p4[0]};
      end
   end


endmodule
