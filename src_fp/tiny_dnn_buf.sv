module src_buf
  (
   input wire        clk,
   input wire        src_v,
   input wire [12:0] src_a,
   input wire [15:0] src_d0,
   input wire [15:0] src_d1,
   input wire [15:0] src_d2,
   input wire [15:0] src_d3,
   input wire        exec,
   input wire [12:0] ia,
   output reg [15:0] d
   );

   reg [15:0]         buff00 [0:1023];
   reg [15:0]         buff01 [0:1023];
   reg [15:0]         buff02 [0:1023];
   reg [15:0]         buff03 [0:1023];
   reg [15:0]         buff10 [0:1023];
   reg [15:0]         buff11 [0:1023];
   reg [15:0]         buff12 [0:1023];
   reg [15:0]         buff13 [0:1023];

   reg [15:0]         d00, d01, d02, d03;
   reg [15:0]         d10, d11, d12, d13;

   reg [2:0]          ia_l;

   always_ff @(posedge clk)
     if(exec)
       ia_l[2:0] <= {ia[12],ia[1:0]};
   always_comb begin
      case(ia_l[2:0])
        3'd0 : d = d00;
        3'd1 : d = d01;
        3'd2 : d = d02;
        3'd3 : d = d03;
        3'd4 : d = d10;
        3'd5 : d = d11;
        3'd6 : d = d12;
        3'd7 : d = d13;
      endcase
   end

   always_ff @(posedge clk)
     if(    src_v &~src_a[12])               buff00[src_a[9:0]] <= src_d0;
     else if(exec &({ia[12],ia[1:0]}==3'd0)) d00 <= buff00[ia[11:2]];
   always_ff @(posedge clk)
     if(    src_v &~src_a[12])               buff01[src_a[9:0]] <= src_d1;
     else if(exec &({ia[12],ia[1:0]}==3'd1)) d01 <= buff01[ia[11:2]];
   always_ff @(posedge clk)
     if(    src_v &~src_a[12])               buff02[src_a[9:0]] <= src_d2;
     else if(exec &({ia[12],ia[1:0]}==3'd2)) d02 <= buff02[ia[11:2]];
   always_ff @(posedge clk)
     if(    src_v &~src_a[12])               buff03[src_a[9:0]] <= src_d3;
     else if(exec &({ia[12],ia[1:0]}==3'd3)) d03 <= buff03[ia[11:2]];

   always_ff @(posedge clk)
     if(    src_v & src_a[12])               buff10[src_a[9:0]] <= src_d0;
     else if(exec &({ia[12],ia[1:0]}==3'd4)) d10 <= buff10[ia[11:2]];
   always_ff @(posedge clk)
     if(    src_v & src_a[12])               buff11[src_a[9:0]] <= src_d1;
     else if(exec &({ia[12],ia[1:0]}==3'd5)) d11 <= buff11[ia[11:2]];
   always_ff @(posedge clk)
     if(    src_v & src_a[12])               buff12[src_a[9:0]] <= src_d2;
     else if(exec &({ia[12],ia[1:0]}==3'd6)) d12 <= buff12[ia[11:2]];
   always_ff @(posedge clk)
     if(    src_v & src_a[12])               buff13[src_a[9:0]] <= src_d3;
     else if(exec &({ia[12],ia[1:0]}==3'd7)) d13 <= buff13[ia[11:2]];

endmodule

module dst_buf
  (
   input wire               clk,
   input wire               dst_v,
   input wire [12:0]        dst_a,
   output wire [31:0]       dst_d0,
   output wire [31:0]       dst_d1,
   input wire               outr,
   input wire               accr,
   input wire [12:0]        oa,
   input wire               signo,
   input wire signed [9:0]  expo,
   input wire signed [31:0] addo
   );

   reg [31:0]        buff00 [0:2047];
   reg [31:0]        buff01 [0:2047];
   reg [31:0]        buff10 [0:2047];
   reg [31:0]        buff11 [0:2047];

   reg               accr2,    outr4, outr5;
   reg [12:0]        oa2, oa3, oa4,   oa5;

   reg [31:0]        y3;
   wire [31:0]       nrm;

   always_ff @(posedge clk)begin
      accr2 <= accr;
      outr4 <= outr;
      outr5 <= outr4;
      oa2   <= {oa[12],oa[11:0]};
      oa3   <= {oa[12],oa2[11:0]};
      oa4   <= {oa[12],oa3[11:0]};
      oa5   <= oa4;
   end

   reg [31:0]        dst_d00, dst_d01, dst_d10, dst_d11;
   assign dst_d0 = (dst_a[12]) ? dst_d01 : dst_d00;
   assign dst_d1 = (dst_a[12]) ? dst_d11 : dst_d10;

   wire [10:0]      ra = (accr) ? oa[11:1] : dst_a[10:0];

   always_ff @(posedge clk)
     if(outr5&~oa5[12]&~oa5[0])
       buff00[oa5[11:1]] <= nrm;
     else if((accr& oa[12]&~oa[0])|(dst_v&~dst_a[12]))
       dst_d00 <= buff00[ra];

   always_ff @(posedge clk)
     if(outr5&~oa5[12]& oa5[0])
       buff01[oa5[11:1]] <= nrm;
     else if((accr& oa[12]& oa[0])|(dst_v&~dst_a[12]))
       dst_d10 <= buff01[ra];

   always_ff @(posedge clk)
     if(outr5& oa5[12]&~oa5[0])
       buff10[oa5[11:1]] <= nrm;
     else if((accr&~oa[12]&~oa[0])|(dst_v& dst_a[12]))
       dst_d01 <= buff10[ra];

   always_ff @(posedge clk)
     if(outr5& oa5[12]& oa5[0])
       buff11[oa5[11:1]] <= nrm;
     else if((accr&~oa[12]& oa[0])|(dst_v& dst_a[12]))
       dst_d11 <= buff11[ra];

   always_ff @(posedge clk)begin
      if(~accr2)begin
         y3 <= 0;
      end else if(~oa[12])begin
         if(oa2[0]) y3 <= dst_d11;
         else       y3 <= dst_d01;
      end else begin
         if(oa2[0]) y3 <= dst_d10;
         else       y3 <= dst_d00;
      end
   end

   reg signed [24:0]         frac;
   reg signed [9:0]          expd;
   reg signed [32:0]         add0;
   reg signed [48:0]         alin;
   reg                       sftout;

   always_comb begin
      frac = {2'b01,y3[22:0]};
      expd = {1'b0,y3[30:23]} - expo + 127 + 7;

      if(signo^y3[31])
        add0 = -addo;
      else
        add0 = addo;

      if(expd[9:6]!=0)
        alin = 0;
      else
        alin = $signed({add0,16'h0})>>>expd[5:0];

      sftout = (expd<0) | (alin[48:30]!={19{1'b0}}) & (alin[48:30]!={19{1'b1}});
   end

   reg             sign4;
   reg signed [9:0] exp4;
   reg signed [31:0] add4;

   always_ff @(posedge clk)begin
      if(!sftout&(y3[30:23]!=8'h0))begin
         sign4 <= y3[31];
         exp4 <= {1'b0,y3[30:23]}+127-9;
         add4 <= alin + frac;
      end else begin
         sign4 <= signo;
         exp4 <= expo;
         add4 <= addo;
      end
   end

   normalize normalize
     (
      .clk(clk),
      .en(outr4),
      .signo(sign4),
      .expo(exp4),
      .addo(add4),
      .nrm(nrm)
   );
endmodule
