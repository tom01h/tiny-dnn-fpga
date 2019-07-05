module normalize
  (
   input wire               clk,
   input wire               en,
   input wire               signo,
   input wire signed [9:0]  expo,
   input wire signed [31:0] addo,
   output reg [31:0]        nrm
   );

   reg [31:0]         nrm5, nrm4, nrm3, nrm2, nrm1, nrm0;
   reg signed [9:0]   expn, expl;
   reg [9:0]          expd;
   reg                sign, signl;

   always_comb begin
      if(addo<0)begin
         nrm5[31:0]=-addo;
         sign=~signo;
      end else begin
         nrm5[31:0]=addo;
         sign=signo;
      end

      if(nrm5[31:16]!=0)begin
         nrm4=nrm5[31:0];
         expd[4]=0;
      end else begin
         nrm4={nrm5[15:0],16'h0000};
         expd[4]=1;
      end

      if(nrm4[31:24]!=0)begin
         expd[3]=0;
      end else begin
         expd[3]=1;
      end
   end

   always_ff @(posedge clk)begin
      if(en)begin
         signl <= sign;
         expl <= expo - {1'b0,expd[4:3],3'b000};
         if(nrm4[31:24]!=0)begin
            nrm3<=nrm4[31:0];
         end else begin
            nrm3<={nrm4[23:0],8'h00};
         end
      end
   end

   always_comb begin
      if(nrm3[31:28]!=0)begin
         nrm2=nrm3[31:0];
         expd[2]=0;
      end else begin
         nrm2={nrm3[27:0],4'h0};
         expd[2]=1;
      end

      if(nrm2[31:30]!=0)begin
         nrm1=nrm2[31:0];
         expd[1]=0;
      end else begin
         nrm1={nrm2[29:0],2'b00};
         expd[1]=1;
      end

      if(nrm1[31])begin
         nrm0=nrm1[31:0];
         expd[0]=0;
      end else begin
         nrm0={nrm1[30:0],1'b0};
         expd[0]=1;
      end

      expn = expl-{1'b0,expd[2:0]}+17-127;

      if(expn<=0)begin
         nrm = 0;
      end else begin
         nrm[31]    = signl;
         nrm[30:23] = expn;
         nrm[22:0]  = nrm0[30:8];
      end
   end

endmodule

module tiny_dnn_core
  (
   input wire                clk,
   input wire                init,
   input wire                write,
   input wire                bwrite,
   input wire                exec,
   input wire                outr,
   input wire                update,
   input wire                bias,
   input wire [10:0]         ra,
   input wire [10:0]         wa,
   input wire [15:0]         d, // bfloat16
   input wire [15:0]         wd, // bfloat16
   input wire                signi,
   input wire signed [9:0]   expi,
   input wire signed [31:0]  addi,
   output wire               signo,
   output wire signed [9:0]  expo,
   output wire signed [31:0] addo
   );

   parameter f_size = 1024;

   reg [15:0]         WM0 [0:f_size-1];
   reg [15:0]         WM1 [0:f_size-1];

   reg                init1, exec1, bias1, init2, exec2, bias2;
   reg                ra10d;
   reg [15:0]         W10, W11, W2;
   reg [15:0]         d2;
   

   always_ff @(posedge clk)begin
      init1 <= init;
      exec1 <= exec;
      bias1 <= bias;
      init2 <= init1;
      exec2 <= exec1;
      bias2 <= bias1;
      ra10d <= ra[10];
   end

   wire [9:0]     biasa = f_size-1;
   wire [10:0]    radr = (bias)   ? {ra[10],biasa} : ra ;
   wire [10:0]    wadr = (bwrite) ? {wa[10],biasa} : wa ;

   always_ff @(posedge clk)
     if(write&~wadr[10])begin
        WM0[wadr[9:0]] <= wd;
     end else if((exec|bias)&~radr[10])begin
        W10 <= WM0[radr[9:0]];
     end

   always_ff @(posedge clk)
     if(write&wadr[10])begin
        WM1[wadr[9:0]] <= wd;
     end else if((exec|bias)&radr[10])begin
        W11 <= WM1[radr[9:0]];
     end


   always_ff @(posedge clk)
     if(exec1|bias1)begin
        if(ra10d)begin
           W2 <= W11;
        end else begin
           W2 <= W10;
        end
        d2 <= (exec1) ? d : 16'h3f80;
     end

   wire                       signl;
   wire signed [9:0]          expl;
   wire signed [31:0]         addl;
   reg                        signt;
   reg signed [9:0]           expt;
   reg signed [31:0]          addt;

   assign signo = (update) ? signl : signt;
   assign expo  = (update) ? expl  : expt;
   assign addo  = (update) ? addl  : addt;

   always_ff @(posedge clk)
     if(outr)begin
        signt <= signi;
        expt <= expi;
        addt <= addi;
     end

   fma fma
     (
      .clk(clk),
      .init(init2),
      .exec(exec2|bias2),
      .update(update),
      .w(W2[15:0]),
      .d(d2[15:0]),
      .signo(signl),
      .expo(expl[9:0]),
      .addo(addl[31:0])
   );

endmodule

module fma
  (
   input wire               clk,
   input wire               init,
   input wire               exec,
   input wire               update,
   input wire [15:0]        w,
   input wire [15:0]        d,
   output reg               signo,
   output reg signed [9:0]  expo,
   output reg signed [31:0] addo
   );

   reg signed [16:0]         frac;
   reg signed [9:0]          expm;
   reg signed [9:0]          expd;
   reg signed [32:0]         add0;
   reg signed [48:0]         alin;
   reg                       sftout;

   always_comb begin
      frac = {9'h1,w[6:0]}  * {9'h1,d[6:0]};
      expm = {1'b0,w[14:7]} + {1'b0,d[14:7]};
      expd = expm - expo + 16;

      if(signo^w[15]^d[15])
        add0 = -addo;
      else
        add0 = addo;

      if(expd[9:6]!=0)
        alin = 0;
      else
        alin = $signed({add0,16'h0})>>>expd[5:0];

      sftout = (expd<0) | (alin[48:30]!={19{1'b0}}) & (alin[48:30]!={19{1'b1}});
   end

   always_ff @(posedge clk)begin
      if(init)begin
         signo <= 0;
         expo <= 0;
         addo <= 0;
      end else if(exec&!sftout)begin
         signo <= w[15]^d[15];
         expo <= expm;
         addo <= frac + alin;
      end
   end

endmodule
