module tiny_dnn_core
  (
   input wire        clk,
   input wire        init,
   input wire        write,
   input wire        bwrite,
   input wire        exec,
   input wire        outr,
   input wire        update,
   input wire        bias,
   input wire [10:0] ra,
   input wire [10:0] wa,
   input real        d,
   input real        wd,
   input real        sum_in,
   output real       sum
   );

   parameter f_size = 1024;

   real          W0 [0:f_size-1];
   real          W1 [0:f_size-1];
   real          w00, w01, w1, d1;
   reg           init1,exec1,bias1, init2,exec2,bias2;
   reg           ra10d;

   wire [9:0]    biasa = f_size-1;
   wire [10:0]   radr = (bias)   ? {ra[10],biasa} : ra ;
   wire [10:0]   wadr = (bwrite) ? {wa[10],biasa} : wa ;

   real          sumt, suml;

   assign sum  = (update) ? suml  : sumt;

   always_ff @(posedge clk)begin
      if(write&~wadr[10])begin
         W0[wadr[9:0]] <= wd;
      end else if((exec|bias)&~radr[10])begin
         w00 <= W0[radr[9:0]];
      end
      if(write&wadr[10])begin
         W1[wadr[9:0]] <= wd;
      end else if((exec|bias)&radr[10])begin
         w01 <= W1[radr[9:0]];
      end

      if(exec1|bias1)begin
         if(ra10d)begin
            w1 <= w01;
         end else begin
            w1 <= w00;
         end
         d1 <= d;
      end
      init1 <= init;
      exec1 <= exec;
      bias1 <= bias;
      init2 <= init1;
      exec2 <= exec1;
      bias2 <= bias1;
      ra10d <= ra[10];
      if(init2)begin
         suml <= 0;
      end else if(exec2)begin
         suml <= suml + w1 * d1;
      end else if(bias2)begin
         suml <= suml + w1;
      end
      if(outr)begin
         sumt <= sum_in;
      end
   end

endmodule
