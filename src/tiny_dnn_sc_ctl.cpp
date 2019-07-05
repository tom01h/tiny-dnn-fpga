#include "tiny_dnn_sc_ctl.h"

void tiny_dnn_sc_ctl::exect()
{
  //Initialization
  s_fin.write(0);
  k_init.write(0);
  k_fin.write(0);
  exec.write(0);
  ia.write(0);
  wa.write(0);

  wait();

  while(true){
    wait();
    if(s_init.read()){
      for(int dc=0; dc<=dd.read(); dc++){
        for(int iy=0; iy<=oh.read(); iy++){
          int yy, sy, ey;
          if(backprop.read()){
            yy = iy-kh.read();
            sy = (yy<0) ? -yy : 0;
            if(iy>ih.read()){ ey=ih.read()-yy; }else{ ey=kh.read(); }
          }else{
            yy=iy;
            sy=0;
            ey=kh.read();
          }
          for(int ix=0; ix<=ow.read(); ix++){
            int xx, sx, ex;
            if(backprop.read()){
              xx = ix-kw.read();
              sx = (xx<0) ? -xx : 0;
              if(ix>iw.read()){ ex=iw.read()-xx; }else{ ex=kw.read(); }
            }else{
              xx=ix;
              sx=0;
              ex=kw.read();
            }
            k_init.write(1);
            wait();
            k_init.write(0);
            for(int ic=0; ic<=id.read(); ic++){
              for(int fy=sy; fy<=ey; fy++){
                for(int fx=sx; fx<=ex; fx++){
                  int i=dc*is.read()+ic*is.read()+
                    (yy+fy)*(iw.read()+1)+(xx+fx);
                  int w = ic*(ks.read()+1)+
                    fy*(kw.read()+1)+fx;

                  ia.write(i);
                  wa.write(w);
                  exec.write(1);
                  wait();
                }
              }
            }
            exec.write(0);
            k_fin.write(1);
            wait();
            k_fin.write(0);
            while(out_busy.read()){
              wait();
            }
          }
        }
      }
      while(!outrf.read()){
        wait();
      }
      s_fin.write(1);
      wait();
      s_fin.write(0);
    }
  }
}
