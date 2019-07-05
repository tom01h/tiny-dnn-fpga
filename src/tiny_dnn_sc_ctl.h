#include <systemc.h>

SC_MODULE(tiny_dnn_sc_ctl)
{
  //Ports
  sc_in  <bool>         clk;
  sc_in  <bool>         backprop;
  sc_in  <bool>         run;
  sc_in  <bool>         wwrite;
  sc_in  <bool>         bwrite;
  sc_in  <bool>         s_init;
  sc_in  <bool>         out_busy;
  sc_in  <bool>         outrf;
  sc_out <bool>         s_fin;
  sc_out <bool>         k_init;
  sc_out <bool>         k_fin;
  sc_out <bool>         exec;
  sc_out <sc_uint<12> > ia;
  sc_out <sc_uint<10> > wa;

  sc_in <sc_uint<4> >   dd;
  sc_in <sc_uint<4> >   id;
  sc_in <sc_uint<10> >  is;
  sc_in <sc_uint<5> >   ih;
  sc_in <sc_uint<5> >   iw;
  sc_in <sc_uint<4> >   od;
  sc_in <sc_uint<10> >  os;
  sc_in <sc_uint<5> >   oh;
  sc_in <sc_uint<5> >   ow;
  sc_in <sc_uint<10> >  fs;
  sc_in <sc_uint<10> >  ks;
  sc_in <sc_uint<5> >   kh;
  sc_in <sc_uint<5> >   kw;

  //Thread Declaration
  void exect();

  //Constructor
  SC_CTOR(tiny_dnn_sc_ctl)
  {
    SC_CTHREAD(exect,clk.pos());

  }
};
