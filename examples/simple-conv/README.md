# シンプル畳み込みニューラルネット

ネットモデルはこんな感じ。  

![](model.svg)

学習データを20000に削っています。

```
nn << conv(28, 28, 5, 1, 6, padding::valid, true, 1, 1, 1, 1, backend_type)
   << max_pool(24, 24, 6, 2)
   << relu()
   << conv(12, 12, 5, 6, 16, padding::valid, true, 1, 1, 1, 1, backend_type)
   << max_pool(8, 8, 16, 2)
   << relu()
   << conv(4, 4, 4, 16, 10, padding::valid, true, 1, 1, 1, 1, backend_type)
   << softmax(10);
```

## 検証環境

### ホスト CPU で実行

ミシュレ―ションは10秒くらいかかります。

```
$ g++ -pthread -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -DNDEBUG -std=gnu++14  -I ../../src_c/soft -I ../../ -DDNN_USE_IMAGE_API train.cpp -o train
$ ./train --data_path ../../data/ --learning_rate 1 --epochs 1 --minibatch_size 16 --backend_type internal
```

### Verilog シミュレーション

SystemC + Verilator とコラボした協調検証環境(全部手彫り)です。   
学習データを1600まで削っています。

##### ツールのバージョン
- g++ (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0
- Verilator 4.010 2019-01-27 rev UNKNOWN_REV
- SystemC 2.3.3-Accellera

この環境の本体は tiny_dnn です。グローバル変数に Verilator のオブジェクト？を割り当てます。  
FPGA で動かす時に PL にオフロードするところで Verilator のオブジェクト？に値を設定して、クロックと時間を進めて評価します。  
PL にオフロードするところだけを RTL シミュレーションします。  
FPGA にもっていくときは、同じところを PL を操作するように書き換えます。

配列インデックス生成部分は SystemC で書いて HLS しようと思っていたのですが、勝手に無駄サイクルを入れられて思った回路が出来ないので断念しました。  
でもやはり SystemC だとかなり書きやすいので、まずは SystemC で書いて、見比べながら Verilog に翻訳していくのが良いような気がしています。  
tiny_dnn_sc_ctl.cpp を人力で書き換えたのが tiny_dnn_ex_ctl.sv です。  
tiny_dnn_sc_ctl は普段は止まっています。

#### FPU が偽物のシミュレーション (real 型を使って計算)

ちょっとだけ速いです。  
ミシュレ―ションは100秒くらいかかります。

```
$ make
$ sim/Vtiny_dnn_top --data_path ../../data/ --learning_rate 1 --epochs 1 --minibatch_size 2 --backend_type internal
```

#### FPU も RTL で作ってシミュレーション

ミシュレ―ションは160秒くらいかかります。

```
$ make hardfp=1
$ sim/Vtiny_dnn_top --data_path ../../data/ --learning_rate 1 --epochs 1 --minibatch_size 2 --backend_type internal
```

