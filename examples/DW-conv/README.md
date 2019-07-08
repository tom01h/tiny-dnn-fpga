# Separable 畳み込みニューラルネット

ネットモデルはこんな感じ。  
真ん中の2個の conv3x3 は Depthwise Separable Convolution です。

![](model.svg)

学習データを20000に削っています。

```
  nn << conv(28, 28, 5, 1, 8, padding::valid, true, 1, 1, 1, 1, backend_type)
     << max_pool(24, 24, 8, 2)
     << relu()
     << dwconv(12, 12, 3, 8, padding::valid, true, 1, 1, 1, 1, backend_type)
     << relu()
     << conv(10, 10, 1, 8, 8, padding::valid, true, 1, 1, 1, 1, backend_type)
     << relu()
     << dwconv(10, 10, 3, 8, padding::valid, true, 1, 1, 1, 1, backend_type)
     << relu()
     << conv(8, 8, 1, 8, 16, padding::valid, true, 1, 1, 1, 1, backend_type)
     << max_pool(8, 8, 16, 2)
     << relu()
     << conv(4, 4, 4, 16, 10, padding::valid, true, 1, 1, 1, 1, backend_type)
     << softmax(10);
```

## 検証環境

### ホスト CPU で実行

ミシュレ―ションは7秒くらいかかります。

```
$ g++ -pthread -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -DNDEBUG -std=gnu++14  -I ../../src_c/soft -I ../../ -DDNN_USE_IMAGE_API train.cpp -o train
$ ./train --data_path ../../data/ --learning_rate 1 --epochs 1 --minibatch_size 16 --backend_type internal
```

