# Z-turn で動かすには

## ブロックデザインを作る

[NahiViva](https://github.com/tokuden/NahiViva) で再現できるようにしました。説明は [こっち](http://nahitafu.cocolog-nifty.com/nahitafu/2019/05/post-2cfa5c.html) を見た方が良いかも。  
次のディレクトリ ```Zturn/``` に必要なファイルをダウンロードして、```open_project_gui.cmd``` 実行でプロジェクトが再現されます。

#### 手動でやるなら

1. サンプルデザイン ```mys-xc7z020-trd``` のブロックデザインを開いて Zynq 以外を消す。*
2. Vivado で tiny_dnn アクセラレータのファイル （```src_fpga/tiny_dnn_top.v, tiny_dnn_reg.v``` と ```src_fp/tiny_dnn_buf.sv, tiny_dnn_core.sv, tiny_dnn_pool.sv``` と ```src/tiny_dnn_control.sv,tiny_dnn_ex_ctl.sv, loop_lib.sv``` ）を開く
3. ブロックデザインの中に ```tiny_dnn_top``` を RTLモジュールとして追加する
4. ほかの部品を ```design_1.pdf``` を参考に追加して結線する
5. PL のクロックは 100MHz (METしないけど…)
6. アドレスマップは下記参照

| master | slave module | Start Address | End Address |
| ------ | ------------ | ------------- | ----------- |
| PS7    | tiny_dnn     | 4000_0000     | 4000_FFFF   |
|        | AXI DMA      | 4040_0000     | 4040_FFFF   |
| DMA    | DDR          | 0000_0000     | 3FFF_FFFF   |

*) 付属の DVD に入っていた mys-xc7z020-trd.rar を解凍します。

また、ACP を使うときには AxCACHE を 1111 or 1110 にする必要があるようなので ```Constant IP``` を使って 1111 を入れています。  
詳しい話は [ここ](https://qiita.com/ikwzm/items/b2ee2e2ade0806a9ec07) が参考になります。  
あと、PL の設定で ```Tie off AxUSER``` にチェックを入れています。

## Petalinux を作る

Vivado でビットストリーム込みの hdf ファイルをエクスポート、```tiny-dnn/mys-xc7z020-trd.sdk```にコピーして、

```
$ source /opt/pkg/petalinux/2019.1/settings.sh
$ petalinux-create --type project --template zynq --name tiny-dnn
$ cd tiny-dnn/
$ petalinux-config --get-hw-description=./mys-xc7z020-trd.sdk
```

menuconfig の画面で ```Image Packaging Configuration ->  Root filesystem type -> SD card``` を選択する。

```
$ petalinux-config -c rootfs
```

menuconfig の画面で ```Filesystem Packages -> misc -> gcc-runtime -> libstdc++``` を選択する。

DMA 転送に使うバッファ用に [udmabuf](https://github.com/ikwzm/udmabuf/blob/master/Readme.ja.md) を作る。

```
$ petalinux-create -t modules --name udmabuf --enable
$ petalinux-build -c rootfs
```

ダウンロードしたファイルで ```project-spec/meta-user/recipes-modules/udmabuf/files/``` を置き換えて、

```
$ petalinux-build -c udmabuf
```

udmabuf の設定をして、DMA と tiny-dnn アクセラレータのレジスタ空間を uio にする。  
DMA に ```dma-coherent``` を設定する。  
デバイスツリーに ```dma-coherent``` 付きで udmabuf を追加する。  
具体的には ```Zturn/system-user.dtsi``` で ```project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi``` を上書きして、

```
$ petalinux-build
```

続けて、

```
$ petalinux-package --boot --force --fsbl images/linux/zynq_fsbl.elf --fpga images/linux/system.bit --u-boot
```

生成物は ```images/linux/BOOT.bin, image.ub, rootfs.ext4``` です。

BOOT.bin,  image.ub を SDカード(FAT32) にコピーする。

```
$ cp images/linux/BOOT.bin /media/tom01h/BOOT
$ cp images/linux/image.ub /media/tom01h/BOOT
```

rootfs.ext4 を SDカード(ext4) にコピーする。SD カードをアンマウントして、

```
$ sudo dd if=images/linux/rootfs.ext4 of=/dev/sdb2 bs=16M
$ sudo sync
$ sudo resize2fs /dev/sdb2
$ sudo sync
```

## プログラムをコンパイルする

対象のサンプルプログラムのディレクトリ ```examples/simple-conv,DW-conv ``` でクロスコンパイルします。

**SDK の 2019.1 ではコンパイルできないようです**

CPU のみで実行する場合は

```
$ ${SDK path}/gnu/aarch32/nt/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++.exe -O3 -mfpu=neon -mtune=cortex-a9 -mcpu=cortex-a9 -mfloat-abi=hard -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -DNDEBUG -std=gnu++14 -I ../../src_c/soft -I ../../ -DDNN_USE_IMAGE_API train.cpp -o train
```

アクセラレータを使って実行する場合は

```
$ ${SDK path}/gnu/aarch32/nt/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++.exe -O3 -mfpu=neon -mtune=cortex-a9 -mcpu=cortex-a9 -mfloat-abi=hard -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -DNDEBUG -std=gnu++14 -I ../../src_c/fpga -I ../../src_c/softfp -I ../../src_c/soft -I ../../ -DDNN_USE_IMAGE_API train_z7.cpp -o train
```

コンパイル済みのソフトと入力データ ```train, data/``` を SD カード(FAT32) にコピーする。

## 実行する

SD カードを挿入して Zynq をブートします。  
ブート後、Zynq の Linux 上で

```
root@tiny-dnn:~# mount /dev/mmcblk0p1 /mnt/
root@tiny-dnn:~# /mnt/train --data_path /mnt/data/ --learning_rate 1 --epochs 1 --minibatch_size 16 --backend_type internal
```
