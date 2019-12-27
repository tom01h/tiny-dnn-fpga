# Ultra96 で動かすには

## ブロックデザインを作る

[NahiViva](https://github.com/tokuden/NahiViva) で再現できるようにしました。説明は [こっち](http://nahitafu.cocolog-nifty.com/nahitafu/2019/05/post-2cfa5c.html) を見た方が良いかも。  
次のディレクトリ ```U96/``` に必要なファイルをダウンロードして、```open_project_gui.cmd``` 実行でプロジェクトが再現されます。

#### 手動でやるなら

1. サンプルデザイン ```xilinx-ultra96-reva-v2018.2-final.bsp``` のブロックデザインを開いて ZynqMP 以外を消す。※
2. Vivado で tiny_dnn アクセラレータのファイル （```src_fpga/tiny_dnn_top.v, tiny_dnn_reg.v``` と ```src_fp/tiny_dnn_buf.sv, tiny_dnn_core.sv, tiny_dnn_pool.sv``` と ```src/tiny_dnn_control.sv,tiny_dnn_ex_ctl.sv, loop_lib.sv``` ）を開く
3. ブロックデザインの中に ```tiny_dnn_top``` を RTLモジュールとして追加する
4. ほかの部品を ```design_1.pdf``` を参考に追加して結線する
5. PL のクロックは 100MHz
6. アドレスマップは下記参照

| master | slave module | Start Address | End Address |
| ------ | ------------ | ------------- | ----------- |
| PS     | top          | a000_0000     | a000_0FFF   |
|        | AXI DMA      | a040_0000     | a040_0FFF   |
| DMA    | DDR          | 0000_0000     | 7FFF_FFFF   |

ACP 周りで Critical Warning 出るけど、良く分からないので放置しています。

```
[BD 41-1629] </zynq_ultra_ps_e_0/SAXIGP0/HPC0_LPS_OCM> is excluded from all addressable master spaces.
```

また、ACP を使うときには AxCACHE に 1111 を、AxPROT に 010 を設定するために ```Constant IP``` を使っています。

※) bsp は tar.gz 形式なので以下で解くことができます。  
NTFS はフルパスで 260 文字の制限があるようなので注意してください。

```
$ tar xvzf xilinx-ultra96-reva-v2018.2-final.bsp xilinx-ultra96-reva-2018.2/hardware/
```



## Petalinux を作る

Vivado でビットストリーム込みの xsa ファイルをエクスポート、```project_1``` をコピーして、

```
$ source /opt/pkg/petalinux/settings.sh
$ petalinux-create -t project -n tiny-dnn --template zynqMP
$ cd petamp
$ petalinux-config --get-hw-description=./project_1
```

menuconfig の画面で

- Subsystem AUTO Hardware Settings → Serial Settings → Primary stdin/stdout → (psu_uart_1)
- DTG Settings → MACHINE_NAME → (avnet-ultra96-rev1)
- u-boot Configuration → u-boot config target → (avnet_ultra96_rev1_defconfig)
- Image Packaging Configuration → Root filesystem type → EXT(SD...)
- Yocto Settings → YOCTO_MACHINE_NAME → (ultra96-zynqmp)

```
$ petalinux-config -c rootfs
```

- Filesystem Packages → base → busybox → busybox-udhcpc
- Filesystem Packages → network → wpa_supplicant → wpa_supplicant
- Filesystem Packages -> misc -> gcc-runtime -> libstdc++

この後、ツールのバグ？対応のため、何度かファイルの編集をします。せっかく編集しても、ツールに元に戻されちゃうからなのですが、正式な対応方法がわからないため…

次のファイルから ```/include/ "mipi-support-ultra96.dtsi"``` を削除する。

```
components/plnx_workspace/device-tree/device-tree/system-top.dts
```

コマンド ```petalinux-config -c u-boot-xlnx``` を実行後に、次のファイルに ```CONFIG_NET=y``` を追加する。

```
components/plnx_workspace/sources/u-boot-xlnx/.config.new
```

参考にしたのは、[ここ](https://github.com/Avnet/Ultra96-PYNQ) と [ここ](https://forums.xilinx.com/t5/Embedded-Linux/petalinux2019-2-u-boot-compile-error-for-ultra96-board/td-p/1039492) 。components/ の下は自動生成されるファイルなので、この2つはエラーが出るたびに何度かやることになります。

つぎに、DMA 転送に使うバッファ用に [udmabuf](https://github.com/ikwzm/udmabuf/blob/master/Readme.ja.md) を作る。

```
$ petalinux-create -t modules --name udmabuf --enable
$ petalinux-build -c rootfs
```

ダウンロードしたファイルで ```project-spec/meta-user/recipes-modules/udmabuf/files/``` を置き換えて、

```
$ petalinux-build -c udmabuf
```

続けて、udmabuf の設定をして、DMA と mem のアドレス空間を uio にする。  
DMA に ```dma-coherent``` を設定する。  
デバイスツリーに ```dma-coherent``` 付きで udmabuf を追加する。  
具体的には ```U96/system-user.dtsi``` で ```project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi``` を上書きして、

```
$ petalinux-build
```

また DMA でキャッシュを有効にする為に ```FF41A040=3``` を設定する必要があるようなので ```U96/regs.init``` を使って、

```
$ petalinux-package --boot --force --fsbl images/linux/zynqmp_fsbl.elf --fpga images/linux/system.bit --pmufw images/linux/pmufw.elf --bif-attribute init --bif-attribute-value ../regs.init --u-boot
```

生成物は ```images/linux/BOOT.bin, image.ub, rootfs.tar.gz``` です。

BOOT.bin,  image.ub を SDカード(FAT32) にコピーする。

```
$ cp images/linux/BOOT.bin /media/tom01h/BOOT
$ cp images/linux/image.ub /media/tom01h/BOOT
```

rootfs.tar.gz を SDカード(ext4) にコピーする。

```
$ sudo tar xvf images/linux/rootfs.tar.gz -C /media/tom01h/${mount_point}
$ sudo sync
```

## プログラムをコンパイルする

対象のサンプルプログラムのディレクトリ ```examples/simple-conv,DW-conv ``` でクロスコンパイルします。

まずはクロスコンパイラを準備します。

```
$ sudo apt install g++-aarch64-linux-gnu
```

ちなみに SDK 2019.1 の gcc は gcc-8 なのでコンパイルできないようです。

CPU のみで実行する場合は

```
$ aarch64-linux-gnu-g++ -O3 -mtune=cortex-a53 -mcpu=cortex-a53 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -DNDEBUG -std=gnu++14 -I ../../src_c/soft -I ../../ -DDNN_USE_IMAGE_API train.cpp -o train
```

アクセラレータを使って実行する場合は

```
$ aarch64-linux-gnu-g++ -O3 -mtune=cortex-a53 -mcpu=cortex-a53 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -DNDEBUG -std=gnu++14 -I ../../src_c/fpga -I ../../src_c/softfp -I ../../src_c/soft -I ../../ -DDNN_USE_IMAGE_API train_mp.cpp -o train
```

コンパイル済みのソフトと入力データ ```train, data/``` を SD カード(FAT32) にコピーする。

## 実行する

SD カードを挿入して Zynq をブートします。

login,password 共に root でログインできます。

```
root@tiny-dnn:~# mount /dev/mmcblk0p1 /mnt/
root@tiny-dnn:~# /mnt/train --data_path /mnt/data/ --learning_rate 1 --epochs 1 --minibatch_size 16 --backend_type internal
```

#### Wifi につなぐ

wpa_passphrase コマンドに SSID と PASS を指定して実行した画面出力を、 /etc/wpa_supplicant/wpa_supplicant.conf に追加。

/etc/network/interfaces に、auto wlan0 を追加。

リブート。

ファイルのコピーに SCP が使えるようになりました。

```
$ scp test root@192.168.0.XXX:test
```

