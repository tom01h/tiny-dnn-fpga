SHELL = /bin/bash

SIM_DIR = sim

SYSTEMC     = /usr/local/systemc-2.3
TARGET_ARCH = linux64
VERILATOR   = verilator

SRC_DIR = ../../src
FP_DIR = ../../src_fp

VERILATOR_OPTS = \
	-Wall \
	-Wno-WIDTH \
	-Wno-UNUSED \
	-Wno-BLKSEQ \
	--cc \
	+1364-2001ext+v \
	-Wno-fatal \
	--Mdir sim \
	--trace \
	-LDFLAGS libsystemc.a \

ifdef hardfp

VERILATOR_MAKE_OPTS = OPT="-Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -DNDEBUG -std=gnu++14 -I ../../../src_c/hardfp/ -I ../../../src_c/softfp/ -I ../../../src_c/soft/ -I ../../../src/ -I ../../../ -I$(SYSTEMC)/include -DDNN_USE_IMAGE_API"

DESIGN_SRCS = $(addprefix $(FP_DIR)/, \
tiny_dnn_top.sv \
tiny_dnn_buf.sv \
tiny_dnn_core.sv \
tiny_dnn_pool.sv \
) \
$(addprefix $(SRC_DIR)/, \
tiny_dnn_control.sv \
tiny_dnn_ex_ctl.sv \
loop_lib.sv \
)

INCLUDE_FILE = \
$(SRC_DIR)/tiny_dnn_sc_ctl.h \
../../src_c/hardfp/tiny_dnn/core/kernels/conv2d_op_internal.h \
../../src_c/softfp/tiny_dnn/core/kernels/dwconv2d_op_internal.h \
../../src_c/hardfp/tiny_dnn/core/kernels/maxpool_op_internal.h \

else

VERILATOR_MAKE_OPTS = OPT="-Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -DNDEBUG -std=gnu++14 -I ../../../src_c/softfp/ -I ../../../src_c/soft/ -I ../../../src/ -I ../../../ -I$(SYSTEMC)/include -DDNN_USE_IMAGE_API"

DESIGN_SRCS = $(addprefix $(SRC_DIR)/, \
tiny_dnn_top.sv \
tiny_dnn_control.sv \
tiny_dnn_buf.sv \
tiny_dnn_core.sv \
tiny_dnn_pool.sv \
tiny_dnn_ex_ctl.sv \
loop_lib.sv \
)

INCLUDE_FILE = \
$(SRC_DIR)/tiny_dnn_sc_ctl.h \
../../src_c/softfp/tiny_dnn/core/kernels/conv2d_op_internal.h \
../../src_c/softfp/tiny_dnn/core/kernels/dwconv2d_op_internal.h \
../../src_c/softfp/tiny_dnn/core/kernels/maxpool_op_internal.h \

endif

VERILATOR_CPP_TB = \
$(SRC_DIR)/tiny_dnn_sc_ctl.cpp \
train_tb.cpp \

default: $(SIM_DIR)/Vtiny_dnn_top

sim: $(SIM_DIR)/Vtiny_dnn_top

$(SIM_DIR)/Vtiny_dnn_top: $(DESIGN_SRCS) $(VERILATOR_CPP_TB) $(INCLUDE_FILE)
	$(VERILATOR) $(VERILATOR_OPTS) $(DESIGN_SRCS) --exe ../$(VERILATOR_CPP_TB)
	cp $(SYSTEMC)/lib-$(TARGET_ARCH)/libsystemc.a sim
	cd sim; make $(VERILATOR_MAKE_OPTS) -f Vtiny_dnn_top.mk Vtiny_dnn_top__ALL.a
	cd sim; make $(VERILATOR_MAKE_OPTS) -f Vtiny_dnn_top.mk Vtiny_dnn_top

clean:
	rm -rf sim/ tmp.vcd tmp.wlf NET-model train

.PHONY:
