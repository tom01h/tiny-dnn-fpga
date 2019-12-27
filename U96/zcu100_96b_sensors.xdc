set_property IOSTANDARD LVCMOS18 [get_ports UART*]
#set_property IOSTANDARD LVCMOS18 [get_ports loopback_out*]

#HD_GPIO_5 on FPGA / Connector pin 13 / UART1_rxd
set_property PACKAGE_PIN G5 [get_ports UART1_rxd]
#HD_GPIO_4 on FPGA / Connector pin 11 / UART1_txd
set_property PACKAGE_PIN F6 [get_ports UART1_txd]

set_property IOSTANDARD LVCMOS18 [get_ports BT*]

#BT_HCI_RTS on FPGA /  emio_uart0_ctsn connect to 
set_property PACKAGE_PIN B7 [get_ports BT_ctsn]
#BT_HCI_CTS on FPGA / emio_uart0_rtsn
set_property PACKAGE_PIN B5 [get_ports BT_rtsn]
