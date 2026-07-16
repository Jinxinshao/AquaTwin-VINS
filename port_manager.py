'''
  ******************************************************************************
  * Copyright (c)  2016 - 2025, Wuhan Yesense Co.,Ltd .  http://www.yesense.com
  * @file    Yesense_Decode.py
  * @version V2.0.0
  * @date    2025
  * @author  Yesense Technical Support Team  
  * @brief   decode yesense output data with python3.
  ******************************************************************************    
/*******************************************************************************
*
* 代码许可和免责信息
* 武汉元生创新科技有限公司授予您使用所有编程代码示例的非专属的版权许可，您可以由此
* 生成根据您的特定需要而定制的相似功能。根据不能被排除的任何法定保证，武汉元生创新
* 科技有限公司及其程序开发商和供应商对程序或技术支持（如果有）不提供任何明示或暗
* 含的保证或条件，包括但不限于暗含的有关适销性、适用于某种特定用途和非侵权的保证
* 或条件。
* 无论何种情形，武汉元生创新科技有限公司及其程序开发商或供应商均不对下列各项负责，
* 即使被告知其发生的可能性时，也是如此：数据的丢失或损坏；直接的、特别的、附带的
* 或间接的损害，或任何后果性经济损害；或利润、业务、收入、商誉或预期可节省金额的
* 损失。
* 某些司法辖区不允许对直接的、附带的或后果性的损害有任何的排除或限制，因此某些或
* 全部上述排除或限制可能并不适用于您。
*
*******************************************************************************/
'''
# -*- encoding:utf-8 -*-
#!/usr/bin/env python3

import sys
import serial

##--------------------------------------------------------------------------------------##
##   打开串口函数                                         								##
##   输入：port -- 待打开的串口号，字符串，baud -- 待打开串口的波特率，整数             ##
##   返回值：串口实例                                                                	##
##--------------------------------------------------------------------------------------##	
def open_port(port, baudrate):
	'''open serial port with given parameters'''
	try:
		ser = serial.Serial(port, int(baudrate), timeout = 1)
		print(f"serial {ser.name} opened @ baud rate {baudrate}")
		ser.flush()
		return ser
	except serial.SerialException as e:
		print(f"Error: Could not open serial port {port}. {e}")
		sys.exit(1)
		
def close_port(ser_ins):
	ser_ins.close()

def rd_data(ser_ins):
	return ser_ins.read_all()

def wr_data(ser_ins, data):
	if ser_ins.is_open():
		ser_ins.write(data)