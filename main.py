'''
  ******************************************************************************
  * Copyright (c)  2016 - 2025, Wuhan Yesense Co.,Ltd .  http://www.yesense.com
  * @file    main.py
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

''' 
// =============================================================================
// =============================================================================
								程序基本说明
								
	python脚本运行示例，python3 main.py -h该命令可以查找如何使用该脚本
	python3 main.py --port COM6 --bps 460800 --dbg TRUE
	--port选项是必须的，在windows下为COMxx，在linux下为/dev/ttyUSBx或/dev/ttySCx或/dev/ttyACMx
	--bps选项为可选，默认值为460800
	--dbg选项为可选，默认值为FALSE
// =============================================================================
// =============================================================================
'''	

import sys
import os
import argparse
import time
import struct
from port_manager import *
from yis_std_dec import *

# variable to save decoded result
yis_out = {'tid':1, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
			'q0':1.0, 'q1':1.0, 'q2':0.0, 'q3':0.0, \
			'sensor_temp':25.0, 'acc_x':0.0, 'acc_y':0.0, 'acc_z':1.0, \
			'gyro_x':0.0, 'gyro_y':0.0, 'gyro_z':0.0,	\
			'norm_mag_x':0.0, 'norm_mag_y':0.0, 'norm_mag_z':0.0,	\
			'raw_mag_x':0.0, 'raw_mag_y':0.0, 'raw_mag_z':0.0,	\
			'lat':0.0, 'longt':0.0, 'alt':0.0,	\
			'vel_e':0.0, 'vel_n':0.0, 'vel_u':0.0, \
			'ms':0, 'year': 2022, 'month':8, 'day': 31, \
			'hour':12, 'minute':0, 'second':0,	\
			'smp_timestamp':0, 'ready_timestamp':0 ,'status':0
			}

#解析缓冲区和解析缓冲区数据长度
dec_buf = bytearray()
			
def check_version():
    '''To make sure the Python version matches at least 3.8'''
    print("Current Python version is {}.{}.{}.".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
    if sys.version_info < (3, 8):
        print("Required Python 3.8 or higher.")
        sys.exit(1)

def check_port_name(port):
	''' Serial port name of Linux Os is /dev/ttyxyz'''
	''' Serial port name of Windows Os is COMx'''		
	if 'posix' == os.name:
		if -1 == port.find("/dev/tty"):
			print('port name error')
	elif 'nt' == os.name:
		if -1 == port.find("COM"):
			print('port name error')
	else:
		print('unknown os')
	
def main_func(port, baudrate, dbg_flg):
	'''YESENSE Python Decoder Version V2.0.0'''
	global dec_buf
	
	check_version()
	if dbg_flg:
		print(port)
		print(baudrate)
	check_port_name(port)
	decoder = std_decoder()
	ser = open_port(port, baudrate)
	
	while True:
		data = rd_data(ser)
		num = len(data)
		if(num > 0):
			#更新本次读取的数据到解析缓冲区
			dec_buf.extend(bytearray(data))	
			if dbg_flg:
				print(f'rd len {num}, total len {len(dec_buf)}')
				decoder.hex_show(dec_buf, len(dec_buf))
				
		if len(dec_buf) > 0:
			ret = decoder.proc_data(dec_buf, len(dec_buf), yis_out, dbg_flg)
			if True == ret:
				''' 示例输出的内容为帧序号、欧拉角、加速度、角速度 ''' 
				print('tid %d, pitch %f, roll %f, yaw %f, acc_x %f, acc_y %f, acc_z %f, gyro_x %f, gyro_y %f, gyro_z %f' \
				%(yis_out['tid'], yis_out['pitch'], yis_out['roll'], yis_out['yaw'], \
				yis_out['acc_x'], yis_out['acc_y'], yis_out['acc_z'],\
				yis_out['gyro_x'], yis_out['gyro_y'], yis_out['gyro_z']))	
				
				''' 示例输出的内容为帧序号、欧拉角、四元数、加速度、角速度 '''
				'''
				print('tid %d, pitch %f, roll %f, yaw %f, q0 %f, q1 %f, q2 %f, q3 %f, acc_x %f, acc_y %f, acc_z %f, gyro_x %f, gyro_y %f, gyro_z %f' \
				%(yis_out['tid'], yis_out['pitch'], yis_out['roll'], yis_out['yaw'], \
				yis_out['q0'], yis_out['q1'], yis_out['q2'], yis_out['q3'],\
				yis_out['acc_x'], yis_out['acc_y'], yis_out['acc_z'],\
				yis_out['gyro_x'], yis_out['gyro_y'], yis_out['gyro_z']))				
				'''
		time.sleep(0.001)  

	close_port(ser)
	
#windows下为comxx，linux下为/dev/ttySCx或/dev/ttyUSBx
if __name__ == '__main__':
	parse = argparse.ArgumentParser(description='YESENSE python decoder!')
	parse.add_argument('--port', type=str, required=True, help='The name of serial port to be connected (e.g., COM3 or /dev/ttyUSB0)')
	parse.add_argument('--bps', type=int, default='460800', help='The baud rate for the serial connection (default: 460800)')	
	parse.add_argument('--dbg', action='store_true' , help='Enable logging for debugging  --dbg ')		
	args = parse.parse_args()
	print(args)
	main_func(args.port, args.bps, args.dbg)
