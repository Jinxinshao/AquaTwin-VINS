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

import struct

#little-endian
YIS_HEADER_1ST				= 0x59
YIS_HEADER_2ND 				= 0x53					
YIS_HEADER 					= b'\x59\x53'

#header(2B) + tid(2B) + len(1B) + CK1(1B) + CK2(1B)
PROTOCOL_MIN_LEN 			= 7	

PROTOCOL_TID_LEN 			= 2
PROTOCOL_PAYLOAD_LEN		= 1
PROTOCOL_CHECKSUM_LEN		= 2
PROTOCOL_TID_POS 			= 2
PROTOCOL_PAYLOAD_LEN_POS	= 4
CRC_CALC_START_POS	        = 2
PAYLOAD_POS					= 5

TLV_HEADER_LEN				= 2		#type(1B) + len(1B)

#输出数据的data id定义
sensor_temp_id 				= 0x01	#data_id
acc_id 						= 0x10				
gyro_id 					= 0x20
norm_mag_id					= 0x30
raw_mag_id					= 0x31
euler_id					= 0x40
quaternion_id				= 0x41
utc_id						= 0x50
smp_timestamp_id			= 0x51
ready_timestamp_id			= 0x52
utc_id						= 0x50
location_id					= 0x68
speed_id					= 0x70
status_id					= 0x80

#输出数据的len定义
sensor_temp_len				= 0x02	#data_len
acc_len						= 0x0C				
gyro_len 					= 0x0C
norm_mag_len				= 0x0C
raw_mag_len					= 0x0C
euler_len					= 0x0C
quaternion_len				= 0x10
utc_len						= 0x0B
smp_timestamp_len			= 0x04
ready_timestamp_len			= 0x04
location_len				= 0x14
speed_len					= 0x0C
status_len					= 0x01

data_factor_not_raw_mag 	= 0.000001	#非原始磁力输出数据与真实数据的转换系数
data_factor_raw_mag			= 0.001		#原始磁力输出数据与真实数据的转换系数
data_factor_sensor_temp 	= 0.01
data_factor_high_res_loc 	= 0.0000000001
data_factor_alt				= 0.001
data_factor_speed			= 0.001

class std_decoder:
##--------------------------------------------------------------------------------------##
##   CRC计算函数                                        								##
##   输入：data -- 待计算CRC的数据, len - 待计算CRC数据的长度                           ##
##   返回值：CRC计算结果，占2字节长度                                                   ##
##--------------------------------------------------------------------------------------##
	@staticmethod	
	def calc_crc16(data, len):
		check_a = 0x00
		check_b = 0x00

		for i in range(0, len):
			check_a += data[i]
			check_b += check_a
			
		return ((check_b % 256) << 8) + (check_a % 256)

##--------------------------------------------------------------------------------------##
##   计算待计算CRC的数据长度的函数                                  					##
##   输入：payload_len -- 当前数据帧的有效载荷长度                                      ##
##   返回值：待计算CRC的总数据长度                                                      ##
##--------------------------------------------------------------------------------------##		
	@staticmethod	
	def CRC_CALC_LEN(payload_len):	#3 = tid(2B) + len(1B) 	
		return (payload_len + PROTOCOL_TID_LEN + PROTOCOL_PAYLOAD_LEN)	

##--------------------------------------------------------------------------------------##
##   计算输出报文数据中CRC的位置的函数                              					##
##   输入：payload_len -- 当前数据帧的有效载荷长度                                      ##
##   返回值：当前输出的数据报文中CRC的起始位置                                          ##
##--------------------------------------------------------------------------------------##			
	@staticmethod	
	def PROTOCOL_CRC_DATA_POS(payload_len):
		return (CRC_CALC_START_POS + std_decoder.CRC_CALC_LEN(payload_len))

	''' 都是从位置0开始清除 '''	
	@staticmethod	
	def clear_data(data, len):
		for i in range(0, len):
			data.pop(0)
		
	def proc_data(self, data, data_len, result, dbg_flg):
	
		'''待解析总长度不足最小一帧数据'''
		if data_len < PROTOCOL_MIN_LEN:
			if dbg_flg:
				print('len not enough')		
			return False			
		
		'''找到帧头'''
		ret = data.find(YIS_HEADER)
		if dbg_flg:
			print(f'header pos {ret}')		
		if -1 != ret:
			if ret > 0:
				std_decoder.clear_data(data, ret)	
				
			''' 从这里开始data的0位置开始就帧头了 '''				
			cnt = len(data)
			if cnt < PROTOCOL_MIN_LEN:
				if dbg_flg:
					print('valid data less')	
				return False
	
			''' 查找到帧头，且剩余帧长大于最小一帧数据长度，开始解析 '''
			msg_len = struct.unpack_from('<B', data, PROTOCOL_PAYLOAD_LEN_POS)[0]
			if dbg_flg:		
				print(f'protocol len =  {msg_len}')	
					
			''' 校验有效总字节长度和协议字节长度对比 '''
			if (PROTOCOL_MIN_LEN + msg_len) > cnt:
				if dbg_flg:		
					print('protocol len is not enough')		
				return False			
			
			''' 计算校验和 '''
			check_sum = self.calc_crc16(data[CRC_CALC_START_POS : CRC_CALC_START_POS + self.CRC_CALC_LEN(msg_len)], self.CRC_CALC_LEN(msg_len))
			crc_received = struct.unpack_from('<H', data, self.PROTOCOL_CRC_DATA_POS(msg_len))[0]
			if check_sum != crc_received:
				if dbg_flg:
					print('valid data less')				
				std_decoder.clear_data(data, 2)
				return False

			if(dbg_flg):
				print('checksum done')		
			result['tid'] = struct.unpack_from('<H', data, PROTOCOL_TID_POS)[0]
			
			''' 解析数据 ''' 
			''' payload数据起始位置 ''' 			
			pos = PAYLOAD_POS
			payload_last = PAYLOAD_POS + msg_len
			
			''' 每个数据包由data_id + data_len + data组成 '''			
			tlv = {'id':0x10, 'len':0x0C}
			while pos < payload_last:
				tlv['id'] = struct.unpack_from('<B', data, pos)[0]
				tlv['len'] = struct.unpack_from('<B', data, pos + 1)[0]
				if dbg_flg:
					print(tlv)				
				ret = self.parse_data_by_id(tlv, data[pos + TLV_HEADER_LEN: cnt], result, dbg_flg)
				if(True == ret):
					pos += tlv['len'] + TLV_HEADER_LEN
				else:
					pos += 1		

				if dbg_flg:
					print('ret = ', ret, ' pos = ', pos)
					
			''' 解析完后清除当前已经解析的帧数据 '''
			self.clear_data(data, PROTOCOL_MIN_LEN + msg_len)
			if dbg_flg:
				if len(data) > 0:
					self.hex_show(data, len(data))		
			return True
		else:
			if dbg_flg:		
				print('clear all data')
			data.clear()
			return False
			
##--------------------------------------------------------------------------------------##
##   数据解析函数                                       								##
##   输入：tlv -- 当前待解析数据头信息，payload -- 当前待解析数据的有效载荷             ##
##   输入：存放解析完成后的真实数据字典，debug_flg -- 调试标志，True打开                ##
##   返回值：True -- 解析成功，False -- 未识别的数据                                    ##
##--------------------------------------------------------------------------------------##	
	''' struct unpack: c -- char; b -- signed char; B -- unsigned char; ? -- _Bool;
		i -- int; I -- unsigned int; h -- short; H -- unsigned short
		l -- long; L -- unsigned long; q -- long long; Q -- unsigned long long
		f -- float; d - double; s -- char[]; p -- char[]
		< -- little-endian; > -- big-endian; ! -- network, same to big-endian
	'''
	@staticmethod	
	def parse_data_by_id(tlv, payload, info, debug_flg):
		ret = True
		if (sensor_temp_id == tlv['id'] and sensor_temp_len == tlv['len']):
			if debug_flg:
				print('data temp')
			info['sensor_temp'] = struct.unpack_from('<h', payload)[0] * data_factor_sensor_temp
		elif (acc_id == tlv['id'] and acc_len == tlv['len']):
			if debug_flg:
				print('data acc')
			info['acc_x'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_not_raw_mag		
			info['acc_y'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_not_raw_mag	
			info['acc_z'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_not_raw_mag			
		elif (gyro_id == tlv['id'] and gyro_len == tlv['len']):
			if debug_flg:
				print('data gyro')
			info['gyro_x'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_not_raw_mag		
			info['gyro_y'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_not_raw_mag	
			info['gyro_z'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_not_raw_mag		
		elif (euler_id == tlv['id'] and euler_len == tlv['len']):
			if debug_flg:
				print('data euler')
			info['pitch'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_not_raw_mag		
			info['roll'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_not_raw_mag	
			info['yaw'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_not_raw_mag		
		elif (quaternion_id == tlv['id'] and quaternion_len == tlv['len']):
			if debug_flg:	
				print('data quaternion')
			info['q0'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_not_raw_mag		
			info['q1'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_not_raw_mag	
			info['q2'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_not_raw_mag		
			info['q3'] = struct.unpack_from('<i', payload, 12)[0] * data_factor_not_raw_mag				
		elif (norm_mag_id == tlv['id'] and norm_mag_len == tlv['len']):
			if debug_flg:	
				print('data norm mag')
			info['norm_mag_x'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_not_raw_mag		
			info['norm_mag_y'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_not_raw_mag	
			info['norm_mag_z'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_not_raw_mag		
		elif (raw_mag_id == tlv['id'] and raw_mag_len == tlv['len']):
			if debug_flg:	
				print('data raw mag')
			info['raw_mag_x'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_raw_mag		
			info['raw_mag_y'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_raw_mag	
			info['raw_mag_z'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_raw_mag		
		elif (location_id == tlv['id'] and location_len == tlv['len']):
			if debug_flg:	
				print('data location')
			info['alt'] = struct.unpack_from('<q', payload, 0)[0] * data_factor_high_res_loc		
			info['longt'] = struct.unpack_from('<q', payload, 8)[0] * data_factor_high_res_loc	
			info['alt'] = struct.unpack_from('<i', payload, 16)[0] * data_factor_alt		
		elif (utc_id == tlv['id'] and utc_len == tlv['len']):
			if debug_flg:	
				print('data utc')
			info['ms'] = struct.unpack_from('<I', payload, 0)[0]
			info['year'] = struct.unpack_from('<H', payload, 4)[0]	
			info['month'] = struct.unpack_from('<B', payload, 6)[0]		
			info['day'] = struct.unpack_from('<B', payload, 7)[0]	
			info['hour'] = struct.unpack_from('<B', payload, 8)[0]
			info['minute'] = struct.unpack_from('<B', payload, 9)[0]
			info['second'] = struct.unpack_from('<B', payload, 10)[0]		
		elif (smp_timestamp_id == tlv['id'] and smp_timestamp_len == tlv['len']):
			if debug_flg:	
				print('data sample timestamp')
			info['smp_timestamp'] = struct.unpack_from('<I', payload)[0]		
		elif (ready_timestamp_id == tlv['id'] and ready_timestamp_len == tlv['len']):
			if debug_flg:	
				print('data ready timestamp')
			info['ready_timestamp'] = struct.unpack_from('<I', payload)[0]		
		elif (speed_id == tlv['id'] and speed_len == tlv['len']):
			if debug_flg:	
				print('data speed')
			info['vel_e'] = struct.unpack_from('<i', payload, 0)[0] * data_factor_speed		
			info['vel_n'] = struct.unpack_from('<i', payload, 4)[0] * data_factor_speed	
			info['vel_u'] = struct.unpack_from('<i', payload, 8)[0] * data_factor_speed		
		elif (status_id == tlv['id'] and status_len == tlv['len']):
			if debug_flg:	
				print('data fusion status')
			info['status'] = payload[0]			
		else:
			print('unknown data id && len')
			ret = False
			
		return ret	
		
	@staticmethod	
	def hex_show(data, num):
		result = ''
		data_len = len(data)
		if num > data_len:
			print('para num error')
			return
		for i in range(0, num):
			hex_val = '%02x' % data[i]
			result += hex_val + ' '
		
		print('hex_show: ', result)		