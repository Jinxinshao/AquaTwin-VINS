#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHEELTEC H30 全功能极速驱动 (Universal High-Speed Driver)
功能: 暴力扫描 Acc, Gyro, Mag, Euler, Quaternion
特性: 
1. 0延迟处理，UDP全速广播
2. 自动容错，支持热拔插
3. 输出 SLAM 所需的所有核心数据
"""

import serial
import struct
import time
import socket
import json
import math
import sys

# ================= 🚀 配置中心 =================
SERIAL_PORT = '/dev/imu_usb'
BAUD_RATE   = 460800           # 波特率
UDP_IP      = "127.0.0.1"      # 广播IP
UDP_PORT    = 9999             # 广播端口

# 协议系数 (源自 Yesense 标准)
SCALE_COMMON = 0.000001        # 通用系数 1e-6
DEG2RAD      = math.pi / 180.0 # 角度转弧度
# ===============================================

class FullIMUDriver:
    def __init__(self):
        self.ser = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = True
        
        # 统计
        self.pkt_count = 0
        self.start_time = time.time()

    def connect(self):
        while self.running:
            try:
                print(f"⚡ [System] 正在全速连接 {SERIAL_PORT} @ {BAUD_RATE}...")
                self.ser = serial.Serial(
                    port=SERIAL_PORT,
                    baudrate=BAUD_RATE,
                    timeout=0.01, # 极短超时，非阻塞模式
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                print("✅ [System] 连接成功！全数据解析引擎已启动...")
                return
            except Exception as e:
                print(f"❌ [Error] 连接失败: {e}")
                time.sleep(1)

    def parse_buffer(self, buffer):
        """
        全量特征扫描器
        在缓冲区中寻找所有已知的数据头
        """
        parsed_data = {}
        max_idx = 0

        # --- 1. 加速度 (ID: 0x10, Len: 12) ---
        idx = buffer.find(b'\x10\x0c')
        if idx != -1 and idx + 14 <= len(buffer):
            try:
                raw = struct.unpack('<iii', buffer[idx+2 : idx+14])
                parsed_data['acc'] = [x * SCALE_COMMON for x in raw] # 单位: m/s²
                max_idx = max(max_idx, idx + 14)
            except: pass

        # --- 2. 陀螺仪 (ID: 0x20, Len: 12) ---
        idx = buffer.find(b'\x20\x0c')
        if idx != -1 and idx + 14 <= len(buffer):
            try:
                raw = struct.unpack('<iii', buffer[idx+2 : idx+14])
                # 输出: [rad/s, rad/s, rad/s]
                parsed_data['gyro'] = [x * SCALE_COMMON * DEG2RAD for x in raw] 
                max_idx = max(max_idx, idx + 14)
            except: pass

        # --- 3. 磁力计 (ID: 0x30, Len: 12) ---
        idx = buffer.find(b'\x30\x0c')
        if idx != -1 and idx + 14 <= len(buffer):
            try:
                raw = struct.unpack('<iii', buffer[idx+2 : idx+14])
                parsed_data['mag'] = [x * SCALE_COMMON for x in raw] # 单位: usually normalized
                max_idx = max(max_idx, idx + 14)
            except: pass

        # --- 4. 欧拉角 (ID: 0x40, Len: 12) ---
        idx = buffer.find(b'\x40\x0c')
        if idx != -1 and idx + 14 <= len(buffer):
            try:
                raw = struct.unpack('<iii', buffer[idx+2 : idx+14])
                # 输出: [Roll, Pitch, Yaw] (单位: 度)
                parsed_data['euler'] = [x * SCALE_COMMON for x in raw] 
                max_idx = max(max_idx, idx + 14)
            except: pass

        # --- 5. 四元数 (ID: 0x50, Len: 16) ---
        # 注意: 四元数长度是 16 字节 (4个float/int)
        idx = buffer.find(b'\x50\x10')
        if idx != -1 and idx + 18 <= len(buffer):
            try:
                raw = struct.unpack('<iiii', buffer[idx+2 : idx+18])
                parsed_data['quat'] = [x * SCALE_COMMON for x in raw] # [w, x, y, z]
                max_idx = max(max_idx, idx + 18)
            except: pass

        return parsed_data, max_idx

    def run(self):
        self.connect()
        
        buffer = b''
        last_print = 0
        
        while self.running:
            try:
                # 1. 极速读取
                if self.ser.in_waiting:
                    # 一次性读完缓冲区，最大4KB
                    chunk = self.ser.read(min(self.ser.in_waiting, 4096))
                    buffer += chunk
                
                # 缓冲区维护 (防止内存溢出)
                if len(buffer) > 4096:
                    buffer = buffer[-2048:]
                
                if len(buffer) < 20: # 数据太少，不够解析
                    continue

                # 2. 解析
                data, consumed_len = self.parse_buffer(buffer)
                
                # 3. 消费数据 (移除已解析部分)
                if consumed_len > 0:
                    buffer = buffer[consumed_len:]
                else:
                    # 没找到任何特征，滑动窗口丢弃旧数据
                    if len(buffer) > 1000: buffer = buffer[100:]

                # 4. 发送与打印
                if data:
                    data['ts'] = time.time()
                    self.udp_broadcast(data)
                    self.pkt_count += 1
                    
                    # 5. 终端监控 (每 100ms 打印一次，给人看)
                    if time.time() - last_print > 0.1:
                        self.print_status(data)
                        last_print = time.time()

            except Exception as e:
                # 容错处理
                # print(f"⚠️ {e}") # 生产环境可注释掉
                try: self.ser.close()
                except: pass
                self.connect()

    def udp_broadcast(self, data):
        try:
            msg = json.dumps(data).encode('utf-8')
            self.sock.sendto(msg, (UDP_IP, UDP_PORT))
        except: pass

    def print_status(self, d):
        # 格式化输出，看起来更专业
        info = []
        if 'acc' in d:   info.append(f"AccZ:{d['acc'][2]:6.2f}")
        if 'gyro' in d:  info.append(f"GyrZ:{d['gyro'][2]:6.3f}")
        if 'euler' in d: info.append(f"Yaw:{d['euler'][2]:6.1f}°")
        if 'mag' in d:   info.append(f"Mag:{d['mag'][0]:.1f}")
        
        # 计算频率
        freq = self.pkt_count / (time.time() - self.start_time)
        print(f"\r🚀 [{freq:3.0f}Hz] " + " | ".join(info) + " " * 10, end="")

if __name__ == '__main__':
    try:
        driver = FullIMUDriver()
        driver.run()
    except KeyboardInterrupt:
        print("\n👋 停止")