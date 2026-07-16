#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Dashboard for PiSLAM
Author: Jinxin Shao (Academic Visualization)
功能: 实时接收 UDP IMU 数据并绘制波形图 (Acc, Gyro, Euler)
"""

import socket
import json
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import sys

# ================= 配置 =================
UDP_PORT = 9999
HISTORY_LEN = 200  # 显示最近多少帧
# =======================================

class IMUDashboard:
    def __init__(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 关键：允许端口复用，防止与 SLAM 系统冲突
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass # 部分系统不支持 SO_REUSEPORT，可能会有竞争

        try:
            self.sock.bind(("0.0.0.0", UDP_PORT))
            print(f"[Dashboard] Listening on port {UDP_PORT}...")
        except Exception as e:
            print(f"[Error] Failed to bind port: {e}")
            sys.exit(1)

        # 数据容器
        self.ts_data = deque(maxlen=HISTORY_LEN)
        self.acc_data = {'x': deque(maxlen=HISTORY_LEN), 'y': deque(maxlen=HISTORY_LEN), 'z': deque(maxlen=HISTORY_LEN)}
        self.gyro_data = {'x': deque(maxlen=HISTORY_LEN), 'y': deque(maxlen=HISTORY_LEN), 'z': deque(maxlen=HISTORY_LEN)}
        self.euler_data = {'r': deque(maxlen=HISTORY_LEN), 'p': deque(maxlen=HISTORY_LEN), 'y': deque(maxlen=HISTORY_LEN)}
        
        # 启动接收线程
        self.thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.thread.start()

    def receive_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode('utf-8'))
                
                # 解析数据
                if 'acc' in msg and 'gyro' in msg and 'euler' in msg:
                    self.ts_data.append(time.time())
                    
                    self.acc_data['x'].append(msg['acc'][0])
                    self.acc_data['y'].append(msg['acc'][1])
                    self.acc_data['z'].append(msg['acc'][2])
                    
                    self.gyro_data['x'].append(msg['gyro'][0])
                    self.gyro_data['y'].append(msg['gyro'][1])
                    self.gyro_data['z'].append(msg['gyro'][2])
                    
                    self.euler_data['r'].append(msg['euler'][0])
                    self.euler_data['p'].append(msg['euler'][1])
                    self.euler_data['y'].append(msg['euler'][2])
            except:
                pass

    def run(self):
        # 设置绘图风格 (深色学术风)
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        fig.canvas.manager.set_window_title('PiSLAM - IMU Realtime Dashboard')
        plt.subplots_adjust(hspace=0.4)

        # 初始化线条
        lines = {}
        # Acc
        lines['ax'], = ax1.plot([], [], 'r-', label='X', lw=1)
        lines['ay'], = ax1.plot([], [], 'g-', label='Y', lw=1)
        lines['az'], = ax1.plot([], [], 'b-', label='Z', lw=1)
        ax1.set_title('Acceleration (m/s²)', fontsize=10, color='cyan')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_ylim(-15, 15)

        # Gyro
        lines['gx'], = ax2.plot([], [], 'r-', label='X', lw=1)
        lines['gy'], = ax2.plot([], [], 'g-', label='Y', lw=1)
        lines['gz'], = ax2.plot([], [], 'b-', label='Z', lw=1)
        ax2.set_title('Gyroscope (rad/s)', fontsize=10, color='orange')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(-5, 5)

        # Euler
        lines['er'], = ax3.plot([], [], 'm-', label='Roll', lw=1)
        lines['ep'], = ax3.plot([], [], 'y-', label='Pitch', lw=1)
        lines['ey'], = ax3.plot([], [], 'c-', label='Yaw', lw=1)
        ax3.set_title('Euler Angles (deg)', fontsize=10, color='lime')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_ylim(-180, 180)

        def update(frame):
            if not self.ts_data: return lines.values()
            
            x_axis = range(len(self.ts_data))
            
            # Update Acc
            lines['ax'].set_data(x_axis, self.acc_data['x'])
            lines['ay'].set_data(x_axis, self.acc_data['y'])
            lines['az'].set_data(x_axis, self.acc_data['z'])
            ax1.set_xlim(0, len(self.ts_data))
            
            # Update Gyro
            lines['gx'].set_data(x_axis, self.gyro_data['x'])
            lines['gy'].set_data(x_axis, self.gyro_data['y'])
            lines['gz'].set_data(x_axis, self.gyro_data['z'])
            ax2.set_xlim(0, len(self.ts_data))

            # Update Euler
            lines['er'].set_data(x_axis, self.euler_data['r'])
            lines['ep'].set_data(x_axis, self.euler_data['p'])
            lines['ey'].set_data(x_axis, self.euler_data['y'])
            ax3.set_xlim(0, len(self.ts_data))

            return lines.values()

        ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
        plt.show()
        self.running = False

if __name__ == "__main__":
    dashboard = IMUDashboard()
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("Dashboard closed.")