"""
IMU Receiver for Tight Coupling (Accel + Gyro buffering) & Real-time Orientation
"""
import socket
import json
import threading
import time
import numpy as np
from collections import deque

class IMUReceiver:
    def __init__(self, port=9999):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False
        
        # [VIO 核心] 数据缓冲区: 存储 (timestamp, accel, gyro)
        self.imu_queue = deque(maxlen=2000) 
        
        # [UI/VO 核心] 实时姿态缓存
        self.latest_quat = np.array([1., 0., 0., 0.]) # [w, x, y, z]
        
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"✅ [IMU] Listening on {self.port} (Hybrid Mode: Buffering + Realtime)")

    def _loop(self):
        self.sock.bind(("0.0.0.0", self.port))
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode('utf-8'))
                
                ts = msg.get('ts', time.time())
                
                with self.lock:
                    # 1. VIO 需要的原始数据 (Acc + Gyro)
                    if 'acc' in msg and 'gyro' in msg:
                        acc = np.array(msg['acc'], dtype=np.float64)
                        gyro = np.array(msg['gyro'], dtype=np.float64)
                        self.imu_queue.append((ts, acc, gyro))
                    
                    # 2. UI/VO 需要的四元数 (用于可视化和初始猜测)
                    if 'quat' in msg:
                        self.latest_quat = np.array(msg['quat'], dtype=np.float64)
                    elif 'euler' in msg:
                        # 如果只有欧拉角，转四元数备用
                        self.latest_quat = self._euler_to_quat(msg['euler'])
                        
            except Exception as e:
                pass

    def get_quaternion(self):
        """兼容旧接口：获取最新姿态"""
        with self.lock:
            return self.latest_quat.copy()

    def get_imu_data_since(self, last_timestamp: float):
        """VIO接口：获取从 last_timestamp 之后的所有数据"""
        data_chunk = []
        with self.lock:
            for item in self.imu_queue:
                ts, acc, gyro = item
                if ts > last_timestamp:
                    data_chunk.append(item)
        return data_chunk

    def _euler_to_quat(self, euler):
        # 简单的欧拉角转四元数 (ZYX顺序)
        r, p, y = np.radians(euler)
        cy, sy = np.cos(y*0.5), np.sin(y*0.5)
        cp, sp = np.cos(p*0.5), np.sin(p*0.5)
        cr, sr = np.cos(r*0.5), np.sin(r*0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])