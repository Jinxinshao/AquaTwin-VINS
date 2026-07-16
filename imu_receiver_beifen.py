"""
IMU Receiver with Quaternion Support
"""
import socket
import json
import threading
import time
import numpy as np

class IMUReceiver:
    def __init__(self, port=9999):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False
        self.quat = np.array([1., 0., 0., 0.]) # [w, x, y, z] Hamilton
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"✅ [IMU] Listening on {self.port} (Quaternion Mode)")

    def _loop(self):
        self.sock.bind(("0.0.0.0", self.port))
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode('utf-8'))
                
                with self.lock:
                    # 1. 优先读取四元数
                    if 'quat' in msg:
                        # 假设驱动发来的是 [w, x, y, z]
                        self.quat = np.array(msg['quat'], dtype=np.float64)
                    
                    # 2. 只有欧拉角时的备选方案
                    elif 'euler' in msg:
                        self.quat = self._euler_to_quat(msg['euler'])
            except:
                pass

    def get_quaternion(self):
        with self.lock:
            return self.quat.copy()

    def _euler_to_quat(self, euler):
        # R-P-Y to Quaternion
        r, p, y = np.radians(euler)
        cy, sy = np.cos(y*0.5), np.sin(y*0.5)
        cp, sp = np.cos(p*0.5), np.sin(p*0.5)
        cr, sr = np.cos(r*0.5), np.sin(r*0.5)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return np.array([w, x, y, z])