import numpy as np
import gtsam
from typing import Tuple, Optional

class IMUPreintegrator:
    """
    IMU 预积分器 (基于 GTSAM)
    负责将高频 IMU 数据压缩为两帧之间的相对约束因子。
    
    Academic Note:
    使用 gtsam.PreintegrationParams 配合 gtsam.ImuFactor。
    在此模式下，Params 仅包含传感器测量噪声 (Sensor Noise)。
    IMU 零偏的随机游走 (Bias Random Walk) 由后端因子图中的 BetweenFactor 处理。
    """
    def __init__(self, config: dict):
        self.config = config
        
        # 1. 配置预积分参数
        # 这里的参数直接决定了优化器对 IMU 数据的信任程度
        # 必须确保 gravity 是 float 类型
        gravity = float(config.get('gravity', 9.81))
        params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        
        # 提取噪声参数 (从 config 读取，单位通常是 sigma)
        # 协方差 = sigma^2
        acc_noise = float(config.get('accel_noise_density', 2.0e-3))
        gyr_noise = float(config.get('gyro_noise_density', 1.7e-4))
        
        # 构造协方差矩阵 (3x3 对角阵)
        k_accel = np.eye(3) * (acc_noise ** 2)
        k_gyro  = np.eye(3) * (gyr_noise ** 2)
        k_integration = np.eye(3) * 1e-8  # 积分数值误差，给一个极小值
        
        # ------------------------------------------------------------------
        # 🔧 [核心修复] 使用 Setter 方法设置协方差
        # ------------------------------------------------------------------
        # 设置加速度计测量噪声协方差 (Sigma_a)
        params.setAccelerometerCovariance(k_accel)
        
        # 设置陀螺仪测量噪声协方差 (Sigma_w)
        params.setGyroscopeCovariance(k_gyro)
        
        # 设置积分不确定性
        params.setIntegrationCovariance(k_integration)
        
        # [学术修正]
        # 对于标准的 ImuFactor，Bias 的随机游走（Random Walk）不在 params 里设置。
        # 它是通过 pose_graph_enhanced.py 中的 BetweenFactorConstantBias 建模的。
        # 因此，这里不需要也不应该调用 setBiasAccCovariance 等方法，避免 API 报错。
        
        self.params = params
        
        # 2. 初始化状态
        # current_bias: 当前估计的 IMU 零偏 (Bias)
        # 初始状态假设零偏为 0，后续由优化器更新
        self.current_bias = gtsam.imuBias.ConstantBias(
            np.zeros(3), np.zeros(3) 
        )
        
        # pim: PreintegratedImuMeasurements (累积器)
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.current_bias)
        
        self.last_imu_time = -1.0
        
    def reset(self):
        """重置预积分器 (通常在关键帧插入后调用)"""
        self.pim.resetIntegration()
        
    def add_imu_measurement(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        """
        加入单个 IMU 测量值
        accel: [ax, ay, az] m/s^2
        gyro: [wx, wy, wz] rad/s
        """
        if self.last_imu_time < 0:
            # 第一帧，使用配置的频率估算 dt，或者默认为 0.005s (200Hz)
            freq = float(self.config.get('frequency', 200.0))
            dt = 1.0 / freq if freq > 0 else 0.005
        else:
            dt = timestamp - self.last_imu_time
            
        if dt <= 0:
            # 防止时间戳乱序或重复导致的数值不稳定
            return 
        
        # GTSAM 积分
        self.pim.integrateMeasurement(accel, gyro, dt)
        self.last_imu_time = timestamp
        
    def predict_state(self, prev_state, dt: float):
        """
        仅用于前端预测 (Predict): 根据上一帧状态和当前预积分，
        预测当前的位姿 (Pose) 和速度 (Velocity)。
        用于给 PnP 提供极其准确的 Initial Guess。
        """
        # prev_state 需要包含: pose (NavState), velocity, bias
        # 注意：这里假设 prev_state['pose'] 已经是 gtsam.Pose3 类型
        # 如果不是，可能需要转换
        nav_state = gtsam.NavState(prev_state['pose'], prev_state['velocity'])
        predicted_nav_state = self.pim.predict(nav_state, self.current_bias)
        return predicted_nav_state

    def get_factor(self):
        """获取当前的预积分因子 (用于插入因子图)"""
        return self.pim

    def update_bias(self, new_bias):
        self.current_bias = new_bias
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.current_bias)
        self.last_imu_time = -1.0