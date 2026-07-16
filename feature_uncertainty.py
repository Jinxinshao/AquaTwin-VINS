#!/usr/bin/env python3
"""
================================================================================
Feature Uncertainty Modeling for High-Precision SLAM
================================================================================

学术背景：
---------
高精度 SLAM 系统必须建模并传播测量不确定性。特征检测的不确定性来源于：

1. 检测器响应强度 → 弱响应的角点定位不准
2. 亚像素精化质量 → 梯度不足区域精化失败
3. 深度估计不确定性 → 单目深度与真实深度的偏差
4. 匹配置信度 → 描述子相似度反映匹配可靠性

不确定性传播路径：
----------------
2D 检测 → 3D 反投影 → PnP 位姿估计 → 地图点更新 → 全局一致性

在每个阶段，正确传播协方差可以：
1. 使优化器正确加权不同质量的观测
2. 检测异常值（马氏距离）
3. 量化最终估计的不确定性

数学模型：
---------
设特征点 2D 坐标为 u = [u, v]^T，其协方差为 Σ_u (2×2)。
深度估计为 d，其方差为 σ_d²。

反投影到 3D 的协方差传播：
    P_cam = d · K^{-1} · [u, v, 1]^T
    
由误差传播公式：
    Σ_P = J_u · Σ_u · J_u^T + J_d · σ_d² · J_d^T

其中 J_u = ∂P/∂u, J_d = ∂P/∂d

参考文献：
---------
[1] Engel et al., "LSD-SLAM: Large-Scale Direct Monocular SLAM", ECCV 2014
[2] Civera et al., "Inverse Depth Parametrization for Monocular SLAM", TRO 2008
[3] Montiel et al., "Unified Inverse Depth Parametrization for Monocular SLAM", RSS 2006

Author: Academic Enhancement
================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import cv2


@dataclass
class UncertainFeature:
    """
    带不确定性的特征点。
    
    Attributes:
        u, v: 像素坐标
        cov_2d: 2D 位置协方差 (2×2)
        depth: 深度估计
        sigma_depth: 深度标准差
        point_3d: 相机坐标系下的 3D 位置 (3,)
        cov_3d: 3D 位置协方差 (3×3)
        response: 检测器响应强度
        descriptor: 描述子
        track_id: 跟踪 ID
        confidence: 综合置信度
    """
    u: float
    v: float
    cov_2d: np.ndarray  # (2, 2)
    depth: float
    sigma_depth: float
    point_3d: Optional[np.ndarray] = None  # (3,)
    cov_3d: Optional[np.ndarray] = None    # (3, 3)
    response: float = 1.0
    descriptor: Optional[np.ndarray] = None
    track_id: int = -1
    confidence: float = 1.0


class FeatureUncertaintyModel:
    """
    特征不确定性建模器。
    
    估计每个特征点的 2D 和 3D 不确定性。
    """
    
    def __init__(
        self,
        base_sigma_pixel: float = 0.5,      # 基础像素噪声
        depth_sigma_coeff: float = 0.02,    # 深度不确定性系数
        min_response_threshold: float = 10, # 最小响应强度
        subpixel_boost: float = 0.5         # 亚像素精化后的噪声降低系数
    ):
        """
        Args:
            base_sigma_pixel: 基础像素标准差（未亚像素精化）
            depth_sigma_coeff: 深度不确定性 = coeff * depth²
            min_response_threshold: 响应低于此值视为低质量
            subpixel_boost: 亚像素精化后协方差缩放系数
        """
        self.base_sigma_pixel = base_sigma_pixel
        self.depth_sigma_coeff = depth_sigma_coeff
        self.min_response = min_response_threshold
        self.subpixel_boost = subpixel_boost
        
    def estimate_2d_covariance(
        self,
        keypoint: cv2.KeyPoint,
        has_subpixel: bool = True,
        gradient_magnitude: Optional[float] = None
    ) -> np.ndarray:
        """
        估计 2D 特征点的协方差。
        
        协方差来源：
        1. 检测器固有噪声
        2. 响应强度（弱响应 → 大不确定性）
        3. 局部梯度（低梯度 → 大不确定性）
        4. 亚像素精化（有 → 低不确定性）
        
        Returns:
            cov_2d: 2×2 协方差矩阵
        """
        # 基础方差
        base_var = self.base_sigma_pixel ** 2
        
        # 响应因子：低响应增加方差
        response = max(keypoint.response, 1.0)
        response_factor = max(1.0, self.min_response / response)
        
        # 梯度因子（如果有）
        if gradient_magnitude is not None:
            # 梯度越小 → 定位越不准
            grad_factor = max(1.0, 20.0 / (gradient_magnitude + 1.0))
        else:
            grad_factor = 1.0
            
        # 亚像素因子
        subpixel_factor = self.subpixel_boost if has_subpixel else 1.0
        
        # 最终方差
        variance = base_var * response_factor * grad_factor * subpixel_factor
        
        # 各向同性协方差（可以扩展为各向异性）
        cov_2d = np.eye(2) * variance
        
        return cov_2d
        
    def estimate_depth_uncertainty(
        self,
        depth: float,
        depth_source: str = "network",
        is_edge: bool = False
    ) -> float:
        """
        估计深度的不确定性。
        
        深度不确定性模型：
        - 网络深度：σ = α × d²（视差噪声传播）
        - 边缘处：额外增加不确定性（深度不连续）
        
        Args:
            depth: 深度值 (米)
            depth_source: 深度来源 ("network", "stereo", "rgbd")
            is_edge: 是否在深度边缘附近
            
        Returns:
            sigma_depth: 深度标准差
        """
        # 基础模型：与深度平方成正比
        sigma = self.depth_sigma_coeff * (depth ** 2)
        
        # 深度来源因子
        source_factors = {
            "network": 1.0,   # 神经网络深度
            "stereo": 0.5,    # 双目立体
            "rgbd": 0.2       # RGBD 传感器
        }
        sigma *= source_factors.get(depth_source, 1.0)
        
        # 边缘惩罚
        if is_edge:
            sigma *= 2.0
            
        # 深度范围惩罚
        if depth < 0.5:
            sigma *= 1.5  # 近距离深度不可靠
        elif depth > 5.0:
            sigma *= 1.2  # 远距离深度不可靠
            
        return sigma
        
    def backproject_with_covariance(
        self,
        u: float, v: float,
        cov_2d: np.ndarray,
        depth: float,
        sigma_depth: float,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        带协方差传播的反投影。
        
        将 2D 点 + 深度反投影到 3D，同时传播不确定性。
        
        数学推导：
        ---------
        P_cam = [X, Y, Z]^T = d × K^{-1} × [u, v, 1]^T
        
        即：
            X = d × (u - cx) / fx
            Y = d × (v - cy) / fy
            Z = d
            
        雅可比矩阵：
        J_uv = ∂P/∂[u,v] = d × [[1/fx, 0], [0, 1/fy], [0, 0]]
        J_d = ∂P/∂d = [(u-cx)/fx, (v-cy)/fy, 1]^T
        
        协方差传播：
        Σ_P = J_uv × Σ_uv × J_uv^T + J_d × σ_d² × J_d^T
        
        Args:
            u, v: 像素坐标
            cov_2d: 2D 协方差 (2×2)
            depth: 深度
            sigma_depth: 深度标准差
            K: 相机内参 (3×3)
            
        Returns:
            point_3d: 相机坐标系下的 3D 点 (3,)
            cov_3d: 3D 协方差 (3×3)
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 3D 点
        X = depth * (u - cx) / fx
        Y = depth * (v - cy) / fy
        Z = depth
        point_3d = np.array([X, Y, Z])
        
        # 雅可比 J_uv (3×2)
        J_uv = depth * np.array([
            [1 / fx, 0],
            [0, 1 / fy],
            [0, 0]
        ])
        
        # 雅可比 J_d (3×1)
        J_d = np.array([
            [(u - cx) / fx],
            [(v - cy) / fy],
            [1]
        ])
        
        # 协方差传播
        cov_from_uv = J_uv @ cov_2d @ J_uv.T
        cov_from_depth = (sigma_depth ** 2) * (J_d @ J_d.T)
        
        cov_3d = cov_from_uv + cov_from_depth
        
        return point_3d, cov_3d


class InverseDepthParametrization:
    """
    逆深度参数化。
    
    学术背景：
    ---------
    单目 SLAM 中，传统的 XYZ 参数化在初始化时深度不确定性很大，
    可能导致数值问题。逆深度参数化 ρ = 1/d 具有以下优势：
    
    1. 深度不确定性在逆深度空间近似高斯
    2. 可以自然处理无穷远点（ρ → 0）
    3. 线性化误差更小
    
    参数化：
    λ = [θ, φ, ρ]^T
    
    其中 (θ, φ) 是视线方向（球坐标），ρ = 1/d 是逆深度。
    
    参考：Civera et al., "Inverse Depth Parametrization for Monocular SLAM"
    """
    
    def __init__(self, min_depth: float = 0.1, max_depth: float = 100.0):
        self.rho_min = 1.0 / max_depth
        self.rho_max = 1.0 / min_depth
        
    def xyz_to_inverse(
        self,
        point_3d: np.ndarray,
        cov_xyz: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        将 XYZ 坐标转换为逆深度参数化。
        
        Returns:
            params: [θ, φ, ρ]
            cov_params: 转换后的协方差 (如果提供了输入协方差)
        """
        X, Y, Z = point_3d
        
        # 逆深度
        d = np.linalg.norm(point_3d)
        rho = 1.0 / d
        
        # 视线方向（球坐标）
        theta = np.arctan2(X, Z)  # azimuth
        phi = np.arcsin(Y / d)    # elevation
        
        params = np.array([theta, phi, rho])
        
        # 协方差传播
        cov_params = None
        if cov_xyz is not None:
            # 计算雅可比
            J = self._jacobian_xyz_to_inverse(point_3d)
            cov_params = J @ cov_xyz @ J.T
            
        return params, cov_params
        
    def inverse_to_xyz(
        self,
        params: np.ndarray,
        cov_params: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        将逆深度参数转换回 XYZ。
        """
        theta, phi, rho = params
        
        d = 1.0 / rho
        X = d * np.sin(theta) * np.cos(phi)
        Y = d * np.sin(phi)
        Z = d * np.cos(theta) * np.cos(phi)
        
        point_3d = np.array([X, Y, Z])
        
        # 协方差传播
        cov_xyz = None
        if cov_params is not None:
            J = self._jacobian_inverse_to_xyz(params)
            cov_xyz = J @ cov_params @ J.T
            
        return point_3d, cov_xyz
        
    def _jacobian_xyz_to_inverse(self, point_3d: np.ndarray) -> np.ndarray:
        """计算 XYZ → 逆深度的雅可比"""
        X, Y, Z = point_3d
        d = np.linalg.norm(point_3d)
        d2 = d ** 2
        d3 = d ** 3
        
        # dθ/dX = Z / (X² + Z²)
        # dθ/dZ = -X / (X² + Z²)
        xz2 = X**2 + Z**2
        
        J = np.array([
            [Z / xz2, 0, -X / xz2],                                    # dθ/d[X,Y,Z]
            [-X*Y / (d2 * np.sqrt(xz2)), np.sqrt(xz2) / d2, -Y*Z / (d2 * np.sqrt(xz2))],  # dφ/d[X,Y,Z]
            [-X / d3, -Y / d3, -Z / d3]                                 # dρ/d[X,Y,Z]
        ])
        
        return J
        
    def _jacobian_inverse_to_xyz(self, params: np.ndarray) -> np.ndarray:
        """计算逆深度 → XYZ 的雅可比"""
        theta, phi, rho = params
        d = 1.0 / rho
        
        # X = d sin(θ) cos(φ)
        # Y = d sin(φ)
        # Z = d cos(θ) cos(φ)
        
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        
        J = np.array([
            [d * ct * cp, -d * st * sp, -d**2 * st * cp],  # dX/d[θ,φ,ρ]
            [0, d * cp, -d**2 * sp],                        # dY/d[θ,φ,ρ]
            [-d * st * cp, -d * ct * sp, -d**2 * ct * cp]   # dZ/d[θ,φ,ρ]
        ])
        
        return J


# =============================================================================
# 综合特征处理器
# =============================================================================

class UncertaintyAwareFeatureProcessor:
    """
    不确定性感知的特征处理器。
    
    整合特征检测、不确定性估计、反投影的完整流水线。
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        depth_sigma_coeff: float = 0.02,
        use_inverse_depth: bool = False
    ):
        """
        Args:
            camera_matrix: 相机内参 K (3×3)
            depth_sigma_coeff: 深度不确定性系数
            use_inverse_depth: 是否使用逆深度参数化
        """
        self.K = camera_matrix
        self.uncertainty_model = FeatureUncertaintyModel(
            depth_sigma_coeff=depth_sigma_coeff
        )
        self.use_inverse_depth = use_inverse_depth
        if use_inverse_depth:
            self.inv_depth = InverseDepthParametrization()
            
    def process_keypoints(
        self,
        keypoints: list,
        depth_map: np.ndarray,
        descriptors: Optional[np.ndarray] = None,
        gradient_map: Optional[np.ndarray] = None,
        has_subpixel: bool = True
    ) -> list:
        """
        处理关键点，添加不确定性信息。
        
        Args:
            keypoints: OpenCV KeyPoint 列表
            depth_map: 深度图
            descriptors: 描述子矩阵
            gradient_map: 梯度幅值图（可选）
            has_subpixel: 是否已进行亚像素精化
            
        Returns:
            uncertain_features: UncertainFeature 列表
        """
        features = []
        
        for i, kp in enumerate(keypoints):
            u, v = kp.pt
            ui, vi = int(round(u)), int(round(v))
            
            # 边界检查
            if not (0 <= ui < depth_map.shape[1] and 0 <= vi < depth_map.shape[0]):
                continue
                
            # 获取深度
            depth = depth_map[vi, ui]
            if depth <= 0 or depth > 100:
                continue
                
            # 检测边缘
            is_edge = self._check_depth_edge(depth_map, ui, vi)
            
            # 估计 2D 协方差
            grad_mag = None
            if gradient_map is not None:
                grad_mag = gradient_map[vi, ui]
            cov_2d = self.uncertainty_model.estimate_2d_covariance(
                kp, has_subpixel, grad_mag
            )
            
            # 估计深度不确定性
            sigma_depth = self.uncertainty_model.estimate_depth_uncertainty(
                depth, "network", is_edge
            )
            
            # 反投影到 3D
            point_3d, cov_3d = self.uncertainty_model.backproject_with_covariance(
                u, v, cov_2d, depth, sigma_depth, self.K
            )
            
            # 创建特征
            feat = UncertainFeature(
                u=u, v=v,
                cov_2d=cov_2d,
                depth=depth,
                sigma_depth=sigma_depth,
                point_3d=point_3d,
                cov_3d=cov_3d,
                response=kp.response,
                descriptor=descriptors[i] if descriptors is not None else None,
                confidence=self._compute_confidence(kp, depth, sigma_depth)
            )
            
            features.append(feat)
            
        return features
        
    def _check_depth_edge(
        self,
        depth_map: np.ndarray,
        u: int, v: int,
        window: int = 3
    ) -> bool:
        """检查是否在深度边缘附近"""
        h, w = depth_map.shape
        
        # 边界安全
        v0 = max(0, v - window)
        v1 = min(h, v + window + 1)
        u0 = max(0, u - window)
        u1 = min(w, u + window + 1)
        
        patch = depth_map[v0:v1, u0:u1]
        valid = patch[patch > 0]
        
        if len(valid) < 4:
            return True
            
        # 深度变化超过 10% 认为是边缘
        depth_range = np.ptp(valid)
        depth_mean = np.mean(valid)
        
        return (depth_range / depth_mean) > 0.1
        
    def _compute_confidence(
        self,
        kp: cv2.KeyPoint,
        depth: float,
        sigma_depth: float
    ) -> float:
        """计算特征的综合置信度"""
        # 响应置信度
        response_conf = min(1.0, kp.response / 50.0)
        
        # 深度置信度（相对不确定性）
        depth_conf = 1.0 / (1.0 + sigma_depth / depth)
        
        # 深度范围置信度
        range_conf = 1.0
        if depth < 0.5 or depth > 5.0:
            range_conf = 0.7
            
        return response_conf * depth_conf * range_conf


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Uncertainty Modeling Test")
    print("=" * 60)
    
    # 相机内参
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 创建处理器
    processor = UncertaintyAwareFeatureProcessor(K)
    
    # 模拟关键点
    class MockKeyPoint:
        def __init__(self, x, y, response):
            self.pt = (x, y)
            self.response = response
            
    keypoints = [
        MockKeyPoint(320, 240, 100),  # 图像中心，高响应
        MockKeyPoint(100, 100, 20),   # 角落，低响应
        MockKeyPoint(500, 400, 50),   # 边缘区域
    ]
    
    # 模拟深度图
    depth_map = np.ones((480, 640)) * 2.0  # 2米深度
    depth_map[100, 100] = 5.0  # 远处
    depth_map[400, 500] = 0.5  # 近处
    
    # 处理
    features = processor.process_keypoints(keypoints, depth_map)
    
    print("\nProcessed features:")
    for i, f in enumerate(features):
        print(f"\nFeature {i}:")
        print(f"  Position: ({f.u:.1f}, {f.v:.1f})")
        print(f"  Depth: {f.depth:.2f} ± {f.sigma_depth:.3f} m")
        print(f"  3D: {f.point_3d}")
        print(f"  Cov_3D diag: {np.sqrt(np.diag(f.cov_3d))}")
        print(f"  Confidence: {f.confidence:.2f}")
        
    # 测试逆深度参数化
    print("\n" + "=" * 60)
    print("Inverse Depth Parametrization Test")
    print("=" * 60)
    
    inv_depth = InverseDepthParametrization()
    
    point = np.array([1.0, 0.5, 3.0])  # 相机坐标
    cov = np.diag([0.01, 0.01, 0.1])   # XYZ 协方差
    
    params, cov_params = inv_depth.xyz_to_inverse(point, cov)
    print(f"\nOriginal XYZ: {point}")
    print(f"Inverse params [θ, φ, ρ]: {params}")
    print(f"Cov diagonal: {np.sqrt(np.diag(cov_params))}")
    
    # 转回来
    point_back, cov_back = inv_depth.inverse_to_xyz(params, cov_params)
    print(f"\nRecovered XYZ: {point_back}")
    print(f"Error: {np.linalg.norm(point - point_back):.6f}")
