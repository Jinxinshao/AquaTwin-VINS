#!/usr/bin/env python3
"""
================================================================================
Adaptive Information Matrix Estimation for Pose Graph SLAM
================================================================================

学术背景：
---------
位姿图优化的目标函数为：

    L = Σ_{ij} ||T_i^{-1} T_j ⊖ Δ_ij||²_Ω_ij

其中 Ω_ij 是信息矩阵（协方差的逆）。信息矩阵的准确估计直接影响：
1. 优化权重分配 → 高质量约束应该有更大权重
2. 不确定性传播 → 最终位姿估计的协方差
3. 异常约束检测 → 马氏距离需要准确的协方差

信息矩阵的来源：
--------------
1. 帧间里程计约束：来自 PnP 求解的 Hessian 矩阵
2. 回环约束：来自几何验证的内点数和重投影误差
3. IMU 约束：来自预积分的协方差传播

本模块实现基于观测质量的自适应信息矩阵估计。

参考文献：
---------
[1] Kaess et al., "iSAM2: Incremental Smoothing and Mapping Using the 
    Bayes Tree", IJRR 2012
[2] Strasdat et al., "Scale Drift-Aware Large Scale Monocular SLAM", RSS 2010
[3] Mur-Artal et al., "ORB-SLAM2: An Open-Source SLAM System for Monocular, 
    Stereo, and RGB-D Cameras", TRO 2017

Author: Academic Enhancement
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class PoseEstimationStatistics:
    """
    位姿估计的统计信息，用于计算信息矩阵。
    """
    n_features: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    mean_reproj_error: float = float('inf')
    median_reproj_error: float = float('inf')
    mean_depth: float = 1.0
    track_length_mean: float = 1.0
    method: str = "unknown"
    has_imu: bool = False


class AdaptiveInformationMatrix:
    """
    自适应信息矩阵估计器。
    
    核心思想：信息矩阵应该反映位姿估计的不确定性。我们从以下因素估计不确定性：
    
    1. 内点数量：更多内点 → 更低不确定性
    2. 重投影误差：更小误差 → 更低不确定性  
    3. 特征分布：均匀分布 → 更低旋转不确定性
    4. 深度分布：深度变化大 → 更低尺度不确定性
    5. 跟踪长度：长轨迹特征 → 更低不确定性
    """
    
    def __init__(
        self,
        base_sigma_trans: float = 0.01,   # 基础平移标准差 (米)
        base_sigma_rot: float = 0.001,    # 基础旋转标准差 (弧度)
        min_inliers_for_good: int = 50,   # "良好"估计的最小内点数
        max_reproj_for_good: float = 1.0  # "良好"估计的最大重投影误差
    ):
        """
        Args:
            base_sigma_trans: 基础平移标准差
            base_sigma_rot: 基础旋转标准差
            min_inliers_for_good: 定义良好估计的内点阈值
            max_reproj_for_good: 定义良好估计的重投影误差阈值
        """
        self.base_sigma_trans = base_sigma_trans
        self.base_sigma_rot = base_sigma_rot
        self.min_inliers_for_good = min_inliers_for_good
        self.max_reproj_for_good = max_reproj_for_good
        
    def estimate_from_pnp(
        self,
        stats: PoseEstimationStatistics,
        feature_points: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = (640, 480)
    ) -> np.ndarray:
        """
        从 PnP 统计信息估计信息矩阵。
        
        返回 6×6 信息矩阵，对应 [tx, ty, tz, rx, ry, rz]
        
        Args:
            stats: 位姿估计统计
            feature_points: 特征点位置 (N×2)
            depths: 特征深度 (N,)
            image_size: 图像尺寸 (宽, 高)
            
        Returns:
            info_matrix: 6×6 信息矩阵
        """
        # 基础协方差
        base_cov_trans = np.eye(3) * (self.base_sigma_trans ** 2)
        base_cov_rot = np.eye(3) * (self.base_sigma_rot ** 2)
        
        # 根据估计质量调整
        quality_factor = self._compute_quality_factor(stats)
        
        # 质量因子越高 → 不确定性越低 → 信息越高
        cov_trans = base_cov_trans / quality_factor
        cov_rot = base_cov_rot / quality_factor
        
        # 如果有特征分布信息，进一步调整旋转不确定性
        if feature_points is not None and len(feature_points) >= 4:
            distribution_factor = self._analyze_feature_distribution(feature_points, image_size)
            # 分布越不均匀 → 某些方向的旋转估计越不可靠
            cov_rot = cov_rot / distribution_factor
            
        # 如果有深度信息，调整 Z 方向平移的不确定性
        if depths is not None and len(depths) > 0:
            depth_factor = self._analyze_depth_distribution(depths)
            # Z 方向不确定性与平均深度相关
            cov_trans[2, 2] *= depth_factor
            
        # 组装完整协方差矩阵
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[:3, :3] = cov_trans
        cov_6x6[3:, 3:] = cov_rot
        
        # 信息矩阵 = 协方差的逆
        info_matrix = np.linalg.inv(cov_6x6 + 1e-10 * np.eye(6))
        
        return info_matrix
        
    def _compute_quality_factor(self, stats: PoseEstimationStatistics) -> float:
        """
        计算位姿估计的质量因子 (0, ∞)。
        
        质量因子 = f(内点数) × f(内点比) × f(重投影误差) × f(跟踪长度)
        """
        # 内点数因子：sigmoid 函数，在 min_inliers 处约为 0.5
        inlier_factor = 1.0 / (1.0 + np.exp(-0.1 * (stats.n_inliers - self.min_inliers_for_good)))
        
        # 内点比因子：高内点比表示少异常值
        ratio_factor = np.clip(stats.inlier_ratio / 0.5, 0.2, 2.0)
        
        # 重投影误差因子：误差越小越好
        reproj_factor = self.max_reproj_for_good / (stats.mean_reproj_error + 0.1)
        reproj_factor = np.clip(reproj_factor, 0.2, 3.0)
        
        # 跟踪长度因子：长轨迹更可靠
        track_factor = np.clip(stats.track_length_mean / 3.0, 0.5, 2.0)
        
        # IMU 辅助加成
        imu_factor = 1.5 if stats.has_imu else 1.0
        
        # 组合
        quality = inlier_factor * ratio_factor * reproj_factor * track_factor * imu_factor
        
        # 确保质量因子有下界
        return max(quality, 0.1)
        
    def _analyze_feature_distribution(
        self,
        points: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        分析特征点分布的均匀性。
        
        返回 3×3 对角矩阵，表示三个旋转轴的可观测性。
        
        原理：
        - 特征点集中在图像中心 → roll 估计差
        - 特征点集中在水平线上 → pitch 估计差
        - 特征点集中在垂直线上 → yaw 估计差
        """
        W, H = image_size
        cx, cy = W / 2, H / 2
        
        # 归一化坐标
        points_norm = points.copy()
        points_norm[:, 0] = (points[:, 0] - cx) / W
        points_norm[:, 1] = (points[:, 1] - cy) / H
        
        # 计算分布矩阵（协方差）
        cov_points = np.cov(points_norm.T)
        if cov_points.shape == ():
            cov_points = np.array([[cov_points]])
            
        # 特征值表示分布的主方向
        eigenvalues = np.linalg.eigvalsh(cov_points)
        
        # 分布因子：特征值越大越均匀
        # 这里简化处理，返回标量
        distribution_factor = np.clip(np.mean(eigenvalues) * 100, 0.2, 2.0)
        
        return np.eye(3) * distribution_factor
        
    def _analyze_depth_distribution(self, depths: np.ndarray) -> float:
        """
        分析深度分布。
        
        返回 Z 方向不确定性的放大因子。
        
        原理：
        - 深度变化大 → 视差变化大 → Z 估计更准
        - 所有特征在同一深度 → Z 估计退化
        """
        if len(depths) < 2:
            return 2.0  # 高不确定性
            
        # 深度变化范围
        depth_range = np.ptp(depths)  # max - min
        mean_depth = np.mean(depths)
        
        # 相对变化
        relative_range = depth_range / (mean_depth + 1e-3)
        
        # 范围越大 → 因子越小 → Z 不确定性越低
        depth_factor = 1.0 / (relative_range + 0.1)
        depth_factor = np.clip(depth_factor, 0.5, 5.0)
        
        # 同时考虑平均深度：远处物体 Z 不确定性更大
        depth_penalty = (mean_depth / 2.0) ** 2  # 2米为基准
        depth_penalty = np.clip(depth_penalty, 1.0, 10.0)
        
        return depth_factor * depth_penalty


class LoopClosureInformationEstimator:
    """
    回环约束的信息矩阵估计器。
    
    回环约束的质量取决于：
    1. 几何验证的内点数
    2. 重投影误差
    3. 两帧之间的共视特征数量
    4. 时间跨度（跨度越大的有效回环越可信）
    """
    
    def __init__(
        self,
        base_sigma_trans: float = 0.05,   # 回环平移基础误差
        base_sigma_rot: float = 0.01,     # 回环旋转基础误差
        min_inliers: int = 30,            # 最小内点数
        min_time_gap: float = 10.0        # 最小时间间隔 (秒)
    ):
        self.base_sigma_trans = base_sigma_trans
        self.base_sigma_rot = base_sigma_rot
        self.min_inliers = min_inliers
        self.min_time_gap = min_time_gap
        
    def estimate(
        self,
        n_inliers: int,
        mean_reproj_error: float,
        time_gap: float,
        relative_pose_error: Optional[float] = None
    ) -> np.ndarray:
        """
        估计回环约束的信息矩阵。
        
        Args:
            n_inliers: 几何验证的内点数
            mean_reproj_error: 平均重投影误差
            time_gap: 两帧的时间间隔
            relative_pose_error: 可选的位姿误差估计
            
        Returns:
            info_matrix: 6×6 信息矩阵
        """
        # 基础协方差
        sigma_trans = self.base_sigma_trans
        sigma_rot = self.base_sigma_rot
        
        # 内点数因子
        inlier_factor = np.clip(n_inliers / self.min_inliers, 0.5, 3.0)
        
        # 重投影误差因子
        reproj_factor = 1.0 / (mean_reproj_error + 0.5)
        reproj_factor = np.clip(reproj_factor, 0.3, 2.0)
        
        # 时间跨度因子：更长跨度的回环如果验证通过则更可信
        time_factor = np.log(time_gap / self.min_time_gap + 1) + 1
        time_factor = np.clip(time_factor, 1.0, 2.0)
        
        # 组合质量因子
        quality = inlier_factor * reproj_factor * time_factor
        
        # 调整协方差
        cov_trans = np.eye(3) * (sigma_trans ** 2) / quality
        cov_rot = np.eye(3) * (sigma_rot ** 2) / quality
        
        # 组装
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[:3, :3] = cov_trans
        cov_6x6[3:, 3:] = cov_rot
        
        # 信息矩阵
        info_matrix = np.linalg.inv(cov_6x6 + 1e-10 * np.eye(6))
        
        return info_matrix
        
    def validate_loop_closure(
        self,
        n_inliers: int,
        inlier_ratio: float,
        reproj_error: float,
        time_gap: float
    ) -> Tuple[bool, float]:
        """
        验证回环约束的有效性。
        
        Returns:
            (is_valid, confidence)
        """
        # 多重条件检查
        conditions = []
        
        # 条件1：足够的内点
        conditions.append(n_inliers >= self.min_inliers)
        
        # 条件2：合理的内点比
        conditions.append(inlier_ratio >= 0.3)
        
        # 条件3：小的重投影误差
        conditions.append(reproj_error < 3.0)
        
        # 条件4：足够的时间跨度（排除连续帧）
        conditions.append(time_gap >= self.min_time_gap)
        
        is_valid = all(conditions)
        
        # 置信度
        confidence = (
            np.clip(n_inliers / 100, 0, 1) * 0.3 +
            np.clip(inlier_ratio, 0, 1) * 0.3 +
            np.clip(2.0 / (reproj_error + 0.5), 0, 1) * 0.2 +
            np.clip(time_gap / 60, 0, 1) * 0.2
        )
        
        return is_valid, confidence


# =============================================================================
# 协方差传播工具
# =============================================================================

def propagate_pose_covariance(
    cov_prev: np.ndarray,
    transform: np.ndarray,
    cov_transform: np.ndarray
) -> np.ndarray:
    """
    传播位姿协方差。
    
    给定：
    - 上一帧位姿协方差 Σ_{k-1}
    - 帧间变换 ΔT 及其协方差 Σ_Δ
    
    计算当前帧位姿协方差：
    Σ_k = Ad(ΔT) Σ_{k-1} Ad(ΔT)^T + Σ_Δ
    
    Args:
        cov_prev: 上一帧位姿协方差 (6×6)
        transform: 帧间变换 (4×4)
        cov_transform: 帧间变换协方差 (6×6)
        
    Returns:
        cov_curr: 当前帧位姿协方差 (6×6)
    """
    # 简化处理：直接叠加协方差（忽略 adjoint 的旋转效应）
    # 这对于小角度变化是合理的近似
    
    R = transform[:3, :3]
    
    # 旋转协方差的平移部分
    cov_trans_rotated = R @ cov_prev[:3, :3] @ R.T
    cov_rot_rotated = R @ cov_prev[3:, 3:] @ R.T
    
    cov_curr = np.zeros((6, 6))
    cov_curr[:3, :3] = cov_trans_rotated + cov_transform[:3, :3]
    cov_curr[3:, 3:] = cov_rot_rotated + cov_transform[3:, 3:]
    
    return cov_curr


def compute_information_from_hessian(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    从 PnP 的 Hessian 矩阵计算信息矩阵。
    
    原理：最小二乘问题的 Hessian 矩阵（在最优解处）近似于信息矩阵。
    
    H ≈ J^T W J
    
    其中 J 是雅可比矩阵，W 是权重矩阵。
    """
    if inlier_mask is not None:
        obj_pts = object_points[inlier_mask]
        img_pts = image_points[inlier_mask]
    else:
        obj_pts = object_points
        img_pts = image_points
        
    n_points = len(obj_pts)
    if n_points < 6:
        return np.eye(6)  # 退化情况
        
    # 数值计算雅可比矩阵
    eps = 1e-6
    jacobian = np.zeros((n_points * 2, 6))
    
    for i in range(6):
        # 正向扰动
        rvec_plus = rvec.copy()
        tvec_plus = tvec.copy()
        if i < 3:
            tvec_plus[i] += eps
        else:
            rvec_plus[i - 3] += eps
            
        proj_plus, _ = cv2.projectPoints(obj_pts, rvec_plus, tvec_plus, camera_matrix, None)
        
        # 负向扰动
        rvec_minus = rvec.copy()
        tvec_minus = tvec.copy()
        if i < 3:
            tvec_minus[i] -= eps
        else:
            rvec_minus[i - 3] -= eps
            
        proj_minus, _ = cv2.projectPoints(obj_pts, rvec_minus, tvec_minus, camera_matrix, None)
        
        # 差分
        diff = (proj_plus.reshape(-1, 2) - proj_minus.reshape(-1, 2)) / (2 * eps)
        jacobian[:, i] = diff.flatten()
        
    # Hessian = J^T J
    hessian = jacobian.T @ jacobian
    
    # 正则化
    hessian += 1e-6 * np.eye(6)
    
    return hessian


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive Information Matrix Test")
    print("=" * 60)
    
    # 测试里程计信息矩阵
    estimator = AdaptiveInformationMatrix()
    
    # 高质量估计
    stats_good = PoseEstimationStatistics(
        n_features=500,
        n_matches=300,
        n_inliers=200,
        inlier_ratio=0.8,
        mean_reproj_error=0.5,
        track_length_mean=5.0,
        has_imu=True
    )
    
    info_good = estimator.estimate_from_pnp(stats_good)
    print("\nHigh-quality estimation:")
    print(f"  Information matrix diagonal: {np.diag(info_good)}")
    
    # 低质量估计
    stats_bad = PoseEstimationStatistics(
        n_features=100,
        n_matches=50,
        n_inliers=20,
        inlier_ratio=0.4,
        mean_reproj_error=3.0,
        track_length_mean=2.0,
        has_imu=False
    )
    
    info_bad = estimator.estimate_from_pnp(stats_bad)
    print("\nLow-quality estimation:")
    print(f"  Information matrix diagonal: {np.diag(info_bad)}")
    
    # 比较
    ratio = np.diag(info_good) / np.diag(info_bad)
    print(f"\nQuality ratio (good/bad): {ratio}")
    
    # 测试回环信息矩阵
    loop_estimator = LoopClosureInformationEstimator()
    
    info_loop = loop_estimator.estimate(
        n_inliers=80,
        mean_reproj_error=1.2,
        time_gap=30.0
    )
    print("\nLoop closure information diagonal:", np.diag(info_loop))
    
    is_valid, conf = loop_estimator.validate_loop_closure(
        n_inliers=80,
        inlier_ratio=0.6,
        reproj_error=1.2,
        time_gap=30.0
    )
    print(f"Loop validation: valid={is_valid}, confidence={conf:.2f}")
