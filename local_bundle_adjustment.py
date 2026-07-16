#!/usr/bin/env python3
"""
================================================================================
Local Bundle Adjustment for Monocular SLAM
================================================================================

学术背景：
---------
Bundle Adjustment (BA) 是 SLAM 后端优化的"黄金标准"。它联合优化相机位姿
和地图点，最小化重投影误差。相比只优化位姿的 Pose Graph Optimization，
BA 能够：

1. 修正地图点位置 → 直接提升建图精度
2. 利用多视图几何约束 → 更强的全局一致性
3. 传播不确定性 → 更准确的协方差估计

本模块实现滑动窗口 BA，在计算效率和精度之间取得平衡。

数学模型：
---------
目标函数：
    L = Σ_{i,j} ρ( ||u_ij - π(T_i, p_j)||²_Σ ) + Σ_i ||T_i ⊖ T_i^prior||²_Λ

其中：
- u_ij: 关键帧 i 对地图点 j 的观测（2D像素坐标）
- π(T, p): 投影函数，将3D点 p 通过位姿 T 投影到图像
- ρ(·): Huber 或 Cauchy 鲁棒核函数
- T_i^prior: 来自视觉里程计的先验位姿
- Λ: 先验约束的信息矩阵

参考文献：
---------
[1] Triggs et al., "Bundle Adjustment - A Modern Synthesis", Vision Algorithms 1999
[2] Kummerle et al., "g2o: A General Framework for Graph Optimization", ICRA 2011
[3] Engel et al., "Direct Sparse Odometry", TPAMI 2017

Author: Academic Enhancement
================================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import cv2
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class MapPoint:
    """
    地图点数据结构。
    
    Attributes:
        id: 唯一标识符
        position: 世界坐标系下的 3D 位置
        covariance: 位置不确定性 (3×3)
        observations: {keyframe_id: (u, v)} 观测记录
        descriptor: 代表性描述子
        track_length: 被观测的帧数
        is_valid: 是否有效（未被裁剪）
    """
    id: int
    position: np.ndarray  # (3,)
    covariance: Optional[np.ndarray] = None  # (3, 3)
    observations: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    descriptor: Optional[np.ndarray] = None
    track_length: int = 1
    is_valid: bool = True
    

@dataclass
class BAKeyframe:
    """
    BA 中的关键帧表示。
    
    Attributes:
        id: 关键帧 ID
        pose: SE(3) 位姿 (4×4)
        pose_prior: 来自 VO 的先验位姿
        fixed: 是否固定（不优化）
        observed_points: 观测到的地图点 ID 集合
    """
    id: int
    pose: np.ndarray  # T_wc (4×4)
    pose_prior: Optional[np.ndarray] = None
    fixed: bool = False
    observed_points: Set[int] = field(default_factory=set)


# =============================================================================
# 投影与雅可比计算
# =============================================================================

def project_point(point_3d: np.ndarray, T_wc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将世界坐标系下的 3D 点投影到图像。
    
    Args:
        point_3d: 世界坐标 (3,)
        T_wc: 世界到相机的变换 (4×4)，即 T_cw 的逆
        K: 相机内参 (3×3)
        
    Returns:
        pixel: 像素坐标 (2,)
    """
    # 变换到相机坐标系
    T_cw = np.linalg.inv(T_wc)
    p_cam = T_cw[:3, :3] @ point_3d + T_cw[:3, 3]
    
    # 投影
    p_norm = p_cam[:2] / p_cam[2]
    pixel = K[:2, :2] @ p_norm + K[:2, 2]
    
    return pixel


def compute_jacobian_pose(point_3d: np.ndarray, T_wc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    计算重投影误差对位姿的雅可比矩阵。
    
    使用李代数参数化：δξ = [δρ, δφ]^T ∈ se(3)
    其中 δρ 是平移扰动，δφ 是旋转扰动
    
    Returns:
        J_pose: (2×6) 雅可比矩阵
    """
    # 变换点到相机坐标系
    T_cw = np.linalg.inv(T_wc)
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    p_cam = R_cw @ point_3d + t_cw
    
    X, Y, Z = p_cam
    fx, fy = K[0, 0], K[1, 1]
    
    # 投影雅可比 ∂π/∂p_cam
    J_proj = np.array([
        [fx / Z, 0, -fx * X / (Z**2)],
        [0, fy / Z, -fy * Y / (Z**2)]
    ])
    
    # 李代数扰动雅可比
    # 对于 T_cw ⊕ exp(δξ)，扰动作用于 p_cam
    # ∂p_cam/∂δρ = I (平移)
    # ∂p_cam/∂δφ = -[p_cam]_× (旋转的叉乘矩阵)
    
    p_skew = np.array([
        [0, -Z, Y],
        [Z, 0, -X],
        [-Y, X, 0]
    ])
    
    # J = J_proj @ [I | -[p]_×]
    J_pose = np.zeros((2, 6))
    J_pose[:, :3] = J_proj  # 对平移
    J_pose[:, 3:] = J_proj @ (-p_skew)  # 对旋转
    
    return J_pose


def compute_jacobian_point(point_3d: np.ndarray, T_wc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    计算重投影误差对地图点位置的雅可比矩阵。
    
    Returns:
        J_point: (2×3) 雅可比矩阵
    """
    T_cw = np.linalg.inv(T_wc)
    R_cw = T_cw[:3, :3]
    p_cam = R_cw @ point_3d + T_cw[:3, 3]
    
    X, Y, Z = p_cam
    fx, fy = K[0, 0], K[1, 1]
    
    # ∂π/∂p_cam
    J_proj = np.array([
        [fx / Z, 0, -fx * X / (Z**2)],
        [0, fy / Z, -fy * Y / (Z**2)]
    ])
    
    # ∂p_cam/∂p_world = R_cw
    J_point = J_proj @ R_cw
    
    return J_point


# =============================================================================
# 鲁棒核函数
# =============================================================================

def huber_weight(residual: float, delta: float = 1.0) -> Tuple[float, float]:
    """
    Huber 核函数及其权重。
    
    ρ(r) = {
        0.5 * r²,           if |r| ≤ δ
        δ * (|r| - 0.5δ),   if |r| > δ
    }
    
    权重 w = ρ'(r) / r
    
    Returns:
        (weighted_residual, weight)
    """
    r_abs = abs(residual)
    if r_abs <= delta:
        return 0.5 * residual**2, 1.0
    else:
        return delta * (r_abs - 0.5 * delta), delta / r_abs


def cauchy_weight(residual: float, c: float = 1.0) -> Tuple[float, float]:
    """
    Cauchy (柯西) 核函数，对异常值更加鲁棒。
    
    ρ(r) = (c²/2) * log(1 + (r/c)²)
    权重 w = 1 / (1 + (r/c)²)
    """
    r_over_c_sq = (residual / c) ** 2
    cost = 0.5 * c**2 * np.log(1 + r_over_c_sq)
    weight = 1.0 / (1 + r_over_c_sq)
    return cost, weight


# =============================================================================
# 滑动窗口 Bundle Adjustment
# =============================================================================

class SlidingWindowBA:
    """
    滑动窗口 Bundle Adjustment 优化器。
    
    核心特点：
    1. 只优化最近 N 个关键帧（窗口大小）
    2. 窗口外的关键帧固定不动，但仍提供约束
    3. 使用稀疏 Schur 补分解高效求解
    4. 鲁棒核函数处理异常值
    """
    
    def __init__(
        self,
        window_size: int = 10,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        huber_delta: float = 1.0,
        use_cauchy: bool = False,
        fix_scale: bool = False
    ):
        """
        Args:
            window_size: 滑动窗口大小
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            huber_delta: Huber 核参数
            use_cauchy: 是否使用 Cauchy 核（更鲁棒）
            fix_scale: 是否固定尺度（单目 SLAM 通常需要）
        """
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.huber_delta = huber_delta
        self.use_cauchy = use_cauchy
        self.fix_scale = fix_scale
        
        # 状态存储
        self.keyframes: Dict[int, BAKeyframe] = {}
        self.map_points: Dict[int, MapPoint] = {}
        
        # 相机内参
        self.K = np.eye(3)
        
        # 统计
        self.optimization_count = 0
        self.total_time = 0.0
        
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """设置相机内参"""
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
    def add_keyframe(
        self,
        kf_id: int,
        pose: np.ndarray,
        pose_prior: Optional[np.ndarray] = None,
        fixed: bool = False
    ):
        """添加关键帧"""
        self.keyframes[kf_id] = BAKeyframe(
            id=kf_id,
            pose=pose.copy(),
            pose_prior=pose_prior.copy() if pose_prior is not None else None,
            fixed=fixed
        )
        
    def add_map_point(
        self,
        point_id: int,
        position: np.ndarray,
        descriptor: Optional[np.ndarray] = None
    ):
        """添加地图点"""
        self.map_points[point_id] = MapPoint(
            id=point_id,
            position=position.copy(),
            descriptor=descriptor
        )
        
    def add_observation(
        self,
        kf_id: int,
        point_id: int,
        pixel: Tuple[float, float],
        uncertainty: float = 1.0
    ):
        """
        添加观测（关键帧看到了某个地图点）。
        
        Args:
            kf_id: 关键帧 ID
            point_id: 地图点 ID
            pixel: 观测像素坐标 (u, v)
            uncertainty: 观测不确定性（像素标准差）
        """
        if kf_id not in self.keyframes or point_id not in self.map_points:
            return
            
        self.map_points[point_id].observations[kf_id] = pixel
        self.map_points[point_id].track_length = len(self.map_points[point_id].observations)
        self.keyframes[kf_id].observed_points.add(point_id)
        
    def get_window_keyframes(self) -> List[int]:
        """获取当前滑动窗口内的关键帧 ID"""
        sorted_ids = sorted(self.keyframes.keys(), reverse=True)
        return sorted_ids[:self.window_size]
        
    def optimize(self, verbose: bool = False) -> Dict:
        """
        执行滑动窗口 BA 优化。
        
        使用 Levenberg-Marquardt 算法求解非线性最小二乘问题。
        
        Returns:
            info: 优化统计信息
        """
        import time
        start_time = time.time()
        
        # 获取窗口内的关键帧
        window_kf_ids = self.get_window_keyframes()
        
        if len(window_kf_ids) < 2:
            return {'success': False, 'reason': 'not_enough_keyframes'}
            
        # 收集窗口内观测到的所有地图点
        point_ids_in_window = set()
        for kf_id in window_kf_ids:
            point_ids_in_window.update(self.keyframes[kf_id].observed_points)
            
        # 过滤：只保留被至少 2 个窗口内关键帧观测的点
        valid_point_ids = []
        for pid in point_ids_in_window:
            if pid not in self.map_points:
                continue
            mp = self.map_points[pid]
            n_obs_in_window = sum(1 for kf in mp.observations if kf in window_kf_ids)
            if n_obs_in_window >= 2:
                valid_point_ids.append(pid)
                
        if len(valid_point_ids) < 10:
            return {'success': False, 'reason': 'not_enough_points'}
            
        # 建立索引映射
        # 关键帧：前6维参数（李代数）
        # 地图点：后3维参数
        n_poses = len(window_kf_ids)
        n_points = len(valid_point_ids)
        
        kf_id_to_idx = {kf_id: i for i, kf_id in enumerate(window_kf_ids)}
        pt_id_to_idx = {pt_id: i for i, pt_id in enumerate(valid_point_ids)}
        
        # 确定固定帧（第一帧固定以消除 gauge 自由度）
        fixed_kf_idx = {0}  # 第一帧固定
        
        # LM 参数
        lambda_lm = 1e-3
        lambda_max = 1e10
        
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # 计算残差和雅可比
            residuals = []
            jacobians_pose = []
            jacobians_point = []
            row_indices = []  # (pose_idx, point_idx)
            
            total_cost = 0.0
            
            for pt_id in valid_point_ids:
                mp = self.map_points[pt_id]
                pt_idx = pt_id_to_idx[pt_id]
                
                for kf_id, obs_pixel in mp.observations.items():
                    if kf_id not in kf_id_to_idx:
                        continue
                        
                    kf_idx = kf_id_to_idx[kf_id]
                    kf = self.keyframes[kf_id]
                    
                    # 投影
                    proj = project_point(mp.position, kf.pose, self.K)
                    
                    # 残差
                    residual = np.array(obs_pixel) - proj
                    
                    # 鲁棒权重
                    r_norm = np.linalg.norm(residual)
                    if self.use_cauchy:
                        _, weight = cauchy_weight(r_norm, self.huber_delta)
                    else:
                        _, weight = huber_weight(r_norm, self.huber_delta)
                        
                    sqrt_weight = np.sqrt(weight)
                    
                    # 加权残差
                    weighted_residual = sqrt_weight * residual
                    residuals.append(weighted_residual)
                    total_cost += 0.5 * np.sum(weighted_residual**2)
                    
                    # 雅可比
                    J_pose = sqrt_weight * compute_jacobian_pose(mp.position, kf.pose, self.K)
                    J_point = sqrt_weight * compute_jacobian_point(mp.position, kf.pose, self.K)
                    
                    jacobians_pose.append(J_pose)
                    jacobians_point.append(J_point)
                    row_indices.append((kf_idx, pt_idx))
                    
            if len(residuals) == 0:
                break
                
            # 构建稀疏海森矩阵（Schur 补方法）
            # H = [H_pp  H_pl]
            #     [H_lp  H_ll]
            #
            # 用 Schur 补消去地图点：
            # (H_pp - H_pl * H_ll^{-1} * H_lp) * δξ = b_p - H_pl * H_ll^{-1} * b_l
            
            n_residuals = len(residuals)
            pose_dim = 6
            point_dim = 3
            
            # 海森块
            H_pp = np.zeros((n_poses * pose_dim, n_poses * pose_dim))
            H_ll = np.zeros((n_points * point_dim, n_points * point_dim))
            H_pl = np.zeros((n_poses * pose_dim, n_points * point_dim))
            
            b_p = np.zeros(n_poses * pose_dim)
            b_l = np.zeros(n_points * point_dim)
            
            for i, (kf_idx, pt_idx) in enumerate(row_indices):
                J_p = jacobians_pose[i]  # 2×6
                J_l = jacobians_point[i]  # 2×3
                r = residuals[i]  # 2
                
                # 位姿块索引
                p_start = kf_idx * pose_dim
                p_end = p_start + pose_dim
                
                # 地图点块索引
                l_start = pt_idx * point_dim
                l_end = l_start + point_dim
                
                # 累加海森矩阵
                H_pp[p_start:p_end, p_start:p_end] += J_p.T @ J_p
                H_ll[l_start:l_end, l_start:l_end] += J_l.T @ J_l
                H_pl[p_start:p_end, l_start:l_end] += J_p.T @ J_l
                
                # 累加右端项
                b_p[p_start:p_end] -= J_p.T @ r
                b_l[l_start:l_end] -= J_l.T @ r
                
            # 添加 LM 阻尼
            H_pp += lambda_lm * np.diag(np.diag(H_pp) + 1e-6)
            
            # 对地图点海森矩阵添加阻尼并求逆
            for pt_idx in range(n_points):
                l_start = pt_idx * point_dim
                l_end = l_start + point_dim
                H_ll[l_start:l_end, l_start:l_end] += lambda_lm * np.eye(point_dim)
                
            # Schur 补求解
            try:
                # 计算 H_ll 的逆（块对角，可以分块求逆）
                H_ll_inv_blocks = []
                for pt_idx in range(n_points):
                    l_start = pt_idx * point_dim
                    l_end = l_start + point_dim
                    H_ll_block = H_ll[l_start:l_end, l_start:l_end]
                    H_ll_inv_blocks.append(np.linalg.inv(H_ll_block + 1e-6 * np.eye(point_dim)))
                    
                # 构造完整的 H_ll_inv
                H_ll_inv = np.zeros_like(H_ll)
                for pt_idx, block_inv in enumerate(H_ll_inv_blocks):
                    l_start = pt_idx * point_dim
                    l_end = l_start + point_dim
                    H_ll_inv[l_start:l_end, l_start:l_end] = block_inv
                    
                # Schur 补
                H_schur = H_pp - H_pl @ H_ll_inv @ H_pl.T
                b_schur = b_p - H_pl @ H_ll_inv @ b_l
                
                # 固定帧处理：将对应行列置零，对角线置1
                for fixed_idx in fixed_kf_idx:
                    p_start = fixed_idx * pose_dim
                    p_end = p_start + pose_dim
                    H_schur[p_start:p_end, :] = 0
                    H_schur[:, p_start:p_end] = 0
                    H_schur[p_start:p_end, p_start:p_end] = np.eye(pose_dim)
                    b_schur[p_start:p_end] = 0
                    
                # 求解位姿增量
                delta_pose = np.linalg.solve(H_schur + 1e-8 * np.eye(H_schur.shape[0]), b_schur)
                
                # 反代求地图点增量
                delta_point = H_ll_inv @ (b_l - H_pl.T @ delta_pose)
                
            except np.linalg.LinAlgError:
                lambda_lm *= 10
                continue
                
            # 尝试更新
            new_poses = {}
            new_points = {}
            
            for i, kf_id in enumerate(window_kf_ids):
                if i in fixed_kf_idx:
                    new_poses[kf_id] = self.keyframes[kf_id].pose.copy()
                    continue
                    
                delta = delta_pose[i*pose_dim:(i+1)*pose_dim]
                new_poses[kf_id] = self._apply_pose_increment(
                    self.keyframes[kf_id].pose, delta
                )
                
            for i, pt_id in enumerate(valid_point_ids):
                delta = delta_point[i*point_dim:(i+1)*point_dim]
                new_points[pt_id] = self.map_points[pt_id].position + delta
                
            # 计算新代价
            new_cost = self._compute_cost(new_poses, new_points, window_kf_ids, valid_point_ids)
            
            # LM 策略
            if new_cost < total_cost:
                # 接受更新
                for kf_id, pose in new_poses.items():
                    self.keyframes[kf_id].pose = pose
                for pt_id, pos in new_points.items():
                    self.map_points[pt_id].position = pos
                    
                lambda_lm = max(lambda_lm / 10, 1e-7)
                
                if verbose:
                    print(f"  Iter {iteration}: cost {total_cost:.4f} -> {new_cost:.4f}, λ={lambda_lm:.2e}")
                    
                if abs(total_cost - new_cost) / total_cost < self.convergence_threshold:
                    break
                    
                prev_cost = new_cost
            else:
                # 拒绝更新，增大阻尼
                lambda_lm = min(lambda_lm * 10, lambda_max)
                
        elapsed = time.time() - start_time
        self.optimization_count += 1
        self.total_time += elapsed
        
        return {
            'success': True,
            'iterations': iteration + 1,
            'final_cost': prev_cost,
            'n_keyframes': n_poses,
            'n_points': n_points,
            'time': elapsed
        }
        
    def _apply_pose_increment(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        将李代数增量应用到位姿。
        
        delta = [δρ, δφ] ∈ se(3)
        T_new = T_old * exp(δξ)
        """
        delta_rho = delta[:3]  # 平移
        delta_phi = delta[3:]  # 旋转
        
        # 构造增量变换矩阵
        angle = np.linalg.norm(delta_phi)
        if angle < 1e-6:
            delta_R = np.eye(3)
        else:
            axis = delta_phi / angle
            delta_R = Rotation.from_rotvec(delta_phi).as_matrix()
            
        delta_T = np.eye(4)
        delta_T[:3, :3] = delta_R
        delta_T[:3, 3] = delta_rho
        
        # 右乘更新
        new_pose = pose @ delta_T
        
        return new_pose
        
    def _compute_cost(
        self,
        poses: Dict[int, np.ndarray],
        points: Dict[int, np.ndarray],
        kf_ids: List[int],
        point_ids: List[int]
    ) -> float:
        """计算总代价"""
        total_cost = 0.0
        
        for pt_id in point_ids:
            if pt_id not in self.map_points:
                continue
            mp = self.map_points[pt_id]
            pos = points.get(pt_id, mp.position)
            
            for kf_id, obs_pixel in mp.observations.items():
                if kf_id not in poses:
                    continue
                    
                pose = poses[kf_id]
                proj = project_point(pos, pose, self.K)
                residual = np.array(obs_pixel) - proj
                r_norm = np.linalg.norm(residual)
                
                if self.use_cauchy:
                    cost, _ = cauchy_weight(r_norm, self.huber_delta)
                else:
                    cost, _ = huber_weight(r_norm, self.huber_delta)
                    
                total_cost += cost
                
        return total_cost
    
    def get_optimized_poses(self) -> Dict[int, np.ndarray]:
        """获取优化后的位姿"""
        return {kf_id: kf.pose.copy() for kf_id, kf in self.keyframes.items()}
    
    def get_optimized_points(self) -> Dict[int, np.ndarray]:
        """获取优化后的地图点"""
        return {pt_id: mp.position.copy() for pt_id, mp in self.map_points.items() if mp.is_valid}
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'n_keyframes': len(self.keyframes),
            'n_map_points': len(self.map_points),
            'n_valid_points': sum(1 for mp in self.map_points.values() if mp.is_valid),
            'optimization_count': self.optimization_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(self.optimization_count, 1)
        }


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sliding Window Bundle Adjustment Test")
    print("=" * 60)
    
    # 创建 BA 优化器
    ba = SlidingWindowBA(window_size=5, max_iterations=20)
    ba.set_intrinsics(fx=500, fy=500, cx=320, cy=240)
    
    # 模拟场景：相机沿直线运动，观察固定的 3D 点
    np.random.seed(42)
    
    # 生成地图点（世界坐标）
    n_points = 50
    map_points_gt = np.random.uniform(-2, 2, (n_points, 3))
    map_points_gt[:, 2] += 3  # 确保在相机前方
    
    for i, pos in enumerate(map_points_gt):
        ba.add_map_point(i, pos + np.random.normal(0, 0.1, 3))  # 加噪声
        
    # 生成关键帧（沿 X 轴移动）
    n_keyframes = 8
    for kf_id in range(n_keyframes):
        # 真实位姿
        T_wc = np.eye(4)
        T_wc[0, 3] = kf_id * 0.3  # X 方向移动
        
        # 加入位姿噪声
        noise_R = Rotation.from_rotvec(np.random.normal(0, 0.01, 3)).as_matrix()
        noise_t = np.random.normal(0, 0.02, 3)
        
        T_noisy = T_wc.copy()
        T_noisy[:3, :3] = T_wc[:3, :3] @ noise_R
        T_noisy[:3, 3] += noise_t
        
        ba.add_keyframe(kf_id, T_noisy, pose_prior=T_noisy, fixed=(kf_id == 0))
        
        # 添加观测
        K = ba.K
        for pt_id, pos in enumerate(map_points_gt):
            proj = project_point(pos, T_wc, K)
            if 0 < proj[0] < 640 and 0 < proj[1] < 480:
                # 加观测噪声
                obs_noise = np.random.normal(0, 1.0, 2)
                ba.add_observation(kf_id, pt_id, tuple(proj + obs_noise))
                
    # 执行优化
    print("\nRunning optimization...")
    result = ba.optimize(verbose=True)
    
    print(f"\nResult: {result}")
    print(f"Statistics: {ba.get_statistics()}")
