#!/usr/bin/env python3
"""
================================================================================
Sim(3) Pose Graph Optimization for Monocular SLAM
================================================================================

解决问题：
---------
单目 SLAM 的核心问题是尺度漂移。标准的 SE(3) 位姿图优化（如 Open3D）只能
优化 6-DOF 位姿（旋转 + 平移），无法修正尺度差异。

当回环闭合检测到"同一地点"但尺度不同时：
- SE(3) 优化：只修正位姿 → 点云分裂（您现在的问题）
- Sim(3) 优化：同时修正位姿和尺度 → 点云正确对齐

数学模型：
---------
Sim(3) 变换群：
    T = [s*R  t]
        [0    1]

其中 s 是尺度因子。

优化目标：
    min Σ_{ij} ||log(T_i^{-1} T_j Z_ij^{-1})||²_Ω

其中 Z_ij 是相对约束（包含尺度），Ω 是信息矩阵。

参考文献：
---------
[1] Strasdat et al., "Scale Drift-Aware Large Scale Monocular SLAM", RSS 2010
[2] Kümmerle et al., "g2o: A General Framework for Graph Optimization", ICRA 2011

Author: Academic Enhancement for Monocular SLAM
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import copy


# =============================================================================
# Sim(3) 李群操作
# =============================================================================

class Sim3:
    """
    Sim(3) 相似变换群。
    
    表示 7-DOF 变换：3 旋转 + 3 平移 + 1 尺度
    
    矩阵形式：
        T = [s*R  t]
            [0    1]
    """
    
    def __init__(self, rotation: np.ndarray = None, translation: np.ndarray = None, scale: float = 1.0):
        """
        Args:
            rotation: 3x3 旋转矩阵或 (3,) 旋转向量
            translation: (3,) 平移向量
            scale: 尺度因子
        """
        if rotation is None:
            self.R = np.eye(3)
        elif rotation.shape == (3,):
            self.R = Rotation.from_rotvec(rotation).as_matrix()
        else:
            self.R = rotation.copy()
            
        if translation is None:
            self.t = np.zeros(3)
        else:
            self.t = translation.copy()
            
        self.s = scale
        
    def matrix(self) -> np.ndarray:
        """返回 4x4 变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.s * self.R
        T[:3, 3] = self.t
        return T
    
    @staticmethod
    def from_matrix(T: np.ndarray) -> 'Sim3':
        """从 4x4 矩阵构造 Sim(3)"""
        sR = T[:3, :3]
        t = T[:3, 3]
        
        # 提取尺度（通过 SVD 或行列式）
        s = np.cbrt(np.linalg.det(sR))  # det(sR) = s³ det(R) = s³
        
        if s < 1e-6:
            s = 1.0
            R = np.eye(3)
        else:
            R = sR / s
            # 确保正交性
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            
        return Sim3(R, t, s)
    
    def inverse(self) -> 'Sim3':
        """返回逆变换"""
        R_inv = self.R.T
        s_inv = 1.0 / self.s
        t_inv = -s_inv * (R_inv @ self.t)
        return Sim3(R_inv, t_inv, s_inv)
    
    def __matmul__(self, other: 'Sim3') -> 'Sim3':
        """复合两个 Sim(3) 变换: self ∘ other"""
        R_new = self.R @ other.R
        t_new = self.s * (self.R @ other.t) + self.t
        s_new = self.s * other.s
        return Sim3(R_new, t_new, s_new)
    
    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """变换 3D 点: p' = s*R*p + t"""
        return self.s * (self.R @ p) + self.t
    
    def log(self) -> np.ndarray:
        """
        李代数映射: Sim(3) → sim(3) ∈ R^7
        
        返回 7 维向量 [ω, υ, σ] 其中：
        - ω: 旋转向量 (3)
        - υ: 平移相关 (3)
        - σ: 尺度对数 (1)
        """
        # 旋转向量
        rot = Rotation.from_matrix(self.R)
        omega = rot.as_rotvec()
        
        # 尺度对数
        sigma = np.log(self.s) if self.s > 0 else 0.0
        
        # 平移（简化处理，假设小角度）
        # 完整公式需要计算 V^{-1}，这里用近似
        upsilon = self.t  # 简化
        
        return np.concatenate([omega, upsilon, [sigma]])
    
    @staticmethod
    def exp(xi: np.ndarray) -> 'Sim3':
        """
        李代数指数映射: sim(3) → Sim(3)
        
        Args:
            xi: 7 维向量 [ω, υ, σ]
        """
        omega = xi[:3]
        upsilon = xi[3:6]
        sigma = xi[6]
        
        # 旋转
        R = Rotation.from_rotvec(omega).as_matrix()
        
        # 尺度
        s = np.exp(sigma)
        
        # 平移（简化）
        t = upsilon
        
        return Sim3(R, t, s)
    
    def copy(self) -> 'Sim3':
        return Sim3(self.R.copy(), self.t.copy(), self.s)


# =============================================================================
# Sim(3) 位姿图节点和边
# =============================================================================

@dataclass
class Sim3Node:
    """Sim(3) 位姿图节点"""
    id: int
    pose: Sim3
    fixed: bool = False
    
    
@dataclass  
class Sim3Edge:
    """Sim(3) 位姿图边"""
    source_id: int
    target_id: int
    measurement: Sim3  # 相对变换 T_source^{-1} @ T_target
    information: np.ndarray  # 7x7 信息矩阵
    is_loop_closure: bool = False


# =============================================================================
# Sim(3) 位姿图优化器
# =============================================================================

class Sim3PoseGraphOptimizer:
    """
    Sim(3) 位姿图优化器。
    
    使用 Gauss-Newton/Levenberg-Marquardt 优化 Sim(3) 位姿图，
    能够同时修正位姿和尺度。
    
    核心特性：
    1. 7-DOF 优化（比 SE(3) 多一个尺度自由度）
    2. 回环闭合时自动传播尺度修正
    3. 支持尺度约束边（来自外部传感器）
    """
    
    def __init__(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
        lambda_init: float = 1e-3,
        lambda_factor: float = 10.0
    ):
        """
        Args:
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            lambda_init: LM 初始阻尼系数
            lambda_factor: LM 阻尼系数调整因子
        """
        self.nodes: Dict[int, Sim3Node] = {}
        self.edges: List[Sim3Edge] = []
        
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor
        
    def add_node(self, node_id: int, pose: Sim3, fixed: bool = False):
        """添加节点"""
        self.nodes[node_id] = Sim3Node(node_id, pose, fixed)
        
    def add_edge(
        self,
        source_id: int,
        target_id: int,
        measurement: Sim3,
        information: np.ndarray = None,
        is_loop_closure: bool = False
    ):
        """
        添加边。
        
        Args:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            measurement: 相对变换 Z_ij = T_i^{-1} @ T_j
            information: 7x7 信息矩阵
            is_loop_closure: 是否是回环边
        """
        if information is None:
            # 默认信息矩阵
            information = np.eye(7)
            # 回环边的尺度信息更可靠
            if is_loop_closure:
                information[6, 6] = 100.0  # 尺度维度权重更高
        
        self.edges.append(Sim3Edge(
            source_id, target_id, measurement, information, is_loop_closure
        ))
    
    def compute_error(self, edge: Sim3Edge) -> np.ndarray:
        """
        计算边的误差向量（7 维）。
        
        误差定义：e_ij = log(Z_ij^{-1} @ T_i^{-1} @ T_j)
        """
        T_i = self.nodes[edge.source_id].pose
        T_j = self.nodes[edge.target_id].pose
        Z_ij = edge.measurement
        
        # 计算误差
        error_transform = Z_ij.inverse() @ (T_i.inverse() @ T_j)
        
        return error_transform.log()
    
    def compute_jacobians(self, edge: Sim3Edge) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算误差相对于节点位姿的雅可比矩阵。
        
        Returns:
            J_i: 对源节点的雅可比 (7x7)
            J_j: 对目标节点的雅可比 (7x7)
        """
        # 数值雅可比（更稳健）
        eps = 1e-6
        
        T_i = self.nodes[edge.source_id].pose
        T_j = self.nodes[edge.target_id].pose
        
        e0 = self.compute_error(edge)
        
        J_i = np.zeros((7, 7))
        J_j = np.zeros((7, 7))
        
        # 对 T_i 的雅可比
        for k in range(7):
            delta = np.zeros(7)
            delta[k] = eps
            
            # T_i + δ
            self.nodes[edge.source_id].pose = T_i @ Sim3.exp(delta)
            e_plus = self.compute_error(edge)
            
            J_i[:, k] = (e_plus - e0) / eps
            
            # 恢复
            self.nodes[edge.source_id].pose = T_i
        
        # 对 T_j 的雅可比
        for k in range(7):
            delta = np.zeros(7)
            delta[k] = eps
            
            # T_j + δ
            self.nodes[edge.target_id].pose = T_j @ Sim3.exp(delta)
            e_plus = self.compute_error(edge)
            
            J_j[:, k] = (e_plus - e0) / eps
            
            # 恢复
            self.nodes[edge.target_id].pose = T_j
        
        return J_i, J_j
    
    def optimize(self) -> Dict:
        """
        执行 Levenberg-Marquardt 优化。
        
        Returns:
            优化结果字典
        """
        if len(self.nodes) < 2 or len(self.edges) == 0:
            return {'success': True, 'iterations': 0, 'final_error': 0}
        
        # 构建节点索引映射（跳过固定节点）
        node_ids = sorted(self.nodes.keys())
        free_node_ids = [nid for nid in node_ids if not self.nodes[nid].fixed]
        
        if len(free_node_ids) == 0:
            return {'success': True, 'iterations': 0, 'final_error': 0}
        
        node_to_idx = {nid: i for i, nid in enumerate(free_node_ids)}
        n_free = len(free_node_ids)
        
        lambda_lm = self.lambda_init
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 构建 Hessian 和梯度
            H = np.zeros((7 * n_free, 7 * n_free))
            b = np.zeros(7 * n_free)
            total_error = 0
            
            for edge in self.edges:
                e = self.compute_error(edge)
                total_error += e.T @ edge.information @ e
                
                J_i, J_j = self.compute_jacobians(edge)
                
                # 源节点贡献
                if edge.source_id in node_to_idx:
                    idx_i = node_to_idx[edge.source_id] * 7
                    H[idx_i:idx_i+7, idx_i:idx_i+7] += J_i.T @ edge.information @ J_i
                    b[idx_i:idx_i+7] += J_i.T @ edge.information @ e
                
                # 目标节点贡献
                if edge.target_id in node_to_idx:
                    idx_j = node_to_idx[edge.target_id] * 7
                    H[idx_j:idx_j+7, idx_j:idx_j+7] += J_j.T @ edge.information @ J_j
                    b[idx_j:idx_j+7] += J_j.T @ edge.information @ e
                
                # 交叉项
                if edge.source_id in node_to_idx and edge.target_id in node_to_idx:
                    idx_i = node_to_idx[edge.source_id] * 7
                    idx_j = node_to_idx[edge.target_id] * 7
                    cross = J_i.T @ edge.information @ J_j
                    H[idx_i:idx_i+7, idx_j:idx_j+7] += cross
                    H[idx_j:idx_j+7, idx_i:idx_i+7] += cross.T
            
            # 检查收敛
            if abs(prev_error - total_error) < self.convergence_threshold * prev_error:
                print(f"  ✓ Converged at iteration {iteration}, error: {total_error:.6f}")
                break
            
            # LM 阻尼
            H_damped = H + lambda_lm * np.diag(np.diag(H) + 1e-6)
            
            # 求解增量
            try:
                delta = np.linalg.solve(H_damped, -b)
            except np.linalg.LinAlgError:
                lambda_lm *= self.lambda_factor
                continue
            
            # 更新位姿
            old_poses = {nid: self.nodes[nid].pose.copy() for nid in free_node_ids}
            
            for nid in free_node_ids:
                idx = node_to_idx[nid] * 7
                dx = delta[idx:idx+7]
                self.nodes[nid].pose = self.nodes[nid].pose @ Sim3.exp(dx)
            
            # 计算新误差
            new_error = sum(
                self.compute_error(e).T @ e.information @ self.compute_error(e)
                for e in self.edges
            )
            
            # LM 接受/拒绝
            if new_error < total_error:
                lambda_lm /= self.lambda_factor
                prev_error = total_error
            else:
                # 回滚
                for nid in free_node_ids:
                    self.nodes[nid].pose = old_poses[nid]
                lambda_lm *= self.lambda_factor
        
        return {
            'success': True,
            'iterations': iteration + 1,
            'final_error': total_error
        }
    
    def get_optimized_poses(self) -> Dict[int, Sim3]:
        """返回优化后的位姿"""
        return {nid: node.pose.copy() for nid, node in self.nodes.items()}
    
    def get_scale_corrections(self, reference_id: int = 0) -> Dict[int, float]:
        """
        获取相对于参考帧的尺度修正因子。
        
        这是关键：将尺度修正传播到点云！
        
        Args:
            reference_id: 参考帧 ID（尺度 = 1）
            
        Returns:
            {node_id: scale_correction}
        """
        if reference_id not in self.nodes:
            reference_id = min(self.nodes.keys())
        
        ref_scale = self.nodes[reference_id].pose.s
        
        return {
            nid: node.pose.s / ref_scale
            for nid, node in self.nodes.items()
        }


# =============================================================================
# 与 Open3D 兼容的包装器
# =============================================================================

class Sim3PoseGraphWrapper:
    """
    包装器：将 Sim(3) 优化集成到现有 SLAM 系统。
    
    使用方法：
    1. 替换 Open3D 的 global_optimization 调用
    2. 优化后调用 correct_pointclouds() 修正点云尺度
    """
    
    def __init__(self):
        self.optimizer = Sim3PoseGraphOptimizer()
        self.original_scales: Dict[int, float] = {}  # 记录原始尺度
        
    def add_node_from_se3(
        self, 
        node_id: int, 
        pose_se3: np.ndarray,
        scale: float = 1.0,
        fixed: bool = False
    ):
        """
        从 SE(3) 位姿添加节点。
        
        Args:
            node_id: 节点 ID
            pose_se3: 4x4 SE(3) 位姿矩阵
            scale: 该帧创建时的深度尺度因子
            fixed: 是否固定
        """
        # 记录原始尺度
        self.original_scales[node_id] = scale
        
        # 转换为 Sim(3)
        R = pose_se3[:3, :3]
        t = pose_se3[:3, 3]
        sim3_pose = Sim3(R, t, scale)
        
        self.optimizer.add_node(node_id, sim3_pose, fixed)
    
    def add_edge_from_se3(
        self,
        source_id: int,
        target_id: int,
        relative_se3: np.ndarray,
        relative_scale: float = 1.0,
        information_se3: np.ndarray = None,
        is_loop_closure: bool = False
    ):
        """
        从 SE(3) 相对变换添加边。
        
        Args:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            relative_se3: 4x4 相对变换
            relative_scale: 相对尺度
            information_se3: 6x6 SE(3) 信息矩阵
            is_loop_closure: 是否回环
        """
        R = relative_se3[:3, :3]
        t = relative_se3[:3, 3]
        measurement = Sim3(R, t, relative_scale)
        
        # 扩展信息矩阵到 7x7
        if information_se3 is not None:
            information = np.eye(7)
            information[:6, :6] = information_se3
            # 尺度维度信息
            if is_loop_closure:
                information[6, 6] = 100.0  # 回环的尺度约束更强
            else:
                information[6, 6] = 1.0
        else:
            information = np.eye(7)
            if is_loop_closure:
                information[6, 6] = 100.0
        
        self.optimizer.add_edge(
            source_id, target_id, measurement, information, is_loop_closure
        )
    
    def optimize(self) -> Dict:
        """执行优化"""
        return self.optimizer.optimize()
    
    def get_corrected_poses_se3(self) -> Dict[int, np.ndarray]:
        """获取修正后的 SE(3) 位姿（4x4）"""
        result = {}
        optimized = self.optimizer.get_optimized_poses()
        
        for node_id, sim3_pose in optimized.items():
            # 提取 SE(3) 部分（忽略尺度）
            T = np.eye(4)
            T[:3, :3] = sim3_pose.R
            T[:3, 3] = sim3_pose.t
            result[node_id] = T
            
        return result
    
    def get_scale_corrections(self) -> Dict[int, float]:
        """
        获取每帧的尺度修正因子。
        
        这是关键！用这个来修正点云。
        
        Returns:
            {node_id: correction_factor}
            点云应该乘以这个因子来修正尺度
        """
        optimized = self.optimizer.get_optimized_poses()
        
        # 参考帧的优化后尺度
        ref_id = min(optimized.keys())
        ref_scale_optimized = optimized[ref_id].s
        
        corrections = {}
        for node_id, sim3_pose in optimized.items():
            # 原始创建时的尺度
            original_scale = self.original_scales.get(node_id, 1.0)
            
            # 优化后的尺度
            optimized_scale = sim3_pose.s
            
            # 修正因子 = 优化后尺度 / 原始尺度
            # 但我们要归一化到参考帧
            correction = (optimized_scale / ref_scale_optimized) / original_scale
            corrections[node_id] = correction
            
        return corrections
    
    def correct_pointcloud(
        self,
        pointcloud,  # o3d.geometry.PointCloud
        node_id: int,
        new_pose: np.ndarray
    ):
        """
        修正单个点云的尺度和位姿。
        
        Args:
            pointcloud: 相机坐标系下的点云
            node_id: 对应的关键帧 ID
            new_pose: 优化后的位姿（4x4 SE(3)）
            
        Returns:
            修正后的世界坐标系点云
        """
        import open3d as o3d
        
        if pointcloud is None or len(pointcloud.points) == 0:
            return pointcloud
        
        # 获取尺度修正
        scale_corrections = self.get_scale_corrections()
        scale = scale_corrections.get(node_id, 1.0)
        
        # 复制点云
        pcd_corrected = copy.deepcopy(pointcloud)
        
        # 1. 尺度修正（在相机坐标系中）
        points = np.asarray(pcd_corrected.points)
        points *= scale
        pcd_corrected.points = o3d.utility.Vector3dVector(points)
        
        # 2. 变换到世界坐标系
        pcd_corrected.transform(new_pose)
        
        return pcd_corrected


# =============================================================================
# 集成到现有 SLAM 系统的辅助函数
# =============================================================================

def correct_map_after_loop_closure(
    keyframes: Dict,  # {id: Keyframe}
    pose_graph,       # Open3D PoseGraph
    scale_history: Dict[int, float]  # {kf_id: scale_factor_at_creation}
) -> Tuple[Dict, Dict]:
    """
    回环闭合后的完整修正流程。
    
    Args:
        keyframes: 关键帧字典
        pose_graph: Open3D 位姿图
        scale_history: 每个关键帧创建时的尺度因子
        
    Returns:
        (corrected_poses, scale_corrections)
    """
    import open3d as o3d
    
    print("🔧 正在执行 Sim(3) 尺度修正...")
    
    # 1. 构建 Sim(3) 位姿图
    wrapper = Sim3PoseGraphWrapper()
    
    # 添加节点
    for i, node in enumerate(pose_graph.nodes):
        scale = scale_history.get(i, 1.0)
        wrapper.add_node_from_se3(i, node.pose, scale, fixed=(i == 0))
    
    # 添加边
    for edge in pose_graph.edges:
        # 检测是否为回环边（通常 source 和 target 相差较大）
        is_loop = abs(edge.target_node_id - edge.source_node_id) > 5
        
        # 估计相对尺度
        s_src = scale_history.get(edge.source_node_id, 1.0)
        s_tgt = scale_history.get(edge.target_node_id, 1.0)
        relative_scale = s_tgt / s_src if s_src > 0 else 1.0
        
        wrapper.add_edge_from_se3(
            edge.source_node_id,
            edge.target_node_id,
            edge.transformation,
            relative_scale=relative_scale,
            information_se3=edge.information,
            is_loop_closure=is_loop
        )
    
    # 2. 优化
    result = wrapper.optimize()
    print(f"   优化完成: {result['iterations']} 次迭代, 误差 = {result['final_error']:.6f}")
    
    # 3. 获取修正
    corrected_poses = wrapper.get_corrected_poses_se3()
    scale_corrections = wrapper.get_scale_corrections()
    
    # 打印尺度修正摘要
    scales = list(scale_corrections.values())
    print(f"   尺度修正范围: [{min(scales):.3f}, {max(scales):.3f}]")
    
    return corrected_poses, scale_corrections


def apply_scale_corrections_to_pointclouds(
    keyframes: Dict,
    corrected_poses: Dict[int, np.ndarray],
    scale_corrections: Dict[int, float]
):
    """
    将尺度修正应用到所有关键帧的点云。
    
    这是解决您问题的关键函数！
    
    Args:
        keyframes: 关键帧字典 {id: Keyframe}
        corrected_poses: 修正后的位姿
        scale_corrections: 尺度修正因子
    """
    import open3d as o3d
    
    print("📐 正在应用尺度修正到点云...")
    
    for kf_id, kf in keyframes.items():
        if kf.pointcloud is None or len(kf.pointcloud.points) == 0:
            continue
            
        # 获取修正
        scale = scale_corrections.get(kf_id, 1.0)
        new_pose = corrected_poses.get(kf_id, kf.pose)
        
        # 修正点云尺度（在相机坐标系中）
        if abs(scale - 1.0) > 0.001:  # 只有显著差异才修正
            points = np.asarray(kf.pointcloud.points)
            points *= scale
            kf.pointcloud.points = o3d.utility.Vector3dVector(points)
            
        # 更新位姿
        kf.pose = new_pose
        
    print(f"   ✓ 已修正 {len(keyframes)} 个关键帧")


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Sim(3) 位姿图优化测试")
    print("="*60)
    
    # 模拟一个有尺度漂移的场景
    wrapper = Sim3PoseGraphWrapper()
    
    # 添加节点（模拟圆形轨迹，尺度逐渐漂移）
    n_nodes = 10
    radius = 2.0
    
    for i in range(n_nodes):
        angle = 2 * np.pi * i / n_nodes
        
        # 真实位置
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # 模拟尺度漂移
        scale_drift = 1.0 + 0.1 * i / n_nodes  # 从 1.0 漂移到 1.1
        
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        
        wrapper.add_node_from_se3(i, pose, scale=scale_drift, fixed=(i == 0))
    
    # 添加里程计边
    for i in range(n_nodes - 1):
        pose_i = wrapper.optimizer.nodes[i].pose.matrix()
        pose_j = wrapper.optimizer.nodes[i + 1].pose.matrix()
        relative = np.linalg.inv(pose_i) @ pose_j
        
        wrapper.add_edge_from_se3(i, i + 1, relative)
    
    # 添加回环边（连接首尾）
    pose_0 = wrapper.optimizer.nodes[0].pose.matrix()
    pose_last = wrapper.optimizer.nodes[n_nodes - 1].pose.matrix()
    relative_loop = np.linalg.inv(pose_last) @ pose_0
    
    wrapper.add_edge_from_se3(
        n_nodes - 1, 0, relative_loop, 
        relative_scale=1.0 / 1.1,  # 回环时尺度应该回到 1.0
        is_loop_closure=True
    )
    
    print(f"\n优化前尺度:")
    for i in range(n_nodes):
        print(f"  节点 {i}: scale = {wrapper.optimizer.nodes[i].pose.s:.4f}")
    
    # 优化
    result = wrapper.optimize()
    
    print(f"\n优化后尺度修正:")
    corrections = wrapper.get_scale_corrections()
    for i in range(n_nodes):
        print(f"  节点 {i}: correction = {corrections[i]:.4f}")
    
    print(f"\n✅ 测试完成")
