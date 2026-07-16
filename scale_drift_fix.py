#!/usr/bin/env python3
"""
================================================================================
尺度漂移修复补丁 - 直接集成到 pose_graph_enhanced.py
================================================================================

问题诊断：
---------
您的点云分裂成两块是因为：
1. 相机绕一圈，尺度从 s=1.0 漂移到 s≈1.3
2. 回环检测成功，Open3D 的 SE(3) 优化修正了位姿
3. 但点云的尺度没有修正 → 上半圈（s≈1.0）和下半圈（s≈1.3）分离

解决方案：
---------
1. 记录每个关键帧创建时的尺度因子
2. 回环闭合后，计算需要的尺度修正
3. 将尺度修正应用到点云

使用方法：
---------
1. 将此文件放到与 pose_graph_enhanced.py 同级目录
2. 在 pose_graph_enhanced.py 中导入并使用

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class ScaleTracker:
    """
    尺度追踪器：记录和管理深度尺度历史。
    
    核心思想：
    - 每帧创建时记录当时的 scale_factor
    - 回环闭合后计算尺度修正
    - 将修正应用到点云
    """
    
    def __init__(self):
        # {keyframe_id: scale_factor_at_creation}
        self.scale_history: Dict[int, float] = {}
        
        # 累积尺度（从起点到当前的尺度乘积）
        self.cumulative_scale: Dict[int, float] = {}
        
        # 参考帧 ID
        self.reference_id: int = 0
        
    def record_scale(self, keyframe_id: int, current_scale_factor: float):
        """
        记录关键帧创建时的尺度因子。
        
        应该在每次创建关键帧时调用。
        
        Args:
            keyframe_id: 关键帧 ID
            current_scale_factor: 当前的 scale_corrector 输出
        """
        self.scale_history[keyframe_id] = current_scale_factor
        
        # 计算累积尺度
        if keyframe_id == 0:
            self.cumulative_scale[keyframe_id] = 1.0
        else:
            # 相对于上一帧的尺度变化
            prev_id = keyframe_id - 1
            if prev_id in self.cumulative_scale:
                prev_cumulative = self.cumulative_scale[prev_id]
                prev_scale = self.scale_history.get(prev_id, 1.0)
                
                # 累积尺度 = 前一帧累积 × (当前尺度 / 前一帧尺度)
                if prev_scale > 0:
                    self.cumulative_scale[keyframe_id] = prev_cumulative * (current_scale_factor / prev_scale)
                else:
                    self.cumulative_scale[keyframe_id] = prev_cumulative
            else:
                self.cumulative_scale[keyframe_id] = current_scale_factor
    
    def compute_loop_closure_scale_correction(
        self,
        query_kf_id: int,
        match_kf_id: int
    ) -> float:
        """
        计算回环闭合时需要的尺度修正。
        
        当检测到 query_kf 与 match_kf 是同一位置时，
        它们的累积尺度应该相同，但实际上可能漂移了。
        
        Args:
            query_kf_id: 当前帧 ID（回环检测的查询帧）
            match_kf_id: 匹配帧 ID（历史帧）
            
        Returns:
            尺度修正因子
        """
        query_cumulative = self.cumulative_scale.get(query_kf_id, 1.0)
        match_cumulative = self.cumulative_scale.get(match_kf_id, 1.0)
        
        if query_cumulative > 0:
            # 理想情况下，query 和 match 的累积尺度应相同
            # 修正因子 = match / query
            correction = match_cumulative / query_cumulative
            return correction
        
        return 1.0
    
    def compute_global_scale_corrections(
        self,
        loop_closure_pairs: List[Tuple[int, int]]
    ) -> Dict[int, float]:
        """
        计算全局尺度修正（考虑多个回环约束）。
        
        使用最小二乘法求解最优尺度修正。
        
        Args:
            loop_closure_pairs: [(query_id, match_id), ...] 回环对列表
            
        Returns:
            {keyframe_id: scale_correction} 每帧需要乘的修正因子
        """
        if not loop_closure_pairs:
            return {kf_id: 1.0 for kf_id in self.scale_history}
        
        n = len(self.scale_history)
        if n == 0:
            return {}
        
        kf_ids = sorted(self.scale_history.keys())
        id_to_idx = {kf_id: i for i, kf_id in enumerate(kf_ids)}
        
        # 构建线性系统: A * x = b
        # 其中 x[i] = log(correction[i])
        # 约束: x[query] - x[match] = log(match_cumulative / query_cumulative)
        
        equations = []
        rhs = []
        
        # 固定第一帧
        eq = np.zeros(n)
        eq[0] = 1.0
        equations.append(eq)
        rhs.append(0.0)  # log(1) = 0
        
        # 回环约束
        for query_id, match_id in loop_closure_pairs:
            if query_id not in id_to_idx or match_id not in id_to_idx:
                continue
                
            eq = np.zeros(n)
            eq[id_to_idx[query_id]] = 1.0
            eq[id_to_idx[match_id]] = -1.0
            equations.append(eq)
            
            correction = self.compute_loop_closure_scale_correction(query_id, match_id)
            rhs.append(np.log(correction) if correction > 0 else 0.0)
        
        # 平滑约束（相邻帧尺度变化应该小）
        for i in range(n - 1):
            eq = np.zeros(n)
            eq[i] = 1.0
            eq[i + 1] = -1.0
            equations.append(eq * 0.1)  # 较小权重
            rhs.append(0.0)
        
        A = np.array(equations)
        b = np.array(rhs)
        
        # 最小二乘求解
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            corrections = np.exp(x)
        except:
            corrections = np.ones(n)
        
        return {kf_ids[i]: corrections[i] for i in range(n)}


def apply_scale_correction_to_pointcloud(
    pointcloud,  # o3d.geometry.PointCloud
    correction_factor: float
):
    """
    将尺度修正应用到单个点云。
    
    Args:
        pointcloud: Open3D 点云（相机坐标系）
        correction_factor: 尺度修正因子
        
    Returns:
        修正后的点云
    """
    import open3d as o3d
    
    if pointcloud is None or len(pointcloud.points) == 0:
        return pointcloud
    
    if abs(correction_factor - 1.0) < 0.001:
        return pointcloud
    
    pcd = copy.deepcopy(pointcloud)
    points = np.asarray(pcd.points)
    points *= correction_factor
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd


# =============================================================================
# 关键补丁：修改 pose_graph_enhanced.py
# =============================================================================

"""
在 PoseGraphOptimizer.__init__ 中添加：
    
    self.scale_tracker = ScaleTracker()
    self.detected_loop_pairs = []  # [(query_id, match_id), ...]


在 add_keyframe() 中，创建 pointcloud 后添加：

    # 记录尺度历史
    if hasattr(self, 'scale_tracker') and scale_factor is not None:
        self.scale_tracker.record_scale(keyframe_id, scale_factor)


在 _add_loop_closure_edge() 或回环验证成功后添加：

    # 记录回环对
    self.detected_loop_pairs.append((query_kf.id, match_kf.id))


修改 optimize() 函数，在 Open3D 优化后添加尺度修正：
"""


def enhanced_optimize(self) -> bool:
    """
    增强版位姿图优化：支持尺度修正。
    
    替换原有的 optimize() 方法。
    """
    import open3d as o3d
    
    if len(self.pose_graph.nodes) < 2:
        return True
    
    print(f"🔍 Global Optimization ({len(self.pose_graph.nodes)} nodes, "
          f"{len(self.pose_graph.edges)} edges)...")
    
    # Step 1: 标准 SE(3) 优化
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=self.icp_threshold,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.0,
        reference_node=0
    )
    
    try:
        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return False
    
    # Step 2: 更新位姿
    for i, node in enumerate(self.pose_graph.nodes):
        if i in self.keyframes:
            self.keyframes[i].pose = node.pose.copy()
        if i in self.nodes:
            self.nodes[i].pose = node.pose.copy()
    
    # Step 3: 尺度修正 (关键新增!)
    if hasattr(self, 'scale_tracker') and hasattr(self, 'detected_loop_pairs'):
        if len(self.detected_loop_pairs) > 0:
            print("📐 Applying scale corrections...")
            
            # 计算全局尺度修正
            corrections = self.scale_tracker.compute_global_scale_corrections(
                self.detected_loop_pairs
            )
            
            # 应用到点云
            corrected_count = 0
            for kf_id, kf in self.keyframes.items():
                if kf.pointcloud is None:
                    continue
                    
                correction = corrections.get(kf_id, 1.0)
                if abs(correction - 1.0) > 0.01:  # 只修正显著差异
                    kf.pointcloud = apply_scale_correction_to_pointcloud(
                        kf.pointcloud, correction
                    )
                    corrected_count += 1
            
            # 打印摘要
            scale_values = list(corrections.values())
            print(f"   Scale range: [{min(scale_values):.3f}, {max(scale_values):.3f}]")
            print(f"   Corrected {corrected_count} pointclouds")
    
    print("✅ Optimization complete (with scale correction)")
    return True


# =============================================================================
# 简化方案：基于回环的线性插值尺度修正
# =============================================================================

def simple_scale_correction(
    keyframes: Dict,
    loop_query_id: int,
    loop_match_id: int,
    scale_at_query: float,
    scale_at_match: float
):
    """
    简化的尺度修正：线性插值。
    
    当检测到回环时：
    - 帧 0 到 match_id: 尺度 = 1.0 (已经正确)
    - 帧 match_id 到 query_id: 尺度线性漂移了
    - 我们需要把这段的尺度修正回来
    
    Args:
        keyframes: 关键帧字典
        loop_query_id: 回环检测的查询帧 ID
        loop_match_id: 回环检测的匹配帧 ID
        scale_at_query: 查询帧创建时的尺度
        scale_at_match: 匹配帧创建时的尺度
    """
    import open3d as o3d
    
    if loop_query_id <= loop_match_id:
        return
    
    # 计算尺度漂移
    scale_drift = scale_at_query / scale_at_match if scale_at_match > 0 else 1.0
    
    if abs(scale_drift - 1.0) < 0.01:
        print("   Scale drift negligible, skipping correction")
        return
    
    print(f"📐 Detected scale drift: {scale_drift:.3f}")
    print(f"   Correcting frames {loop_match_id} to {loop_query_id}...")
    
    # 线性插值修正
    n_frames = loop_query_id - loop_match_id
    
    for kf_id in range(loop_match_id + 1, loop_query_id + 1):
        if kf_id not in keyframes:
            continue
            
        kf = keyframes[kf_id]
        if kf.pointcloud is None or len(kf.pointcloud.points) == 0:
            continue
        
        # 线性插值: 从 1.0 到 1/scale_drift
        progress = (kf_id - loop_match_id) / n_frames
        correction = 1.0 / (1.0 + (scale_drift - 1.0) * progress)
        
        # 应用修正
        if abs(correction - 1.0) > 0.005:
            points = np.asarray(kf.pointcloud.points)
            points *= correction
            kf.pointcloud.points = o3d.utility.Vector3dVector(points)
    
    print(f"   ✓ Scale correction applied")


# =============================================================================
# 完整集成示例
# =============================================================================

INTEGRATION_CODE = """
# ============================================================================
# 集成到 pose_graph_enhanced.py 的完整补丁
# ============================================================================

# 1. 在文件顶部添加导入
from scale_drift_fix import ScaleTracker, apply_scale_correction_to_pointcloud, simple_scale_correction

# 2. 在 PoseGraphOptimizer.__init__() 末尾添加
self.scale_tracker = ScaleTracker()
self.detected_loop_pairs = []
self.scale_history = {}  # {kf_id: scale_factor}

# 3. 修改 add_keyframe() 方法签名，添加 scale_factor 参数
def add_keyframe(self, rgb, depth, pose, timestamp, 
                 odometry_transform=None, scale_factor=1.0):  # <-- 新增参数
    ...
    
    # 在创建关键帧后添加：
    self.scale_history[keyframe_id] = scale_factor
    self.scale_tracker.record_scale(keyframe_id, scale_factor)
    
    ...

# 4. 在回环闭合验证成功后添加
# (在 _verify_loop_with_icp 或 _verify_loop_closure 返回 True 后)
if verified:
    self.detected_loop_pairs.append((query_kf.id, match_kf.id))
    
    # 简单方案：立即修正
    simple_scale_correction(
        self.keyframes,
        query_kf.id,
        match_kf.id,
        self.scale_history.get(query_kf.id, 1.0),
        self.scale_history.get(match_kf.id, 1.0)
    )

# 5. 在 run_slam.py 中，调用 add_keyframe 时传入尺度
scale_factor = depth_corrector.get_scale_factor()
kf_id, loop = optimizer.add_keyframe(
    rgb, depth, pose, timestamp, 
    odometry_transform=odom,
    scale_factor=scale_factor  # <-- 传入尺度
)
"""

print(INTEGRATION_CODE)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("尺度漂移修复测试")
    print("="*60)
    
    # 模拟尺度漂移
    tracker = ScaleTracker()
    
    # 模拟 20 帧，尺度从 1.0 漂移到 1.3
    for i in range(20):
        scale = 1.0 + 0.015 * i  # 每帧增加 1.5%
        tracker.record_scale(i, scale)
        print(f"帧 {i:2d}: scale = {scale:.3f}, cumulative = {tracker.cumulative_scale[i]:.3f}")
    
    # 模拟回环：帧 19 回到帧 0 的位置
    loop_correction = tracker.compute_loop_closure_scale_correction(19, 0)
    print(f"\n回环尺度修正: {loop_correction:.4f}")
    
    # 全局修正
    corrections = tracker.compute_global_scale_corrections([(19, 0)])
    
    print("\n全局尺度修正:")
    for kf_id in sorted(corrections.keys()):
        print(f"  帧 {kf_id:2d}: correction = {corrections[kf_id]:.4f}")
    
    print("\n✅ 测试完成")
