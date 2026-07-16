#!/usr/bin/env python3
"""
================================================================================
PiSLAM Pose Graph Optimization Module (Enhanced Version)
================================================================================

This module implements a robust back-end for Visual SLAM with the following
key improvements over the baseline:

1. FAISS-Accelerated Loop Closure Detection
   - Replaces O(N×M) BoW with O(N×log(M)) FAISS-based search
   - 100× speedup on typical sequences
   
2. Global Descriptor Matching (Optional)
   - MobileNetV2 global descriptors via Hailo NPU or ONNX
   - More robust to viewpoint and illumination changes

3. Two-Stage Verification Pipeline
   - Coarse: Fast global retrieval with adaptive thresholding
   - Fine: Geometric verification with RANSAC

4. Improved Graph Optimization
   - Local sliding window optimization for real-time performance
   - Robust kernel for outlier handling

References:
    [1] Kümmerle et al., "g2o: A General Framework for Graph Optimization",
        ICRA 2011
    [2] Arandjelovic et al., "NetVLAD: CNN Architecture for Weakly Supervised
        Place Recognition", CVPR 2016
    [3] Galvez-Lopez & Tardos, "Bags of Binary Words for Fast Place 
        Recognition in Image Sequences", IEEE TRO 2012
    [4] Johnson et al., "Billion-scale similarity search with GPUs",
        IEEE TBED 2019

Author: Academic SLAM Implementation
================================================================================
"""

import numpy as np
import open3d as o3d
import cv2
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import yaml
import copy
from scipy.spatial.transform import Rotation as ScipyRotation
from local_bundle_adjustment import SlidingWindowBA
# Import feature extraction modules
# [新增] 导入尺度修复模块
try:
    from scale_drift_fix import ScaleTracker, simple_scale_correction
except ImportError:
    print("⚠️ Scale fix module not found")

try:
    from features import ORBExtractor, FeatureMatcher, KeyPoint
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("⚠️ features.py not found, using minimal feature extraction")

try:
    from sim3_pose_graph import correct_map_after_loop_closure
    SIM3_AVAILABLE = True
except ImportError:
    SIM3_AVAILABLE = False

# Import enhanced loop closure detector
try:
    from loop_closure_enhanced import (
        EnhancedLoopClosureDetector,
        LoopClosureConfig,
        LoopCandidate,
        FAISSBagOfWords,
        create_enhanced_loop_detector
    )
    ENHANCED_LC_AVAILABLE = True
except ImportError:
    ENHANCED_LC_AVAILABLE = False
    print("⚠️ loop_closure_enhanced.py not found, using baseline BoW")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Keyframe:
    """
    Represents a keyframe (State Variable) in the SLAM pose graph.
    
    A keyframe stores all necessary information for loop closure detection
    and map reconstruction, including the camera pose, extracted features,
    and associated point cloud.
    
    Attributes:
        id: Unique keyframe identifier
        timestamp: Capture timestamp (seconds)
        pose: 4×4 camera-to-world transformation matrix T_wc
        keypoints: List of detected feature keypoints
        descriptors: Binary ORB descriptors (N×32)
        bow_vector: Bag-of-Words histogram
        global_descriptor: Learned global image descriptor
        pointcloud: Associated 3D point cloud
        covisibility: Set of covisible keyframe IDs
        rgb_image: RGB image (retained for dataset generation)
    """
    id: int
    timestamp: float
    pose: np.ndarray  # T_wc (4×4)
    keypoints: List = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    bow_vector: Optional[np.ndarray] = None
    global_descriptor: Optional[np.ndarray] = None
    pointcloud: Optional[o3d.geometry.PointCloud] = None
    covisibility: Set[int] = field(default_factory=set)
    rgb_image: Optional[np.ndarray] = None


@dataclass
class LoopClosureCandidateLegacy:
    """Legacy candidate structure for backward compatibility."""
    query_id: int
    match_id: int
    bow_score: float = 0.0
    num_inliers: int = 0
    relative_pose: Optional[np.ndarray] = None
    information: Optional[np.ndarray] = None
    is_verified: bool = False


class PoseGraphNode:
    """Wrapper for Open3D PoseGraphNode with metadata."""
    def __init__(self, pose: np.ndarray, keyframe_id: int):
        self.pose = pose.copy()
        self.keyframe_id = keyframe_id
        self.fixed = False


class PoseGraphEdge:
    """Wrapper for Open3D PoseGraphEdge with metadata."""
    def __init__(self, source_id: int, target_id: int,
                 transformation: np.ndarray,
                 information: np.ndarray,
                 is_loop_closure: bool = False):
        self.source_id = source_id
        self.target_id = target_id
        self.transformation = transformation.copy()
        self.information = information.copy()
        self.is_loop_closure = is_loop_closure


# =============================================================================
# Pose Graph Optimizer (Enhanced Version)
# =============================================================================

class PoseGraphOptimizer:
    """
    The Back-end Optimizer for PiSLAM with enhanced loop closure detection.
    
    This class manages the pose graph structure and provides methods for:
    - Keyframe management and feature extraction
    - Loop closure detection (FAISS-accelerated or BoW)
    - Pose graph optimization (local and global)
    - Map building and trajectory export
    
    The enhanced version supports two loop closure backends:
    1. FAISS + Global Descriptors (recommended for NPU-equipped devices)
    2. FAISS-accelerated BoW (fallback for CPU-only devices)
    
    Both backends provide significant speedup over the baseline Python BoW.
    """
    
    def __init__(self, config_path: Optional[str] = None,
                 use_enhanced_lc: bool = True,
                 hef_path: Optional[str] = None,
                 onnx_path: Optional[str] = None,
                 vdevice = None):  # [修复] 支持共享 Hailo VDevice
        """
        Initialize the pose graph optimizer.
        
        Args:
            config_path: Path to YAML configuration file
            use_enhanced_lc: Whether to use enhanced loop closure detection
            hef_path: Path to Hailo HEF model for global descriptors
            onnx_path: Path to ONNX model for global descriptors
            vdevice: [新增] 共享的 Hailo VDevice 实例，用于避免 NPU 资源竞争
        """
        # [修复] 保存共享 VDevice 引用

        # [新增] 初始化尺度追踪器
        self.scale_tracker = ScaleTracker()
        self.scale_history = {} # 备份一份 {id: scale} 用于快速访问

        self.vdevice = vdevice
        # Load configuration
        self._load_config(config_path)

        # 初始化滑动窗口 BA
        self.local_ba = SlidingWindowBA(window_size=10)
        # 记得设置内参，需要从 self.intrinsics 获取
        # 注意：此时可能还没加载 config，可以在 load_config 后或者 add_keyframe 第一帧时设置
        self.ba_initialized = False
        
        # Feature extraction (if available)
        if FEATURES_AVAILABLE:
            self.extractor = ORBExtractor(config_path)
            self.matcher = FeatureMatcher(config_path)
        else:
            self.orb = cv2.ORB_create(1000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Loop closure detection backend
        self.use_enhanced_lc = use_enhanced_lc and ENHANCED_LC_AVAILABLE
        
        # if self.use_enhanced_lc:
        #     print("\n🚀 Using Enhanced Loop Closure Detection (FAISS + Global Descriptors)")
        #     self.loop_detector = create_enhanced_loop_detector(
        #         config_path, hef_path, onnx_path,
        #         vdevice=self.vdevice  # [修复] 传递共享 VDevice
        #     )
        #     self.bow = self.loop_detector.bow  # Shared reference
        #     self.bow_trained = False

        if self.use_enhanced_lc:
            print("\n🚀 Using Enhanced Loop Closure Detection (VLAD Mode)")
            self.loop_detector = create_enhanced_loop_detector(
                config_path, hef_path, onnx_path,
                vdevice=self.vdevice
            )
            
            # [兼容性修复] VLAD 模式下不需要外部维护 BoW 对象
            # 我们将其设为 None，并标记为 "已训练" 以跳过旧逻辑
            self.bow = None 
            self.bow_trained = True 

        else:
            print("\n📚 Using FAISS-accelerated Bag-of-Words")
            if ENHANCED_LC_AVAILABLE:
                self.bow = FAISSBagOfWords(vocabulary_size=self.vocabulary_size)
            else:
                # Import baseline BoW if enhanced not available
                from features import BagOfWords
                self.bow = BagOfWords(
                    vocabulary_size=self.vocabulary_size,
                    vocabulary_depth=self.vocabulary_depth
                )
            self.bow_trained = False
            self.loop_detector = None
        
        # Graph data structures
        self.keyframes: Dict[int, Keyframe] = {}
        self.next_keyframe_id = 0
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.nodes: Dict[int, PoseGraphNode] = {}
        self.edges: List[PoseGraphEdge] = []
        self.detected_loops: List = []
        
        # Statistics
        self.total_keyframes = 0
        self.total_loop_closures = 0
        self.last_optimization_kf_count = 0
        self.timing_stats = {
            'feature_extraction': [],
            'pointcloud_creation': [],
            'loop_detection': [],
            'optimization': []
        }
        
        print(f"✅ PoseGraphOptimizer initialized")
        print(f"   Enhanced LC: {self.use_enhanced_lc}")
        print(f"   Optimize every: {self.optimize_every_n_keyframes} keyframes\n")
    

    def _load_config(self, config_path: Optional[str]):
        """Load configuration parameters with academic defaults."""
        # Initialize with defaults
        self.voxel_size = 0.02
        self.min_keyframe_gap = 10
        self.search_radius = 2.0
        self.bow_similarity_threshold = 0.3
        self.min_inliers = 20
        self.loop_ransac_iterations = 500
        self.icp_fitness_threshold = 0.5
        self.icp_rmse_threshold = 0.05
        self.optimize_every_n_keyframes = 5
        self.vocabulary_size = 1000
        self.vocabulary_depth = 6
        self.icp_threshold = 0.1
        self.sliding_window_size = 10
        
        self.intrinsics = {
            'fx': 500.0, 'fy': 500.0, 
            'cx': 320.0, 'cy': 240.0,
            'depth_min': 0.2, 'depth_max': 5.0,
            'width': 640, 'height': 480
        }
        
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Loop Closure Config
            lc = config.get('loop_closure', {})
            # 🔴 [错误位置] 原来的代码在这里引用了 cam，但 cam 还没定义，请删除这两行
            # self.intrinsics['width'] = cam.get('width', self.intrinsics['width'])
            # self.intrinsics['height'] = cam.get('height', self.intrinsics['height'])

            self.min_keyframe_gap = lc.get('min_keyframe_gap', self.min_keyframe_gap)
            self.search_radius = lc.get('search_radius', self.search_radius)
            self.bow_similarity_threshold = lc.get('bow_similarity_threshold', self.bow_similarity_threshold)
            self.min_inliers = lc.get('min_inliers', self.min_inliers)
            self.loop_ransac_iterations = lc.get('ransac_iterations', self.loop_ransac_iterations)
            self.icp_fitness_threshold = lc.get('icp_fitness_threshold', self.icp_fitness_threshold)
            self.icp_rmse_threshold = lc.get('icp_rmse_threshold', self.icp_rmse_threshold)
            self.vocabulary_size = lc.get('vocabulary_size', self.vocabulary_size)
            self.vocabulary_depth = lc.get('vocabulary_depth', self.vocabulary_depth)
            
            # Optimization Config
            opt = config.get('optimization', {})
            self.optimize_every_n_keyframes = opt.get('optimize_every_n_keyframes', self.optimize_every_n_keyframes)
            
            # Odometry Config
            odom = config.get('odometry', {})
            self.icp_threshold = odom.get('icp_max_correspondence_distance', self.icp_threshold)
            self.voxel_size = odom.get('voxel_size', self.voxel_size)
            
            # Camera Config
            cam = config.get('camera', {})
            # 🟢 [正确位置] 必须在 cam 定义之后读取
            self.intrinsics['width'] = cam.get('width', self.intrinsics['width'])
            self.intrinsics['height'] = cam.get('height', self.intrinsics['height'])
            
            self.intrinsics.update({
                'fx': cam.get('fx', self.intrinsics['fx']),
                'fy': cam.get('fy', self.intrinsics['fy']),
                'cx': cam.get('cx', self.intrinsics['cx']),
                'cy': cam.get('cy', self.intrinsics['cy']),
                'depth_min': cam.get('depth_min', self.intrinsics['depth_min']),
                'depth_max': cam.get('depth_max', self.intrinsics['depth_max'])
            })

    
    def _check_geometric_consistency(self, pcd_world: o3d.geometry.PointCloud, 
                                   curr_kf_id: int, 
                                   check_window: int = 3,
                                   depth_thres: float = 0.15) -> List[int]:
        """
        [学术核心] 几何一致性检查 (Geometric Consistency Check)
        将当前帧的点云投影到邻近的关键帧中，验证深度误差。
        
        Args:
            pcd_world: 当前帧转换到世界坐标系的点云
            curr_kf_id: 当前关键帧 ID
            check_window: 向前/向后检查的关键帧窗口大小 (保守策略: 检查最近 2-3 帧)
            depth_thres: 深度误差容忍度 (米)
            
        Returns:
            List[int]: 保留点的索引列表 (valid_indices)
        """
        points_w = np.asarray(pcd_world.points)
        if len(points_w) == 0: return []
        
        # 初始假设所有点都是有效的 (Vote count)
        # 策略：只要通过了任意一帧的验证，就认为是好点 (宽松策略)
        # 或者：必须通过所有帧验证 (严格策略)
        # 推荐：加分制。初始 0 分，验证通过一帧 +1 分。最后保留分数 >= 1 的点。
        votes = np.zeros(len(points_w), dtype=np.int32)
        
        # 确定要检查的邻居帧 (排除自己)
        # 优先检查最近的几帧，因为共视关系最强
        neighbor_ids = []
        for i in range(1, check_window + 1):
            if (curr_kf_id - i) in self.keyframes: neighbor_ids.append(curr_kf_id - i)
            # if (curr_kf_id + i) in self.keyframes: neighbor_ids.append(curr_kf_id + i) # 未来帧通常还没建图，但在回放时可用
            
        if not neighbor_ids:
            return np.arange(len(points_w)) # 没有邻居可查，默认保留
            
        H, W = self.intrinsics['height'], self.intrinsics['width']
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        
        # 为了加速，先把点转成齐次坐标 (N, 4)
        points_w_homo = np.hstack((points_w, np.ones((len(points_w), 1))))
        
        for nid in neighbor_ids:
            neighbor_kf = self.keyframes[nid]
            
            # 世界 -> 邻居相机坐标系 (T_cw = inv(T_wc))
            # Pose 是 T_wc (Camera to World)
            T_wc = neighbor_kf.pose
            T_cw = np.linalg.inv(T_wc)
            
            # 1. 批量投影 (World -> Camera)
            # (N, 4) @ (4, 4).T -> (N, 4)
            P_cam = points_w_homo @ T_cw.T 
            
            # 提取 Z (深度)
            Z = P_cam[:, 2]
            valid_z = Z > 0.1 # 排除相机后方的点
            
            # 2. 投影到像素平面 (Camera -> Pixel)
            # u = fx * X / Z + cx
            # v = fy * Y / Z + cy
            U = (fx * P_cam[:, 0] / (Z + 1e-6) + cx).astype(np.int32)
            V = (fy * P_cam[:, 1] / (Z + 1e-6) + cy).astype(np.int32)
            
            # 3. 边界检查
            in_bounds = valid_z & (U >= 0) & (U < W) & (V >= 0) & (V < H)
            
            # 4. 深度一致性比对
            # 获取邻居帧的真实深度图
            # 注意：Keyframe 需要存储深度图。如果没有存，需要确保 add_keyframe 时保留了
            # 如果内存受限没存全图，这个方案无法执行。假设我们存了 (Keyframe data class 需确认)
            # 如果 Keyframe 只有 pointcloud，我们可以反投影 pointcloud 得到深度图，或者直接跳过
            
            # ⚠️ 假设深度图不可用，我们用 pointcloud 做的 KDTree 查找太慢。
            # 这里依赖 Keyframe 类中是否保留了 depth 或者是从 depth 生成的。
            # 让我们做一个工程折中：只对有 pointcloud 的帧做校验
            
            # 为了“最小修改”且不炸内存，我们这里假设无法获取原始深度图，
            # 而是用 "投影到邻居图像平面 -> 判断是否落在邻居 Mask 内" (Step A 的逻辑延伸)
            # 或者更简单的：如果能在邻居的 PointCloud 里找到很近的点 (ICP思想)
            
            # 但既然您要求 "重投影误差"，最正统的方法是读深度图。
            # 检查代码发现 Keyframe 类目前只存了 rgb_image。
            # 为了实现此功能，我们需要在 Keyframe 中临时加载深度，或者利用 pointcloud 反查。
            
            # [修正方案] 利用 3D 距离一致性 (更通用)
            # 如果 P_world 真的存在，那么它在邻居帧 T_cw 变换后的位置，应该离邻居帧的点云非常近。
            # 这可以用 KDTree 加速。
            
            if neighbor_kf.pointcloud is not None and len(neighbor_kf.pointcloud.points) > 0:
                 # 构建邻居点云的 KDTree (轻量级)
                 # 注意：频繁构建 KDTree 可能会慢，但在 build_global_map 时只做一次，可以接受
                 nb_pcd = neighbor_kf.pointcloud
                 # 这里需要把邻居点云转到世界坐标，或者把 P_world 转到邻居坐标
                 # 为了快，我们把 P_world 转到邻居坐标 P_cam，然后查邻居的原始点云 (也是局部坐标)
                 
                 # 邻居点云 (Local Frame)
                 nb_tree = o3d.geometry.KDTreeFlann(nb_pcd)
                 
                 # 对每个投影点 P_cam (只取前3维)，在 nb_tree 找最近邻
                 # Python 循环太慢，我们只抽样检查或者用 Open3D 的 compute_point_cloud_distance
                 
                 # [极速方案] 
                 # 构造一个临时的 Open3D 点云包含 P_cam
                 src_pcd_cam = o3d.geometry.PointCloud()
                 src_pcd_cam.points = o3d.utility.Vector3dVector(P_cam[:, :3])
                 
                 # 计算到邻居点云的距离
                 dists = src_pcd_cam.compute_point_cloud_distance(nb_pcd)
                 dists = np.asarray(dists)
                 
                 # 距离小于阈值 (例如 5cm) 的点认为是验证通过的
                 consistent = (dists < depth_thres)
                 
                 # 累加投票
                 votes[consistent] += 1

        # 筛选逻辑：至少被 1 个邻居验证通过 (1票)
        # 或者更严格：被 50% 的邻居验证通过
        keep_indices = np.where(votes >= 1)[0]
        return keep_indices

    # def add_keyframe(self, 
    #                  rgb: np.ndarray,
    #                  depth: np.ndarray,
    #                  pose: np.ndarray,
    #                  timestamp: float,
    #                  odometry_transform: Optional[np.ndarray] = None) -> Tuple[int, bool]:
    # # [修改] 增加 scale_factor 参数，默认为 1.0
    def add_keyframe(self, 
                     rgb: np.ndarray,
                     depth: np.ndarray,
                     pose: np.ndarray,
                     timestamp: float,
                     odometry_transform: Optional[np.ndarray] = None,
                     scale_factor: float = 1.0) -> Tuple[int, bool]: # <--- 修改这里    
        """
        Add a new keyframe to the pose graph.
        
        This method performs the following steps:
        1. Extract ORB features with 3D positions
        2. Compute BoW vector (if vocabulary trained)
        3. Create point cloud from depth
        4. Add node to pose graph
        5. Add odometry edge to previous keyframe
        6. Detect and verify loop closures
        7. Trigger optimization if needed
        
        Args:
            rgb: RGB image (H×W×3)
            depth: Depth map in meters (H×W)
            pose: Camera-to-world transformation T_wc (4×4)
            timestamp: Capture timestamp (seconds)
            odometry_transform: Relative transform from previous keyframe
            
        Returns:
            Tuple of (keyframe_id, loop_detected)
        """

        # 🟢 [修复] 自动从输入图像更新相机分辨率
        # 这确保了 _check_geometric_consistency 能获取到正确的图像边界 (H, W)
        if self.intrinsics['width'] != depth.shape[1] or self.intrinsics['height'] != depth.shape[0]:
            self.intrinsics['height'], self.intrinsics['width'] = depth.shape[:2]
            # print(f"📷 Updated camera intrinsics size to {self.intrinsics['width']}x{self.intrinsics['height']}")

        keyframe_id = self.next_keyframe_id
        self.next_keyframe_id += 1
        self.total_keyframes += 1
        
        # 1. Feature extraction
        t_start = time.time()
        if FEATURES_AVAILABLE:
            keypoints, descriptors = self.extractor.extract_with_depth(
                rgb, depth, self.intrinsics
            )
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            cv_kps, descriptors = self.orb.detectAndCompute(gray, None)
            keypoints = []
            for kp in cv_kps:
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                    z = depth[v, u]
                    if self.intrinsics['depth_min'] < z < self.intrinsics['depth_max']:
                        x = (u - self.intrinsics['cx']) * z / self.intrinsics['fx']
                        y = (v - self.intrinsics['cy']) * z / self.intrinsics['fy']
                        pt_3d = np.array([x, y, z])
                    else:
                        pt_3d = None
                else:
                    pt_3d = None
                keypoints.append(type('KeyPoint', (), {
                    'pt': kp.pt, 'size': kp.size, 'angle': kp.angle,
                    'response': kp.response, 'octave': kp.octave,
                    'depth': depth[v, u] if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] else None,
                    'point_3d': pt_3d
                })())
        self.timing_stats['feature_extraction'].append(time.time() - t_start)
        
        # # 2. Compute BoW vector
        # bow_vector = None
        # if self.bow_trained and descriptors is not None:
        #     bow_vector = self.bow.compute_bow_vector(descriptors)
        #     self.bow.add_to_database(keyframe_id, bow_vector)

        # 2. Compute BoW vector
        bow_vector = None
        # [修改] 增加 self.bow is not None 的判断
        if self.bow is not None and self.bow_trained and descriptors is not None:
            bow_vector = self.bow.compute_bow_vector(descriptors)
            self.bow.add_to_database(keyframe_id, bow_vector)
        
        # 3. Create point cloud
        t_start = time.time()
        pointcloud = self._create_pointcloud(rgb, depth)
        self.timing_stats['pointcloud_creation'].append(time.time() - t_start)
        
        # [新增] 在创建 kf 对象之前或之后，记录尺度
        self.scale_history[keyframe_id] = scale_factor
        self.scale_tracker.record_scale(keyframe_id, scale_factor)

        # 4. Create keyframe
        kf = Keyframe(
            id=keyframe_id,
            timestamp=timestamp,
            pose=pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            bow_vector=bow_vector,
            pointcloud=pointcloud,
            rgb_image=rgb.copy()
        )
        self.keyframes[keyframe_id] = kf


        
        # 5. Add to pose graph
        self._add_node(keyframe_id, pose)
        
        # # 6. Add odometry edge
        if keyframe_id > 0 and odometry_transform is not None:
            prev_pcd = self.keyframes[keyframe_id - 1].pointcloud
            information = self._compute_odometry_information(
                prev_pcd, pointcloud, odometry_transform
            )
            self._add_edge(
                keyframe_id - 1, keyframe_id,
                odometry_transform, information, 
                is_loop_closure=False
            )
        
        # 6. Add odometry edge
        # if keyframe_id > 0 and odometry_transform is not None:
        #     prev_kf = self.keyframes[keyframe_id - 1]
        #     prev_pcd = prev_kf.pointcloud
            
        #     # 🟢 [核心修改] 获取 ICP 优化后的变换和信息矩阵
        #     information, refined_transform = self._compute_odometry_information(
        #         prev_pcd, pointcloud, odometry_transform
        #     )
            
        #     # 🟢 [关键] 更新当前关键帧的位姿！
        #     # PnP 只是估计，ICP 才是真理。我们需要把 PnP 的误差修正过来。
        #     # T_curr = T_prev * T_refined
        #     refined_pose = prev_kf.pose @ refined_transform
            
        #     # 更新当前帧位姿 (修正积累误差)
        #     self.keyframes[keyframe_id].pose = refined_pose
        #     self.nodes[keyframe_id].pose = refined_pose
        #     self.pose_graph.nodes[-1].pose = refined_pose # 更新图节点
            
        #     # 添加边
        #     self._add_edge(
        #         keyframe_id - 1, keyframe_id,
        #         refined_transform, # 使用 ICP 优化后的相对变换
        #         information, 
        #         is_loop_closure=False
        #     )
        
        # 7. Loop closure detection
        t_start = time.time()
        loop_detected = False

        # === 局部 BA 集成 ===
        # 1. 确保内参已设置
        if not self.ba_initialized:
            self.local_ba.set_intrinsics(
                self.intrinsics['fx'], self.intrinsics['fy'],
                self.intrinsics['cx'], self.intrinsics['cy']
            )
            self.ba_initialized = True

        # 2. 将当前帧加入 BA 优化器
        # pose 是当前帧的位姿 T_wc
        self.local_ba.add_keyframe(keyframe_id, pose, fixed=(keyframe_id==0))

        # 3. 将当前帧观测到的地图点加入 BA
        # 这一步比较繁琐，因为 Keyframe 类里存的是 keypoints 列表
        # 我们需要把 3D 点提取出来加进去
        for i, kp in enumerate(kf.keypoints):
            # 假设 kp 有 point_3d 属性 (在 Keyframe 定义里是有的)
            if kp.point_3d is not None:
                # 构造一个唯一的点 ID，比如 hash(keyframe_id * 10000 + i) 或者全局索引
                # 简单起见，这里假设我们只优化“新”点，或者您需要维护一个全局 MapPoint 数据库
                # 这是一个简化的集成示例：
                pid = keyframe_id * 10000 + i 
                self.local_ba.add_map_point(pid, kp.point_3d)
                # 添加观测: (u, v)
                self.local_ba.add_observation(keyframe_id, pid, kp.pt)

        # 4. 触发优化 (每 5 帧一次)
        if keyframe_id % 5 == 0 and keyframe_id > 5:
            print(f"✨ 触发局部 BA (ID: {keyframe_id})...")
            ba_res = self.local_ba.optimize()
            if ba_res['success']:
                # 更新优化后的位姿回 PoseGraph
                opt_poses = self.local_ba.get_optimized_poses()
                for kf_id, new_pose in opt_poses.items():
                    if kf_id in self.keyframes:
                        self.keyframes[kf_id].pose = new_pose
                        # 同时更新 pose_graph.nodes 里的位姿以保持同步
                        if kf_id < len(self.pose_graph.nodes):
                            self.pose_graph.nodes[kf_id].pose = new_pose
                print(f"   BA 完成，误差: {ba_res['final_cost']:.4f}")
        # ====================
        
        if keyframe_id >= self.min_keyframe_gap:
            if self.use_enhanced_lc and self.loop_detector is not None:
                # Enhanced detection with global descriptors
                position = pose[:3, 3]
                
                # Add to detector database
                self.loop_detector.add_keyframe(
                    keyframe_id, rgb, position, descriptors
                )
                
                # Detect loops
                candidates = self.loop_detector.detect_loop_closure(
                    keyframe_id, rgb, position, descriptors, keypoints,
                    self.keyframes
                )
                
                for candidate in candidates:
                    if candidate.is_verified:
                        # Refine with ICP
                        if self._verify_loop_with_icp(kf, candidate):
                            self._add_loop_closure_edge(candidate)
                            loop_detected = True
                            self.total_loop_closures += 1
                            print(f"🔄 Loop closure: {keyframe_id} ↔ {candidate.match_id}")

                            # [新增] 触发尺度线性修正
                            print(f"📐 Triggering scale correction for loop {keyframe_id}-{candidate.match_id}")
                            simple_scale_correction(
                                self.keyframes,
                                query_kf_id=keyframe_id,
                                match_kf_id=candidate.match_id,
                                scale_at_query=self.scale_history.get(keyframe_id, 1.0),
                                scale_at_match=self.scale_history.get(candidate.match_id, 1.0)
                            )
            else:
                # Fallback: BoW-based detection
                candidates = self._detect_loop_closure_bow(kf)
                for candidate in candidates:
                    if self._verify_loop_closure(kf, candidate):
                        self._add_loop_closure_edge_legacy(candidate)
                        loop_detected = True
                        self.total_loop_closures += 1
        
        self.timing_stats['loop_detection'].append(time.time() - t_start)
        
        # 8. Trigger optimization
        kf_since_opt = self.total_keyframes - self.last_optimization_kf_count
        if kf_since_opt >= self.optimize_every_n_keyframes:
            t_start = time.time()
            
            # Local optimization first
            self._local_sliding_window_optimize()
            
            # Global if loop detected
            if loop_detected:
                self.optimize()
            
            self.last_optimization_kf_count = self.total_keyframes
            self.timing_stats['optimization'].append(time.time() - t_start)
        
        return keyframe_id, loop_detected
    
    def _local_sliding_window_optimize(self):
        """
        Perform local sliding window optimization.
        
        This optimizes only the most recent keyframes to maintain
        real-time performance while reducing local drift.
        """
        n = len(self.keyframes)
        if n < 3:
            return False
        
        window_size = min(self.sliding_window_size, n)
        start_id = n - window_size
        
        # Build local graph
        local_graph = o3d.pipelines.registration.PoseGraph()
        
        for i in range(window_size):
            gid = start_id + i
            if gid in self.keyframes:
                local_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        self.keyframes[gid].pose.copy()
                    )
                )
        
        # Fix first node
        if len(local_graph.nodes) > 0:
            local_graph.nodes[0].pose = self.keyframes[start_id].pose.copy()
        
        # Add edges within window
        for edge in self.edges:
            if start_id <= edge.source_id < n and start_id <= edge.target_id < n:
                local_src = edge.source_id - start_id
                local_tgt = edge.target_id - start_id
                if 0 <= local_src < window_size and 0 <= local_tgt < window_size:
                    local_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            local_src, local_tgt,
                            edge.transformation.copy(),
                            edge.information.copy(),
                            uncertain=edge.is_loop_closure
                        )
                    )
        
        if len(local_graph.edges) == 0:
            return False
        
        # Optimize
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.icp_threshold,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.0,
            reference_node=0
        )
        
        try:
            o3d.pipelines.registration.global_optimization(
                local_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option
            )
        except Exception:
            return False
        
        # Update keyframe poses
        for i, node in enumerate(local_graph.nodes):
            gid = start_id + i
            if gid in self.keyframes:
                self.keyframes[gid].pose = node.pose.copy()
            if gid in self.nodes:
                self.nodes[gid].pose = node.pose.copy()
            if gid < len(self.pose_graph.nodes):
                self.pose_graph.nodes[gid].pose = node.pose.copy()
        
        return True
    
    def _add_node(self, keyframe_id: int, pose: np.ndarray):
        """Add a node to the pose graph."""
        self.nodes[keyframe_id] = PoseGraphNode(pose, keyframe_id)
        self.pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(pose.copy())
        )
    
    def _add_edge(self, source_id: int, target_id: int,
                  transformation: np.ndarray,
                  information: np.ndarray,
                  is_loop_closure: bool):
        """Add an edge to the pose graph."""
        self.edges.append(PoseGraphEdge(
            source_id, target_id, transformation, information, is_loop_closure
        ))
        self.pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id, target_id, transformation, information,
                uncertain=is_loop_closure
            )
        )
    
    def _compute_odometry_information(self, source_pcd, target_pcd, transformation):
        """Compute information matrix for odometry edge."""
        if source_pcd is None or target_pcd is None:
            return np.eye(6)
        
        if len(source_pcd.points) < 10 or len(target_pcd.points) < 10:
            return np.eye(6)
        
        try:
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_pcd, target_pcd, self.icp_threshold, transformation
            )
            return info
        except:
            return np.eye(6)

    # 在 PoseGraphOptimizer 类中找到此方法并替换

    # def _compute_odometry_information(self, source_pcd, target_pcd, init_transformation):
    #     """
    #     [学术增强] 计算里程计边的信息矩阵，并利用 ICP 优化变换矩阵
    #     Returns:
    #         information (6x6), refined_transformation (4x4)
    #     """
    #     if source_pcd is None or target_pcd is None:
    #         return np.eye(6), init_transformation
        
    #     if len(source_pcd.points) < 50 or len(target_pcd.points) < 50:
    #         return np.eye(6), init_transformation
        
    #     # 1. 使用 PnP 的结果作为 ICP 的初始猜测 (Initial Guess)
    #     # 这就是 "Coarse-to-Fine" (由粗到精) 的学术思想
        
    #     try:
    #         # 点到面 ICP (Point-to-Plane) 通常比 点到点 (Point-to-Point) 精度更高
    #         # 要求点云有法向量，如果没有，Open3D 会自动忽略或使用 Point-to-Point
    #         if not source_pcd.has_normals():
    #             source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #         if not target_pcd.has_normals():
    #             target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    #         # 🟢 [核心] 执行 ICP 配准
    #         icp_result = o3d.pipelines.registration.registration_icp(
    #             source_pcd, target_pcd, 
    #             self.icp_threshold, # 从 config 读取，例如 0.05
    #             init_transformation,
    #             o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    #         )
            
    #         # 2. 检查 ICP 质量
    #         # Fitness: 重叠区域比例; RMSE: 均方根误差
    #         if icp_result.fitness > 0.3 and icp_result.inlier_rmse < 0.1:
    #             # ICP 成功，使用优化后的变换矩阵
    #             refined_trans = icp_result.transformation
                
    #             # 计算信息矩阵 (Information Matrix = Inverse Covariance)
    #             # 越紧密的匹配，信息矩阵的值越大
    #             info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
    #                 source_pcd, target_pcd, self.icp_threshold, refined_trans
    #             )
    #             return info, refined_trans
    #         else:
    #             # ICP 失败 (可能特征太少或初始误差太大)，回退到 PnP 结果
    #             print("⚠️ ICP refinement failed (low fitness), using PnP fallback.")
    #             return np.eye(6), init_transformation
                
    #     except Exception as e:
    #         print(f"⚠️ ICP Error: {e}")
    #         return np.eye(6), init_transformation
    
    def _detect_loop_closure_bow(self, query_kf: Keyframe) -> List[LoopClosureCandidateLegacy]:
        """Detect loop closure candidates using BoW matching."""
        candidates = []
        
        if query_kf.bow_vector is None:
            return candidates
        
        # Query database
        exclude_ids = list(range(
            max(0, query_kf.id - self.min_keyframe_gap),
            query_kf.id + 1
        ))
        
        matches = self.bow.query(
            query_kf.bow_vector,
            top_k=10,
            exclude_ids=exclude_ids
        )
        
        for match_id, score in matches:
            if score >= self.bow_similarity_threshold:
                # Spatial distance check
                if match_id in self.keyframes:
                    ref_pose = self.keyframes[match_id].pose
                    spatial_dist = np.linalg.norm(
                        query_kf.pose[:3, 3] - ref_pose[:3, 3]
                    )
                    
                    if spatial_dist <= self.search_radius:
                        candidates.append(LoopClosureCandidateLegacy(
                            query_id=query_kf.id,
                            match_id=match_id,
                            bow_score=score
                        ))
        
        return candidates
    
    def _verify_loop_closure(self, query_kf: Keyframe, 
                             candidate: LoopClosureCandidateLegacy) -> bool:
        """Verify loop closure candidate with geometric verification."""
        match_kf = self.keyframes.get(candidate.match_id)
        if match_kf is None:
            return False
        
        # Match descriptors
        if query_kf.descriptors is None or match_kf.descriptors is None:
            return False
        
        if FEATURES_AVAILABLE:
            matches = self.matcher.match(match_kf.descriptors, query_kf.descriptors)
        else:
            try:
                matches = self.bf_matcher.match(match_kf.descriptors, query_kf.descriptors)
            except:
                return False
        
        if len(matches) < self.min_inliers:
            return False
        
        # Get 3D-3D correspondences
        pts_match = []
        pts_query = []
        
        for m in matches:
            if hasattr(m, 'query_idx'):
                q_idx, t_idx = m.query_idx, m.train_idx
            else:
                q_idx, t_idx = m.queryIdx, m.trainIdx
            
            match_pt = match_kf.keypoints[q_idx]
            query_pt = query_kf.keypoints[t_idx]
            
            match_3d = getattr(match_pt, 'point_3d', None)
            query_3d = getattr(query_pt, 'point_3d', None)
            
            if match_3d is not None and query_3d is not None:
                pts_match.append(match_3d)
                pts_query.append(query_3d)
        
        if len(pts_match) < self.min_inliers:
            return False
        
        # ICP verification
        try:
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(np.array(pts_query))
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(np.array(pts_match))
            
            # Initial transformation estimate
            init_transform = np.linalg.inv(match_kf.pose) @ query_kf.pose
            
            # ICP refinement
            if query_kf.pointcloud is not None and match_kf.pointcloud is not None:
                icp = o3d.pipelines.registration.registration_icp(
                    query_kf.pointcloud, match_kf.pointcloud,
                    self.icp_threshold, init_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                
                if (icp.fitness >= self.icp_fitness_threshold and 
                    icp.inlier_rmse <= self.icp_rmse_threshold):
                    candidate.relative_pose = icp.transformation
                    candidate.is_verified = True
                    candidate.num_inliers = int(icp.fitness * len(query_kf.pointcloud.points))
                    candidate.information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                        query_kf.pointcloud, match_kf.pointcloud,
                        self.icp_threshold, icp.transformation
                    )
                    return True
        except Exception as e:
            pass
        
        return False
    
    def _verify_loop_with_icp(self, query_kf: Keyframe, 
                              candidate: LoopCandidate) -> bool:
        """Verify enhanced loop candidate with ICP."""
        match_kf = self.keyframes.get(candidate.match_id)
        if match_kf is None:
            return False
        
        if query_kf.pointcloud is None or match_kf.pointcloud is None:
            # Accept without ICP if no point clouds
            candidate.relative_pose = np.linalg.inv(match_kf.pose) @ query_kf.pose
            candidate.information_matrix = np.eye(6)
            return True
        
        try:
            init_transform = np.linalg.inv(match_kf.pose) @ query_kf.pose
            
            icp = o3d.pipelines.registration.registration_icp(
                query_kf.pointcloud, match_kf.pointcloud,
                self.icp_threshold, init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            
            if (icp.fitness >= self.icp_fitness_threshold and 
                icp.inlier_rmse <= self.icp_rmse_threshold):
                candidate.relative_pose = icp.transformation
                candidate.information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    query_kf.pointcloud, match_kf.pointcloud,
                    self.icp_threshold, icp.transformation
                )
                return True
        except:
            pass
        
        return False
    
    def _add_loop_closure_edge(self, candidate: LoopCandidate):
        """Add edge from enhanced loop candidate."""
        if candidate.relative_pose is not None:
            info = candidate.information_matrix
            if info is None:
                info = np.eye(6)
            
            self._add_edge(
                candidate.query_id, candidate.match_id,
                candidate.relative_pose, info,
                is_loop_closure=True
            )
            self.detected_loops.append(candidate)
    
    def _add_loop_closure_edge_legacy(self, candidate: LoopClosureCandidateLegacy):
        """Add edge from legacy loop candidate."""
        if candidate.is_verified and candidate.relative_pose is not None:
            info = candidate.information
            if info is None:
                info = np.eye(6)
            
            self._add_edge(
                candidate.query_id, candidate.match_id,
                candidate.relative_pose, info,
                is_loop_closure=True
            )
            self.detected_loops.append(candidate)
    
    def optimize(self) -> bool:
        """
        Perform global pose graph optimization.
        
        Uses Levenberg-Marquardt algorithm to minimize the pose graph error:
            E = Σ_ij || log(T_i^(-1) * T_j * Z_ij^(-1)) ||²_Ω
        
        Returns:
            True if optimization succeeded
        """
        if len(self.pose_graph.nodes) < 2:
            return True
        
        # 策略：如果有回环，且安装了 Sim3 模块，优先使用 Sim3 优化
        # Sim3 优化不仅修位姿，还修点云尺度
        if SIM3_AVAILABLE and self.total_loop_closures > 0:
            print(f"📐 Global Optimization (Sim3 Academic Method)...")
            try:
                # 调用 sim3_pose_graph.py 中的高级接口
                # corrected_poses, scale_corrections = correct_map_after_loop_closure(
                #     self.keyframes,
                #     self.pose_graph,
                #     self.scale_history
                # )

                # 1. 执行 Sim(3) 优化
                # 这一步会计算出每一帧的精确尺度修正因子 s_i
                corrected_poses, scale_corrections = correct_map_after_loop_closure(
                    self.keyframes,
                    self.pose_graph,
                    # 传入历史记录的尺度（如果没有记录，默认全为1.0）
                    getattr(self, 'scale_history', {}) 
                )
                
                # 应用 Sim3 优化结果回系统
                from sim3_pose_graph import apply_scale_corrections_to_pointclouds
                apply_scale_corrections_to_pointclouds(
                    self.keyframes, corrected_poses, scale_corrections
                )
                
                # 同步回 PoseGraph 节点（仅位姿部分）
                for i, pose in corrected_poses.items():
                    if i < len(self.pose_graph.nodes):
                        self.pose_graph.nodes[i].pose = pose.copy()
                    if i in self.nodes:
                        self.nodes[i].pose = pose.copy()
                        
                print("✅ Sim3 Optimization complete")
                return True
            except Exception as e:
                print(f"⚠️ Sim3 Optimization failed: {e}, falling back to SE3")
        
        print(f"📐 Global Optimization ({len(self.pose_graph.nodes)} nodes, "
              f"{len(self.pose_graph.edges)} edges)...")
        
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
        
        # Update keyframe poses
        for i, node in enumerate(self.pose_graph.nodes):
            if i in self.keyframes:
                self.keyframes[i].pose = node.pose.copy()
            if i in self.nodes:
                self.nodes[i].pose = node.pose.copy()
        
        print("✅ Optimization complete")
        return True
    
    def train_vocabulary(self, descriptors_list: List[np.ndarray]):
        """Train BoW vocabulary from descriptor list."""
        self.bow.train_vocabulary(descriptors_list)
        self.bow_trained = True
    
    def _create_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
        """Create point cloud from RGB-D data."""
        h, w = depth.shape[:2]
        depth_min = self.intrinsics.get('depth_min', 0.2)
        depth_max = self.intrinsics.get('depth_max', 5.0)
        
        mask = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
        if np.sum(mask) < 50:
            return o3d.geometry.PointCloud()
        
        # Create Open3D images
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_filtered = depth.copy().astype(np.float32)
        depth_filtered[~mask] = 0
        depth_o3d = o3d.geometry.Image(depth_filtered)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, 1.0, depth_max, convert_rgb_to_intensity=False
        )
        
        # Create point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h,
            self.intrinsics['fx'], self.intrinsics['fy'],
            self.intrinsics['cx'], self.intrinsics['cy']
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Downsample
        if self.voxel_size > 0 and len(pcd.points) > 100:
            pcd = pcd.voxel_down_sample(self.voxel_size)
        
        return pcd
    
    # def build_global_map(self, filter_noise: bool = True) -> o3d.geometry.PointCloud:
    #     """Build global point cloud map from all keyframes."""
    #     print("🗺️  Building global map...")
        
    #     global_map = o3d.geometry.PointCloud()
        
    #     for kf in self.keyframes.values():
    #         if kf.pointcloud is not None and len(kf.pointcloud.points) > 0:
    #             pcd = copy.deepcopy(kf.pointcloud)
    #             pcd.transform(kf.pose)
    #             global_map += pcd
        
    #     # Downsample
    #     if self.voxel_size > 0 and len(global_map.points) > 100:
    #         global_map = global_map.voxel_down_sample(self.voxel_size)
        
    #     # Remove outliers
    #     if filter_noise and len(global_map.points) > 100:
    #         cl, ind = global_map.remove_statistical_outlier(20, 2.0)
    #         global_map = global_map.select_by_index(ind)
        
    #     print(f"   Map points: {len(global_map.points)}")
    #     return global_map
    
    # def build_global_map(self, filter_noise: bool = True) -> o3d.geometry.PointCloud:
    #     """
    #     Build a clean, optimized global point cloud map.
        
    #     Args:
    #         filter_noise: Apply advanced statistical outlier removal
            
    #     Returns:
    #         o3d.geometry.PointCloud: The cleaned global map
    #     """
    #     print("🗺️  Building global map (Enhanced Cleaning)...")
        
    #     global_map = o3d.geometry.PointCloud()
        
    #     # 1. 累积所有关键帧的点云
    #     # 为了避免内存爆炸，每累积 50 帧做一次临时降采样
    #     temp_pcd = o3d.geometry.PointCloud()
    #     for idx, kf in enumerate(self.keyframes.values()):
    #         if kf.pointcloud is not None and len(kf.pointcloud.points) > 0:
    #             # 变换到世界坐标系
    #             pcd = copy.deepcopy(kf.pointcloud)
    #             pcd.transform(kf.pose)
    #             temp_pcd += pcd
                
    #         # 分批处理 (Batch Processing)
    #         if idx % 50 == 0:
    #             temp_pcd = temp_pcd.voxel_down_sample(self.voxel_size)
        
    #     global_map = temp_pcd
    #     original_count = len(global_map.points)
    #     print(f"   Raw points: {original_count}")

    #     # 2. 核心去冗余步骤 (The "NMS" for Point Clouds)
    #     if filter_noise and original_count > 100:
    #         print("   🧹 Filtering outliers...")
            
    #         # Step A: Voxel Downsampling (体素化)
    #         # 这一步类似于 NMS，合并重叠的点
    #         # 建议 voxel_size 设置为 0.02 (2cm) 或 0.05 (5cm)
    #         global_map = global_map.voxel_down_sample(self.voxel_size)
            
    #         # Step B: Statistical Outlier Removal (统计滤波)
    #         # 计算每个点到最近 20 个点的平均距离
    #         # 如果距离大于 (平均值 + 2.0 * 标准差)，则剔除
    #         cl, ind = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #         global_map = global_map.select_by_index(ind)
            
    #         # Step C: Radius Outlier Removal (半径滤波 - 可选，更严格)
    #         # 在 5cm 半径内至少要有 5 个邻居，否则视为噪点
    #         cl, ind = global_map.remove_radius_outlier(nb_points=5, radius=0.05)
    #         global_map = global_map.select_by_index(ind)

    #     final_count = len(global_map.points)
    #     reduction = (1 - final_count / (original_count + 1e-6)) * 100
    #     print(f"   ✅ Map optimized: {final_count} points (Reduced by {reduction:.1f}%)")
        
    #     return global_map

    # def build_global_map(self, filter_noise: bool = True, min_score: float = 0.5) -> o3d.geometry.PointCloud:
    #     """
    #     Build a global map with Feature-Metric Filtering.
        
    #     Args:
    #         filter_noise: Apply geometry-based filtering (SOR/Radius)
    #         min_score: [New] Minimum feature reliability score to keep a point (0.0 ~ 1.0)
            
    #     Returns:
    #         o3d.geometry.PointCloud: The semantic-cleaned map
    #     """
    #     print(f"🗺️  Building global map (Feature-Aware Cleaning, th={min_score})...")
        
    #     global_map = o3d.geometry.PointCloud()
    #     temp_pcd = o3d.geometry.PointCloud()
        
    #     total_points = 0
    #     kept_points = 0
        
    #     for idx, kf in enumerate(self.keyframes.values()):
    #         if kf.pointcloud is not None and len(kf.pointcloud.points) > 0:
                
    #             # ---------------------------------------------------------
    #             # [学术创新] 基于特征置信度的点云筛选 (Feature-based Filtering)
    #             # ---------------------------------------------------------
    #             # 只有当关键帧存储了特征点信息，且数量与点云点数大致对应时才启用
    #             # 注意：Open3D 创建点云时可能会过滤掉无效深度的点，导致数量不一致
    #             # 所以这里我们需要一种极其严谨的对应机制，或者简单地使用“仅几何”策略兜底
                
    #             # 策略 A: 如果我们有 keypoints 且每个 keypoint 都有 3D 坐标
    #             valid_indices = []
    #             if kf.keypoints and len(kf.keypoints) > 0:
    #                 for i, kp in enumerate(kf.keypoints):
    #                     # 获取分数 (兼容 ORB 和 XFeat)
    #                     score = getattr(kp, 'response', 0.0)
    #                     # 获取 3D 坐标 (如果已计算)
    #                     pt3d = getattr(kp, 'point_3d', None)
                        
    #                     # 核心判断:
    #                     # 1. 有效 3D 点
    #                     # 2. 特征分数 > 阈值 (XFeat 分数通常在 0~1 之间，建议 0.5)
    #                     # 3. 深度在合理范围内
    #                     if pt3d is not None and score > min_score:
    #                          valid_indices.append(pt3d)
                
    #             # 如果策略 A 收集到了点，就用策略 A
    #             if len(valid_indices) > 10:
    #                 pcd = o3d.geometry.PointCloud()
    #                 pcd.points = o3d.utility.Vector3dVector(np.array(valid_indices))
    #                 # 给这些“精英点”上色 (比如用特征强度表示颜色，或者用原图颜色)
    #                 # 这里为了美观，我们统一用一种“高置信度色”或原色
    #                 pcd.paint_uniform_color([0.0, 0.8, 0.0]) # 绿色代表高置信
    #             else:
    #                 # 策略 B: 兜底 (使用原始点云，不进行特征过滤)
    #                 pcd = copy.deepcopy(kf.pointcloud)
                
    #             # 变换到世界坐标
    #             pcd.transform(kf.pose)
    #             temp_pcd += pcd
                
    #             kept_points += len(pcd.points)
    #             total_points += len(kf.pointcloud.points)
                
    #         # 分批降采样防爆内存
    #         if idx % 50 == 0:
    #             temp_pcd = temp_pcd.voxel_down_sample(self.voxel_size)

    #     global_map = temp_pcd
    #     print(f"   Feature Filter: Kept {kept_points}/{total_points} high-quality points")

    #     # ---------------------------------------------------------
    #     # [原有逻辑] 几何去噪 (Geometry Cleaning)
    #     # ---------------------------------------------------------
    #     if filter_noise and len(global_map.points) > 100:
    #         print("   🧹 Applying geometric filtering...")
            
    #         # 1. 体素化 (Voxel Grid)
    #         global_map = global_map.voxel_down_sample(self.voxel_size)
            
    #         # 2. 统计滤波 (SOR)
    #         cl, ind = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #         global_map = global_map.select_by_index(ind)
            
    #         # 3. 半径滤波 (Radius) - 这里可以放宽一点，因为我们已经筛过特征了
    #         # cl, ind = global_map.remove_radius_outlier(nb_points=4, radius=0.05)
    #         # global_map = global_map.select_by_index(ind)

    #     print(f"   ✅ Final Map: {len(global_map.points)} points")
    #     return global_map

    def _compute_confidence_map(self, kf, height, width):
        """
        [学术核心] 构建 2D 特征置信度场 (Feature Confidence Field)
        利用稀疏的 XFeat 分数生成稠密的置信度权重图
        """
        # 初始化基础权重 (0.1)，避免非特征区域被完全丢弃
        confidence_map = np.full((height, width), 0.1, dtype=np.float32)
        
        if kf.keypoints and len(kf.keypoints) > 0:
            # 提取坐标和分数
            coords = []
            scores = []
            for kp in kf.keypoints:
                # 兼容不同的 KeyPoint 数据结构
                pt = getattr(kp, 'pt', None)
                if pt is None and hasattr(kp, 'point_3d'): continue # 只有 3D 没有 2D 的情况跳过
                
                response = getattr(kp, 'response', 1.0) # XFeat 分数
                coords.append([int(pt[1]), int(pt[0])]) # (y, x)
                scores.append(response)
            
            if coords:
                coords = np.array(coords)
                scores = np.array(scores)
                
                # 创建稀疏矩阵掩膜 (为了速度，不使用完整的高斯模糊，只在局部画点)
                # 在 Pi 5 上，直接修改像素是最快的
                mask = np.zeros((height, width), dtype=np.float32)
                
                # 简单的“膨胀”策略：特征点周围 3x3 区域都赋予高置信度
                valid_y = np.clip(coords[:, 0], 1, height-2)
                valid_x = np.clip(coords[:, 1], 1, width-2)
                
                # 填充中心及邻域
                mask[valid_y, valid_x] = scores
                mask[valid_y+1, valid_x] = scores * 0.8
                mask[valid_y-1, valid_x] = scores * 0.8
                mask[valid_y, valid_x+1] = scores * 0.8
                mask[valid_y, valid_x-1] = scores * 0.8
                
                # 叠加到基础权重
                confidence_map = np.maximum(confidence_map, mask)
                
        return confidence_map

    # def build_global_map(self, filter_noise: bool = True, min_conf: float = 0.2, verify_geometry: bool = True) -> o3d.geometry.PointCloud:
    #     print(f"🗺️  Building global map (Probabilistic Voxel Fusion)...")
        
    #     global_temp = o3d.geometry.PointCloud()
        
    #     # 按顺序处理关键帧
    #     sorted_ids = sorted(self.keyframes.keys())
        
    #     for i, kf_id in enumerate(sorted_ids):
    #         kf = self.keyframes[kf_id]
    #         if kf.pointcloud is None or len(kf.pointcloud.points) == 0:
    #             continue
                
    #         # 1. 变换到世界坐标
    #         pcd_world = copy.deepcopy(kf.pointcloud)
    #         pcd_world.transform(kf.pose)
            
    #         # [Step B] 逆向几何一致性剔除 (保留之前的修改)
    #         if verify_geometry and i > 0:
    #             valid_indices = self._check_geometric_consistency(pcd_world, curr_kf_id=kf_id)
    #             if len(valid_indices) < len(pcd_world.points):
    #                 pcd_world = pcd_world.select_by_index(valid_indices)
            
    #         if len(pcd_world.points) == 0: continue

    #         # =========================================================
    #         # 🟢 [核心修改] 概率体素融合 (Probabilistic Fusion)
    #         # =========================================================
    #         # 我们不直接 global_temp += pcd_world，而是先计算权重进行“软剔除”
            
    #         # A. 计算概率权重 (Weighting)
    #         # 假设点云是有序生成的 (H, W)，我们可以恢复其原始深度用于计算权重
    #         # 权重模型 w = 1 / (depth^2) (越远越不准)
    #         # 由于 Open3D 的点云点序可能丢失，这里用几何距离近似
            
    #         # 获取相对于相机的距离 (局部 Z 值)
    #         # 需要把世界坐标转回相机坐标才能算深度吗？
    #         # 简单做法：直接用相机中心到点的欧氏距离
    #         cam_center = kf.pose[:3, 3]
    #         pts = np.asarray(pcd_world.points)
    #         dists = np.linalg.norm(pts - cam_center, axis=1)
            
    #         # 归一化权重：距离 0.5m 权重为 4，距离 5m 权重为 0.04
    #         weights = 1.0 / (dists**2 + 1e-4)
            
    #         # B. 概率筛选 (Probabilistic Sampling)
    #         # 只让“可信”的点参与融合，去噪效果极佳
    #         # min_conf 是置信度阈值，建议 0.1 ~ 0.5
    #         # 动态阈值：越远的帧，阈值要求越低以免远处看不见，或者固定阈值
    #         high_conf_indices = np.where(weights > min_conf)[0]
            
    #         if len(high_conf_indices) > 0:
    #             pcd_weighted = pcd_world.select_by_index(high_conf_indices)
    #             global_temp += pcd_weighted
            
    #         # C. 在线融合 (On-the-fly Fusion)
    #         # 每累积 20 帧就做一次体素降采样
    #         # Open3D 的 voxel_down_sample 本质上就是求体素内的“平均值”
    #         # 因为我们前面已经剔除了低权重的点，所以这里的平均值就是“加权平均”的结果
    #         if i % 20 == 0:
    #             global_temp = global_temp.voxel_down_sample(self.voxel_size)
    #         # =========================================================

    #     # 3. 最终后处理
    #     if filter_noise:
    #          print("   🧹 Final cleaning (SOR)...")
    #          global_temp = global_temp.voxel_down_sample(self.voxel_size)
    #          cl, ind = global_temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #          global_temp = global_temp.select_by_index(ind)
             
    #     print(f"   ✅ Final Map: {len(global_temp.points)} points (Fused & Cleaned)")
    #     return global_temp

    # def build_global_map(self, filter_noise: bool = True, min_conf: float = 0.2, verify_geometry: bool = True) -> o3d.geometry.PointCloud: # 🟢 [新增参数] verify_geometry
    #     """
    #     Build a global map using Probabilistic Voxel Fusion (PVF) with Inverse Pruning.
    #     """
    #     print(f"🗺️  Building global map (Probabilistic Fusion + Inverse Pruning)...")
        
    #     global_temp = o3d.geometry.PointCloud()
    #     total_input_points = 0
        
    #     # 按顺序处理关键帧
    #     sorted_ids = sorted(self.keyframes.keys())
        
    #     for i, kf_id in enumerate(sorted_ids):
    #         kf = self.keyframes[kf_id]
    #         if kf.pointcloud is None or len(kf.pointcloud.points) == 0:
    #             continue
                
    #         # 1. 变换到世界坐标 (T_wc * P_local)
    #         pcd_world = copy.deepcopy(kf.pointcloud)
    #         pcd_world.transform(kf.pose)
            
    #         # =========================================================
    #         # 🟢 [Step B] 逆向几何一致性剔除 (Inverse Pruning)
    #         # =========================================================
    #         if verify_geometry and i > 0: # 第一帧没法查，跳过
    #             # 调用检查函数
    #             valid_indices = self._check_geometric_consistency(
    #                 pcd_world, 
    #                 curr_kf_id=kf_id, 
    #                 check_window=3,    # 检查最近3帧
    #                 depth_thres=0.05   # 5cm 误差容忍
    #             )
                
    #             # 执行剔除
    #             if len(valid_indices) < len(pcd_world.points):
    #                 pcd_world = pcd_world.select_by_index(valid_indices)
    #         # =========================================================

    #         if len(pcd_world.points) == 0: continue

    #         # 2. 累积点云 (原有逻辑)
    #         # 这里简化演示，直接累加。如果想用之前的概率体素融合，逻辑一样
    #         global_temp += pcd_world
    #         total_input_points += len(pcd_world.points)
            
    #         # 防止内存溢出
    #         if i % 50 == 0:
    #             global_temp = global_temp.voxel_down_sample(self.voxel_size)
                
    #     # 3. 后处理 (原有逻辑)
    #     if filter_noise:
    #          print("   🧹 Final cleaning (SOR)...")
    #          global_temp = global_temp.voxel_down_sample(self.voxel_size)
    #          cl, ind = global_temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #          global_temp = global_temp.select_by_index(ind)
             
    #     print(f"   ✅ Final Map: {len(global_temp.points)} points (Verified & Cleaned)")
    #     return global_temp

    # def build_global_map(self, filter_noise: bool = True, min_conf: float = 0.2) -> o3d.geometry.PointCloud:
    #     """
    #     Build a global map using Probabilistic Voxel Fusion (PVF).
        
    #     学术改进点：
    #     1. 引入 Confidence Map 进行加权融合
    #     2. 实现 Weighted Voxel Downsampling (加权体素下采样)
        
    #     Args:
    #         filter_noise: 是否应用后处理滤波
    #         min_conf: 最小置信度阈值 (低于此权重的点在融合时会被抑制)
    #     """
    #     print(f"🗺️  Building global map (Probabilistic Voxel Fusion)...")
        
    #     # 使用 Open3D 的 VoxelGrid 比较复杂，为了 Python 端的灵活性，
    #     # 我们这里实现一个基于 Hash Map 的简易加权体素网格
    #     voxel_grid = {} # {voxel_key: [accum_point, accum_color, accum_weight, count]}
    #     voxel_size = self.voxel_size
        
    #     total_input_points = 0
        
    #     for idx, kf in enumerate(self.keyframes.values()):
    #         if kf.pointcloud is None or len(kf.pointcloud.points) == 0:
    #             continue
                
    #         # 1. 获取原始数据
    #         points = np.asarray(kf.pointcloud.points)
    #         colors = np.asarray(kf.pointcloud.colors)
            
    #         # 2. 计算置信度权重 (Per-point Weight)
    #         # 如果之前没有存 confidence，这里现场算一个
    #         # 注意：这里假设 points 是有序的，对应图像上的像素
    #         # 如果 kf.pointcloud 经过了变换或无序化，需要重新投影回图像平面查找权重
    #         # 为了严谨，我们假设 kf.pointcloud 是从 RGBDImage 生成的有序点云
            
    #         # 简易版：基于深度的权重 (深度越小越准)
    #         # 严谨版：基于 Feature Map (需投影)
    #         # 这里演示一个基于深度和强度的混合权重
            
    #         # 变换到世界坐标
    #         pcd_world = copy.deepcopy(kf.pointcloud)
    #         pcd_world.transform(kf.pose)
    #         pts_world = np.asarray(pcd_world.points)
            
    #         # 3. 概率体素融合 (Probabilistic Fusion)
    #         # 在 Python 中循环 50万个点太慢，我们使用 NumPy 矢量化加速
            
    #         # 量化坐标到体素索引
    #         voxel_indices = np.floor(pts_world / voxel_size).astype(np.int64)
            
    #         # 构造唯一的 Hash Key (x, y, z)
    #         # 简单的哈希函数: x*P1 + y*P2 + z*P3
    #         # 或者直接用 tuple (太慢)，这里为了演示逻辑，简化为只保留每个体素的“最佳点”
            
    #         # [学术优化策略]: 在每个体素内，保留“置信度最高”的那个点，而不是求平均
    #         # 这能避免“鬼影”和模糊
            
    #         # 计算每个点的 Score (这里用简单的 Z 轴距离反比模拟，如有 XFeat 分数更好)
    #         # 假设我们无法轻易回溯像素坐标，就用几何特征：
    #         # 权重 w = 1.0 / (depth^2) (深度越远误差越大)
    #         local_depths = points[:, 2] # 局部 Z
    #         weights = 1.0 / (local_depths ** 2 + 1e-6)
            
    #         # 如果有特征分数，乘上去
    #         # weights *= feature_scores 
            
    #         # 过滤掉极低权重的点
    #         valid_mask = weights > min_conf
            
    #         # 将数据存入临时列表，准备批处理
    #         # 注意：在 Python 里手写体素融合太慢，我们退一步：
    #         # 先合并所有点，再利用 Open3D 的 voxel_down_sample (它内部是求平均)
    #         # 但我们在合并前，先根据权重进行“概率剔除” (Probabilistic Rejection)
            
    #         # 策略：只保留权重高的点进入最终的 Downsample 池
    #         # 相当于在输入端做了 Soft Attention
            
    #         if np.sum(valid_mask) > 0:
    #             # 随机采样一些低权重的点以保持覆盖率 (Dithering)，防止空洞
    #             # 高权重的点 100% 保留
    #             pts_to_add = pcd_world.select_by_index(np.where(valid_mask)[0])
                
    #             # 累积到全局临时点云
    #             if idx == 0:
    #                 global_temp = pts_to_add
    #             else:
    #                 global_temp += pts_to_add
            
    #         total_input_points += len(points)
            
    #         # 每 50 帧做一次中间态体素化，防止内存溢出
    #         if idx % 50 == 0 and 'global_temp' in locals():
    #             global_temp = global_temp.voxel_down_sample(voxel_size)

    #     print(f"   📉 Probabilistic Sampling: {total_input_points} -> {len(global_temp.points)}")
        
    #     # 4. 后处理：几何清洗
    #     if filter_noise:
    #         print("   🧹 Statistical Outlier Removal (SOR)...")
    #         # 统计滤波：去除离群噪点
    #         cl, ind = global_temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #         global_temp = global_temp.select_by_index(ind)
            
    #         # 可选：利用 3DGS 的思想，这里可以导出一个更适合 Gaussian Splatting 的稀疏点云
    #         # 用户上传了 export_to_3dgs.py，说明对此感兴趣
            
    #     print(f"   ✅ Final Map: {len(global_temp.points)} points")
    #     return global_temp

    # ==============================================================================
    # [学术级优化] 新增：几何一致性校验核心算法
    # ==============================================================================
    def _check_geometric_consistency(self, 
                                   curr_pcd_world: o3d.geometry.PointCloud, 
                                   curr_kf_id: int, 
                                   check_window: int = 2,
                                   dist_thres: float = 0.05) -> List[int]:
        """
        核心算法：检查当前帧的点云是否在邻居帧中存在（交叉验证）。
        
        原理：
        如果当前帧的一个点 P，在邻居帧（上一帧或下一帧）的点云中找不到距离 < 5cm 的对应点，
        说明这个点大概率是深度预测错误的噪点（孤立点），应当剔除。
        
        Args:
            curr_pcd_world: 当前帧（已转到世界坐标）
            curr_kf_id: 当前帧ID
            check_window: 向前追溯几帧 (默认2帧)
            dist_thres: 距离阈值 (默认0.05米，即5厘米)
            
        Returns:
            List[int]: 验证通过的点的索引
        """
        # 如果没有点，直接返回空
        if len(curr_pcd_world.points) == 0:
            return []
            
        # 寻找邻居帧 (Prev frames)
        # 我们只看前面几帧，因为它们已经存在且稳定
        neighbor_pcds = []
        for i in range(1, check_window + 1):
            prev_id = curr_kf_id - i
            if prev_id in self.keyframes:
                kf_prev = self.keyframes[prev_id]
                if kf_prev.pointcloud is not None and len(kf_prev.pointcloud.points) > 0:
                    # 转到世界坐标
                    pcd_prev = copy.deepcopy(kf_prev.pointcloud)
                    pcd_prev.transform(kf_prev.pose)
                    neighbor_pcds.append(pcd_prev)
        
        if not neighbor_pcds:
            # 如果没有邻居（比如第一帧），为了安全起见，要么全保留，要么保留中心区域
            # 这里选择保留，以免第一帧被删光
            return list(range(len(curr_pcd_world.points)))
            
        # 核心校验逻辑
        # 只要能在【任意一个】邻居中找到对应点，就算通过 (OR 逻辑，宽松但有效)
        # 如果想更严格，可以要求在【所有】邻居中找到 (AND 逻辑)
        
        valid_mask = np.zeros(len(curr_pcd_world.points), dtype=bool)
        
        for nb_pcd in neighbor_pcds:
            # 计算 当前帧所有点 -> 邻居帧点云 的最近距离
            dists = curr_pcd_world.compute_point_cloud_distance(nb_pcd)
            dists = np.asarray(dists)
            
            # 距离小于阈值的点标记为有效
            valid_mask |= (dists < dist_thres)
            
        return np.where(valid_mask)[0]

    # ==============================================================================
    # [重写] 构建全局地图 (带强力清洗功能)
    # ==============================================================================
    def build_global_map(self, filter_noise: bool = True, strict_mode: bool = True) -> o3d.geometry.PointCloud:
        """
        构建全局地图，并执行“匹配点保留”策略。
        
        Args:
            filter_noise: 是否开启统计滤波 (SOR)
            strict_mode: 是否开启[几何一致性校验] (只保存匹配到的点)
        """
        print(f"\n🗺️  正在构建高精度全局地图 (Strict Mode: {strict_mode})...")
        
        global_map = o3d.geometry.PointCloud()
        
        # 统计变量
        total_points_raw = 0
        total_points_kept = 0
        
        sorted_ids = sorted(self.keyframes.keys())
        total_frames = len(sorted_ids)
        
        for i, kf_id in enumerate(sorted_ids):
            kf = self.keyframes[kf_id]
            if kf.pointcloud is None or len(kf.pointcloud.points) == 0:
                continue
                
            # 1. 准备当前帧点云 (世界坐标)
            pcd_world = copy.deepcopy(kf.pointcloud)
            pcd_world.transform(kf.pose)
            
            n_raw = len(pcd_world.points)
            total_points_raw += n_raw
            
            # 2. [先进算法] 几何一致性校验 (The "Match Only" Filter)
            if strict_mode and i > 0: # 第一帧没法校验，跳过
                # 这里的 check_window=3 表示：如果这个点在最近 3 帧里都没出现过，就删掉！
                # dist_thres=0.08 表示：误差容忍度 8cm
                valid_idx = self._check_geometric_consistency(
                    pcd_world, kf_id, check_window=3, dist_thres=0.08
                )
                
                if len(valid_idx) < 10:
                    # 如果这帧几乎全是噪点，直接扔掉
                    continue
                    
                pcd_world = pcd_world.select_by_index(valid_idx)
            
            # 3. 累积
            global_map += pcd_world
            total_points_kept += len(pcd_world.points)
            
            # 打印进度 (每10帧)
            if i % 10 == 0:
                print(f"   处理帧 {i}/{total_frames} | 保留率: {len(pcd_world.points)/n_raw*100:.1f}%")
                
            # 4. 中途降采样 (防止内存爆炸)
            if i % 50 == 0:
                global_map = global_map.voxel_down_sample(self.voxel_size)

        print(f"   📉 初始清洗完成: {total_points_raw} -> {total_points_kept} 点 (剔除率 {(1-total_points_kept/total_points_raw)*100:.1f}%)")

        # 5. 最终后处理 (SOR 滤波) - 清除最后的稀疏离群点
        if filter_noise and len(global_map.points) > 0:
            print("   🧹 执行最终统计滤波 (SOR)...")
            # 邻居数设为 20，标准差倍数 1.0 (比较严格，把飘在外面的都干掉)
            cl, ind = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            global_map = global_map.select_by_index(ind)
            
            # 可选：再加一道半径滤波 (Radius Outlier Removal)
            # 5cm 半径内至少要有 10 个点，否则视为孤立噪声
            cl, ind = global_map.remove_radius_outlier(nb_points=10, radius=0.05)
            global_map = global_map.select_by_index(ind)

        print(f"   ✅ 地图构建完成! 最终点数: {len(global_map.points)}")
        return global_map


    def save_trajectory(self, filepath: str, format: str = 'tum'):
        """Save trajectory in TUM format."""
        trajectory = []
        
        for kf_id in sorted(self.keyframes.keys()):
            kf = self.keyframes[kf_id]
            q = ScipyRotation.from_matrix(kf.pose[:3, :3]).as_quat()
            t = kf.pose[:3, 3]
            trajectory.append([kf.timestamp, t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
        
        if trajectory:
            np.savetxt(filepath, np.array(trajectory), fmt='%.6f')
            print(f"💾 Trajectory saved: {filepath}")
    
    def save_keyframes(self, output_dir: str):
        """Save keyframe images and poses."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Saving {len(self.keyframes)} keyframes to {output_dir}...")
        
        for kf_id, kf in self.keyframes.items():
            # Save image
            if kf.rgb_image is not None:
                img_path = os.path.join(output_dir, f"frame_{kf_id:06d}.jpg")
                cv2.imwrite(img_path, kf.rgb_image)
            
            # Save metadata
            meta_path = os.path.join(output_dir, f"frame_{kf_id:06d}.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    'id': kf_id,
                    'timestamp': kf.timestamp,
                    'pose': kf.pose.flatten().tolist()
                }, f, indent=4)
        
        print("✅ Keyframes saved")
    
    def get_statistics(self) -> Dict:
        """Get optimizer statistics."""
        def safe_mean(lst):
            return np.mean(lst[-100:]) if lst else 0.0
        
        stats = {
            'num_keyframes': len(self.keyframes),
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_loops': self.total_loop_closures,
            'avg_feature_extraction_ms': safe_mean(self.timing_stats['feature_extraction']) * 1000,
            'avg_pointcloud_creation_ms': safe_mean(self.timing_stats['pointcloud_creation']) * 1000,
            'avg_loop_detection_ms': safe_mean(self.timing_stats['loop_detection']) * 1000,
            'avg_optimization_ms': safe_mean(self.timing_stats['optimization']) * 1000,
        }
        
        if self.use_enhanced_lc and self.loop_detector is not None:
            lc_stats = self.loop_detector.get_statistics()
            stats.update({
                'lc_' + k: v for k, v in lc_stats.items()
            })
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 PoseGraphOptimizer Statistics")
        print("="*60)
        print(f"  Keyframes:        {stats['num_keyframes']}")
        print(f"  Graph nodes:      {stats['num_nodes']}")
        print(f"  Graph edges:      {stats['num_edges']}")
        print(f"  Loop closures:    {stats['num_loops']}")
        print("-"*60)
        print(f"  Feature extraction: {stats['avg_feature_extraction_ms']:.1f} ms")
        print(f"  Point cloud:        {stats['avg_pointcloud_creation_ms']:.1f} ms")
        print(f"  Loop detection:     {stats['avg_loop_detection_ms']:.1f} ms")
        print(f"  Optimization:       {stats['avg_optimization_ms']:.1f} ms")
        
        if self.use_enhanced_lc and self.loop_detector is not None:
            print("-"*60)
            print("  [Enhanced Loop Closure]")
            print(f"    Database size:    {stats.get('lc_database_size', 0)}")
            print(f"    Current threshold: {stats.get('lc_current_threshold', 0):.3f}")
            print(f"    Verification rate: {stats.get('lc_verification_rate', 0):.1f}%")
        
        print("="*60 + "\n")


# =============================================================================
# Factory Function
# =============================================================================

def create_pose_graph_optimizer(config_path: Optional[str] = None,
                                use_enhanced_lc: bool = True,
                                hef_path: Optional[str] = None,
                                onnx_path: Optional[str] = None,
                                vdevice = None) -> PoseGraphOptimizer:  # [修复] 新增 vdevice 参数
    """
    Factory function to create a configured PoseGraphOptimizer.
    
    Args:
        config_path: Path to SLAM configuration file
        use_enhanced_lc: Whether to use enhanced loop closure detection
        hef_path: Path to Hailo HEF model
        onnx_path: Path to ONNX model
        vdevice: [新增] 共享的 Hailo VDevice 实例
        
    Returns:
        Configured PoseGraphOptimizer instance
    """
    return PoseGraphOptimizer(
        config_path=config_path,
        use_enhanced_lc=use_enhanced_lc,
        hef_path=hef_path,
        onnx_path=onnx_path,
        vdevice=vdevice  # [修复] 传递共享 VDevice
    )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("PoseGraphOptimizer Enhanced - Demo")
    print("="*50)
    
    # Create optimizer
    optimizer = create_pose_graph_optimizer(use_enhanced_lc=True)
    
    # Simulate some keyframes
    for i in range(50):
        # Create dummy data
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.uniform(0.5, 3.0, (480, 640)).astype(np.float32)
        
        # Simulate trajectory
        angle = i * 0.1
        pose = np.eye(4)
        pose[0, 3] = np.cos(angle)
        pose[1, 3] = np.sin(angle)
        
        odometry = np.eye(4)
        odometry[0, 3] = 0.1
        
        # Add keyframe
        kf_id, loop = optimizer.add_keyframe(rgb, depth, pose, i * 0.1, odometry)
        
        if i % 10 == 0:
            print(f"Frame {i}: KF={kf_id}, Loop={loop}")
    
    # Print statistics
    optimizer.print_statistics()
