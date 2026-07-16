#!/usr/bin/env python3
"""
================================================================================
PiSLAM Pose Graph Optimization Module (VIO Enhanced / Tightly-Coupled)
================================================================================

This module implements a robust back-end for Visual-Inertial SLAM (VIO).

Key Improvements (Academic & Engineering):
1. GTSAM Backend: Replaced generic graph optimization with ISAM2 (Bayes Tree).
2. Tightly-Coupled VIO: Joint optimization of Pose, Velocity, and IMU Biases.
3. Robust Parameter Setting: Auto-detects GTSAM API version (Properties vs Setters).
4. Legacy Retention: Kept all original map building, feature extraction, and 
   geometric verification logic intact.

Author: Academic SLAM Implementation (Fused with User Engineering)
================================================================================
"""

import numpy as np
import open3d as o3d
import cv2
import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import yaml
import copy
from scipy.spatial.transform import Rotation as ScipyRotation

# [GTSAM] 引入因子图核心库
try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B  # X:Pose, V:Velocity, B:Bias
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("⚠️ GTSAM not found! VIO features will be disabled. Please install gtsam.")

from local_bundle_adjustment import SlidingWindowBA

# Import feature extraction modules
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
# Helper Function for Robust GTSAM Config (Fixes AttributeError)
# =============================================================================
def _set_gtsam_param(obj, name_base, value):
    """
    学术级兼容性处理：尝试用 setter 或 property 设置参数
    Example: name_base='relinearizeSkip', value=1
    Tries: obj.setRelinearizeSkip(1) OR obj.relinearizeSkip = 1
    """
    setter_name = "set" + name_base[0].upper() + name_base[1:]
    
    # 1. 优先尝试 Setter (GTSAM 旧版风格)
    if hasattr(obj, setter_name):
        try:
            getattr(obj, setter_name)(value)
            return
        except Exception:
            pass # 继续尝试属性赋值
            
    # 2. 尝试直接属性赋值 (GTSAM 新版风格)
    if hasattr(obj, name_base):
        try:
            setattr(obj, name_base, value)
            return
        except Exception:
            pass

    # 3. 都没有成功，打印警告但不崩溃
    print(f"⚠️ Warning: Could not set GTSAM param '{name_base}'. Ignoring.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Keyframe:
    """
    Represents a keyframe (State Variable) in the SLAM system.
    Updated to include VIO states.
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
    
    # [VIO 新增状态]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_bias: Optional[object] = None # gtsam.imuBias.ConstantBias


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
# Pose Graph Optimizer (VIO Enhanced Version)
# =============================================================================

class PoseGraphOptimizer:
    """
    The Back-end Optimizer for PiSLAM with Tightly-Coupled VIO capabilities.
    Manages both GTSAM (for optimization) and Open3D (for mapping/visualization).
    """
    
    def __init__(self, config_path: Optional[str] = None,
                 use_enhanced_lc: bool = True,
                 hef_path: Optional[str] = None,
                 onnx_path: Optional[str] = None,
                 vdevice = None): 
        """
        Initialize the optimizer with VIO support.
        """
        # [基础] 共享资源与配置
        self.vdevice = vdevice
        self.scale_tracker = ScaleTracker()
        self.scale_history = {} 
        self._load_config(config_path)

        # [VIO 核心] GTSAM 初始化
        self.isam_healthy = False
        if GTSAM_AVAILABLE:
            try:
                # ISAM2 参数配置
                params = gtsam.ISAM2Params()
                
                # 🔧 [核心修复] 使用鲁棒的参数设置函数
                _set_gtsam_param(params, 'relinearizeThreshold', 0.1)
                _set_gtsam_param(params, 'relinearizeSkip', 1)
                
                # 其他可选参数
                # _set_gtsam_param(params, 'enableRelinearization', True)

                self.isam = gtsam.ISAM2(params)
                
                self.graph = gtsam.NonlinearFactorGraph()
                self.initial_estimate = gtsam.Values()
                
                # 噪声模型定义
                self.vo_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])) 
                self.vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
                # self.bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

                # [优化] 将初始 Bias 的约束收紧 100 倍
                # 告诉优化器："我很确定初始 Bias 就是 0，别乱猜！"
                self.bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5]*6))
                
                self.vio_initialized = False
                self.isam_healthy = True
                print("✅ [Backend] GTSAM ISAM2 Initialized (VIO Ready)")
                
            except Exception as e:
                print(f"❌ [Backend] GTSAM Init Failed: {e}")
                print("   System will fallback to basic graph optimization (Open3D).")
                self.isam = None 
        
        # [原有] 局部 BA
        self.local_ba = SlidingWindowBA(window_size=10)
        self.ba_initialized = False
        
        # [原有] 特征提取
        if FEATURES_AVAILABLE:
            self.extractor = ORBExtractor(config_path)
            self.matcher = FeatureMatcher(config_path)
        else:
            self.orb = cv2.ORB_create(1000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # [原有] 回环检测
        self.use_enhanced_lc = use_enhanced_lc and ENHANCED_LC_AVAILABLE
        if self.use_enhanced_lc:
            print("\n🚀 Using Enhanced Loop Closure Detection (VLAD Mode)")
            self.loop_detector = create_enhanced_loop_detector(
                config_path, hef_path, onnx_path, vdevice=self.vdevice
            )
            self.bow = None 
            self.bow_trained = True 
        else:
            print("\n📚 Using FAISS-accelerated Bag-of-Words")
            if ENHANCED_LC_AVAILABLE:
                self.bow = FAISSBagOfWords(vocabulary_size=self.vocabulary_size)
            else:
                from features import BagOfWords
                self.bow = BagOfWords(vocabulary_size=self.vocabulary_size, vocabulary_depth=self.vocabulary_depth)
            self.bow_trained = False
            self.loop_detector = None
        
        # Graph data structures
        self.keyframes: Dict[int, Keyframe] = {}
        self.next_keyframe_id = 0
        self.detected_loops: List = []
        
        # Open3D PoseGraph (用于保留 build_global_map 的兼容性)
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.nodes: Dict[int, PoseGraphNode] = {}
        self.edges: List[PoseGraphEdge] = []
        
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
        
        self.lock = threading.Lock()
    

    def _load_config(self, config_path: Optional[str]):
        """Load configuration parameters with academic defaults."""
        # Defaults
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
            
            lc = config.get('loop_closure', {})
            self.min_keyframe_gap = lc.get('min_keyframe_gap', self.min_keyframe_gap)
            self.search_radius = lc.get('search_radius', self.search_radius)
            self.bow_similarity_threshold = lc.get('bow_similarity_threshold', self.bow_similarity_threshold)
            self.min_inliers = lc.get('min_inliers', self.min_inliers)
            self.icp_fitness_threshold = lc.get('icp_fitness_threshold', self.icp_fitness_threshold)
            self.icp_rmse_threshold = lc.get('icp_rmse_threshold', self.icp_rmse_threshold)
            
            opt = config.get('optimization', {})
            self.optimize_every_n_keyframes = opt.get('optimize_every_n_keyframes', self.optimize_every_n_keyframes)
            
            odom = config.get('odometry', {})
            self.icp_threshold = odom.get('icp_max_correspondence_distance', self.icp_threshold)
            self.voxel_size = odom.get('voxel_size', self.voxel_size)
            
            cam = config.get('camera', {})
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

    def add_keyframe(self, 
                     rgb: np.ndarray,
                     depth: np.ndarray,
                     pose: np.ndarray,
                     timestamp: float,
                     odometry_transform: Optional[np.ndarray] = None,
                     scale_factor: float = 1.0,
                     pim: Optional[object] = None) -> Tuple[int, bool]: 
        """
        Add a new keyframe with VIO support.
        """
        with self.lock:
            # 自动更新相机参数
            if self.intrinsics['width'] != depth.shape[1] or self.intrinsics['height'] != depth.shape[0]:
                self.intrinsics['height'], self.intrinsics['width'] = depth.shape[:2]

            keyframe_id = self.next_keyframe_id
            self.next_keyframe_id += 1
            self.total_keyframes += 1
            
            # 1. Feature Extraction
            t_start = time.time()
            if FEATURES_AVAILABLE:
                keypoints, descriptors = self.extractor.extract_with_depth(rgb, depth, self.intrinsics)
            else:
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                cv_kps, descriptors = self.orb.detectAndCompute(gray, None)
                keypoints = [] 
                pass 
            self.timing_stats['feature_extraction'].append(time.time() - t_start)
            
            bow_vector = None
            if self.bow is not None and self.bow_trained and descriptors is not None:
                bow_vector = self.bow.compute_bow_vector(descriptors)
                self.bow.add_to_database(keyframe_id, bow_vector)
            
            t_start = time.time()
            pointcloud = self._create_pointcloud(rgb, depth)
            self.timing_stats['pointcloud_creation'].append(time.time() - t_start)
            
            self.scale_history[keyframe_id] = scale_factor
            self.scale_tracker.record_scale(keyframe_id, scale_factor)

            # 2. Keyframe Creation
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
            
            # 3. VIO / Graph Insertion (GTSAM Core)
            # 优先使用 GTSAM，如果 GTSAM 健康且可用
            if GTSAM_AVAILABLE and self.isam is not None and self.isam_healthy:
                try:
                    self._update_gtsam_graph(kf, pim, odometry_transform)
                    # Sync Pose back to Keyframe
                    kf.pose = self._get_gtsam_pose(keyframe_id)
                except Exception as e:
                    print(f"⚠️ [Backend] GTSAM Update Failed: {e}")
                    # GTSAM 失败，回退：尝试清空状态以恢复健康，本次只做纯视觉记录
                    self.graph.resize(0)
                    self.initial_estimate.clear()
            
            # Fallback / Dual Graph Maintenance: Always update Open3D graph for visualization/backup
            self._add_node(keyframe_id, kf.pose) # Use the (potentially GTSAM-optimized) pose
            if keyframe_id > 0 and odometry_transform is not None:
                prev_pcd = self.keyframes[keyframe_id - 1].pointcloud
                # Recalculate or reuse odometry info
                info = self._compute_odometry_information(prev_pcd, pointcloud, odometry_transform)
                self._add_edge(keyframe_id - 1, keyframe_id, odometry_transform, info, False)

            self.keyframes[keyframe_id] = kf
            
            # 4. Loop Closure Detection
            t_start = time.time()
            loop_detected = self._process_loop_closure(kf, rgb, pose, descriptors, keypoints)
            self.timing_stats['loop_detection'].append(time.time() - t_start)
            
            # 5. Local BA (Open3D-based)
            # 保留你的原始 Local BA 逻辑，作为 VIO 的补充或验证
            if not self.ba_initialized:
                self.local_ba.set_intrinsics(self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy'])
                self.ba_initialized = True
            
            self.local_ba.add_keyframe(keyframe_id, kf.pose, fixed=(keyframe_id==0))
            for i, kp in enumerate(kf.keypoints):
                if hasattr(kp, 'point_3d') and kp.point_3d is not None:
                    pid = keyframe_id * 10000 + i 
                    self.local_ba.add_map_point(pid, kp.point_3d)
                    if hasattr(kp, 'pt'):
                        self.local_ba.add_observation(keyframe_id, pid, kp.pt)

            # Local Optimization (Trigger existing Open3D logic)
            # 虽然 GTSAM 已经做了优化，但这里保留你的 _local_sliding_window_optimize 逻辑
            # 为了防止冲突，我们可以只更新 Open3D graph 结构，或者作为 visualization 的 refine
            kf_since_opt = self.total_keyframes - self.last_optimization_kf_count
            if kf_since_opt >= self.optimize_every_n_keyframes:
                self._local_sliding_window_optimize()
                self.last_optimization_kf_count = self.total_keyframes

            return keyframe_id, loop_detected

    def _update_gtsam_graph(self, kf: Keyframe, pim, odometry_transform):
        """核心 VIO 更新逻辑"""
        import gtsam
        
        # Prepare data
        gtsam_pose = gtsam.Pose3(gtsam.Rot3(kf.pose[:3, :3]), gtsam.Point3(kf.pose[:3, 3]))
        kf_id = kf.id
        
        # === A. 初始化 (第一帧) ===
        if not self.vio_initialized:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
            self.graph.add(gtsam.PriorFactorPose3(X(kf_id), gtsam_pose, prior_noise))
            self.graph.add(gtsam.PriorFactorVector(V(kf_id), np.zeros(3), self.vel_noise))
            
            prev_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
            self.graph.add(gtsam.PriorFactorConstantBias(B(kf_id), prev_bias, self.bias_noise))
            
            self.initial_estimate.insert(X(kf_id), gtsam_pose)
            self.initial_estimate.insert(V(kf_id), np.zeros(3))
            self.initial_estimate.insert(B(kf_id), prev_bias)
            
            self.vio_initialized = True
            print(f"✨ [Backend] VIO Initialized at KF {kf_id}")
            
        # === B. 增量更新 (后续帧) ===
        else:
            prev_id = kf_id - 1
            
            if pim is not None:
                # Tight Coupling
                imu_factor = gtsam.ImuFactor(X(prev_id), V(prev_id), X(kf_id), V(kf_id), B(prev_id), pim)
                self.graph.add(imu_factor)
                
                self.graph.add(gtsam.BetweenFactorConstantBias(
                    B(prev_id), B(kf_id), gtsam.imuBias.ConstantBias(),
                    gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)
                ))
                
                # Predict
                if self.isam.valueExists(X(prev_id)):
                    prev_pose = self.isam.calculateEstimatePose3(X(prev_id))
                    prev_vel = self.isam.calculateEstimateVector(V(prev_id))
                    prev_bias = self.isam.calculateEstimateConstantBias(B(prev_id))
                    nav_state = gtsam.NavState(prev_pose, prev_vel)
                    predicted_nav = pim.predict(nav_state, prev_bias)
                    
                    self.initial_estimate.insert(X(kf_id), predicted_nav.pose())
                    self.initial_estimate.insert(V(kf_id), predicted_nav.velocity())
                    self.initial_estimate.insert(B(kf_id), prev_bias)
                else:
                    # Fallback if prev_id missing (rare)
                    self.initial_estimate.insert(X(kf_id), gtsam_pose)
                    self.initial_estimate.insert(V(kf_id), np.zeros(3))
                    self.initial_estimate.insert(B(kf_id), gtsam.imuBias.ConstantBias())
                
            elif odometry_transform is not None:
                # Loose Coupling Fallback
                rel_pose = gtsam.Pose3(gtsam.Rot3(odometry_transform[:3,:3]), gtsam.Point3(odometry_transform[:3,3]))
                self.graph.add(gtsam.BetweenFactorPose3(X(prev_id), X(kf_id), rel_pose, self.vo_noise))
                
                if self.isam.valueExists(X(prev_id)):
                    prev_pose = self.isam.calculateEstimatePose3(X(prev_id))
                    self.initial_estimate.insert(X(kf_id), prev_pose.compose(rel_pose))
                    # Propagate states
                    if self.isam.valueExists(V(prev_id)):
                        self.initial_estimate.insert(V(kf_id), self.isam.calculateEstimateVector(V(prev_id)))
                        self.initial_estimate.insert(B(kf_id), self.isam.calculateEstimateConstantBias(B(prev_id)))
                    else:
                        self.initial_estimate.insert(V(kf_id), np.zeros(3))
                        self.initial_estimate.insert(B(kf_id), gtsam.imuBias.ConstantBias())
                else:
                    self.initial_estimate.insert(X(kf_id), gtsam_pose)
                    self.initial_estimate.insert(V(kf_id), np.zeros(3))
                    self.initial_estimate.insert(B(kf_id), gtsam.imuBias.ConstantBias())

        # === C. 执行优化 ===
        # 使用 try-finally 确保 graph 始终被清理
        try:
            self.isam.update(self.graph, self.initial_estimate)
            self.isam.update()
        finally:
            self.graph.resize(0)
            self.initial_estimate.clear()

    def _get_gtsam_pose(self, kf_id):
        """Retrieve optimized pose from ISAM2"""
        if self.isam.valueExists(X(kf_id)):
            return self.isam.calculateEstimatePose3(X(kf_id)).matrix()
        return np.eye(4)

    def _process_loop_closure(self, kf, rgb, pose, descriptors, keypoints):
        """处理回环检测并添加到 GTSAM"""
        loop_detected = False
        position = pose[:3, 3]
        
        if kf.id >= self.min_keyframe_gap and self.use_enhanced_lc and self.loop_detector is not None:
            self.loop_detector.add_keyframe(kf.id, rgb, position, descriptors)
            candidates = self.loop_detector.detect_loop_closure(kf.id, rgb, position, descriptors, keypoints, self.keyframes)
            
            for candidate in candidates:
                if candidate.is_verified:
                    if self._verify_loop_with_icp(kf, candidate):
                        # Add to GTSAM
                        if GTSAM_AVAILABLE and self.isam is not None and self.isam_healthy:
                            try:
                                rel_pose = candidate.relative_pose 
                                gtsam_rel = gtsam.Pose3(gtsam.Rot3(rel_pose[:3,:3]), gtsam.Point3(rel_pose[:3,3]))
                                noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6)) 
                                self.graph.add(gtsam.BetweenFactorPose3(X(candidate.match_id), X(kf.id), gtsam_rel, noise))
                                
                                self.isam.update(self.graph, self.initial_estimate)
                                self.isam.update()
                                kf.pose = self._get_gtsam_pose(kf.id)
                            except:
                                pass # GTSAM update failed silently, proceed
                            finally:
                                self.graph.resize(0)
                                self.initial_estimate.clear()

                        # Add to Open3D Graph (Legacy)
                        self._add_loop_closure_edge(candidate)

                        loop_detected = True
                        self.total_loop_closures += 1
                        print(f"🔄 Loop closed: {kf.id} ↔ {candidate.match_id}")
                        
                        simple_scale_correction(
                            self.keyframes, kf.id, candidate.match_id,
                            self.scale_history.get(kf.id, 1.0),
                            self.scale_history.get(candidate.match_id, 1.0)
                        )
        return loop_detected

    def _local_sliding_window_optimize(self):
        """
        Perform local sliding window optimization (Legacy Open3D).
        Kept for visualization smoothing and fallback.
        """
        n = len(self.keyframes)
        if n < 3: return False
        
        window_size = min(self.sliding_window_size, n)
        start_id = n - window_size
        
        local_graph = o3d.pipelines.registration.PoseGraph()
        for i in range(window_size):
            gid = start_id + i
            if gid in self.keyframes:
                local_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.keyframes[gid].pose.copy()))
        
        if len(local_graph.nodes) > 0:
            local_graph.nodes[0].pose = self.keyframes[start_id].pose.copy() # Fix start
            
        for edge in self.edges:
            if start_id <= edge.source_id < n and start_id <= edge.target_id < n:
                local_src = edge.source_id - start_id
                local_tgt = edge.target_id - start_id
                if 0 <= local_src < window_size and 0 <= local_tgt < window_size:
                    local_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        local_src, local_tgt, edge.transformation.copy(), edge.information.copy(), uncertain=edge.is_loop_closure
                    ))
        
        if len(local_graph.edges) == 0: return False
        
        try:
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=self.icp_threshold,
                edge_prune_threshold=0.25,
                preference_loop_closure=1.0,
                reference_node=0
            )
            o3d.pipelines.registration.global_optimization(
                local_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option
            )
            
            # Sync back to keyframes (Optional: disable if trusting GTSAM more)
            # 这里我们选择相信 GTSAM，如果不启用 GTSAM 才回写
            if not (GTSAM_AVAILABLE and self.isam_healthy):
                for i, node in enumerate(local_graph.nodes):
                    gid = start_id + i
                    if gid in self.keyframes:
                        self.keyframes[gid].pose = node.pose.copy()
                    if gid in self.nodes:
                        self.nodes[gid].pose = node.pose.copy()
                    if gid < len(self.pose_graph.nodes):
                        self.pose_graph.nodes[gid].pose = node.pose.copy()
            return True
        except Exception:
            return False

    def _add_node(self, keyframe_id: int, pose: np.ndarray):
        """Add a node to the pose graph."""
        self.nodes[keyframe_id] = PoseGraphNode(pose, keyframe_id)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose.copy()))

    def _add_edge(self, source_id: int, target_id: int,
                  transformation: np.ndarray, information: np.ndarray, is_loop_closure: bool):
        """Add an edge to the pose graph."""
        self.edges.append(PoseGraphEdge(source_id, target_id, transformation, information, is_loop_closure))
        self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            source_id, target_id, transformation, information, uncertain=is_loop_closure
        ))

    def _add_loop_closure_edge(self, candidate: LoopCandidate):
        if candidate.relative_pose is not None:
            info = candidate.information_matrix if candidate.information_matrix is not None else np.eye(6)
            self._add_edge(candidate.query_id, candidate.match_id, candidate.relative_pose, info, is_loop_closure=True)
            self.detected_loops.append(candidate)

    def _add_loop_closure_edge_legacy(self, candidate: LoopClosureCandidateLegacy):
        if candidate.is_verified and candidate.relative_pose is not None:
            info = candidate.information if candidate.information is not None else np.eye(6)
            self._add_edge(candidate.query_id, candidate.match_id, candidate.relative_pose, info, is_loop_closure=True)
            self.detected_loops.append(candidate)

    def _compute_odometry_information(self, source_pcd, target_pcd, transformation):
        if source_pcd is None or target_pcd is None: return np.eye(6)
        if len(source_pcd.points) < 10 or len(target_pcd.points) < 10: return np.eye(6)
        try:
            return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_pcd, target_pcd, self.icp_threshold, transformation
            )
        except: return np.eye(6)

    def optimize(self) -> bool:
        """
        Global optimization trigger.
        Ensures final poses are synced from GTSAM and applies Sim3 if needed.
        """
        # 1. Sync GTSAM poses
        if GTSAM_AVAILABLE and self.isam is not None and self.isam_healthy:
            try:
                for kf_id, kf in self.keyframes.items():
                    if self.isam.valueExists(X(kf_id)):
                        pose_gtsam = self.isam.calculateEstimatePose3(X(kf_id))
                        kf.pose = pose_gtsam.matrix()
                        # Sync Open3D graph for consistency
                        if kf_id < len(self.pose_graph.nodes):
                            self.pose_graph.nodes[kf_id].pose = kf.pose
            except Exception as e:
                print(f"⚠️ [Backend] Sync Failed: {e}")

        # 2. Sim3 Optimization
        if SIM3_AVAILABLE and self.total_loop_closures > 0:
            print(f"📐 Global Optimization (Sim3 Academic Method)...")
            try:
                corrected_poses, scale_corrections = correct_map_after_loop_closure(
                    self.keyframes, self.pose_graph, getattr(self, 'scale_history', {})
                )
                from sim3_pose_graph import apply_scale_corrections_to_pointclouds
                apply_scale_corrections_to_pointclouds(self.keyframes, corrected_poses, scale_corrections)
                
                # Apply corrections
                for i, pose in corrected_poses.items():
                    if i in self.keyframes: self.keyframes[i].pose = pose.copy()
                    if i < len(self.pose_graph.nodes): self.pose_graph.nodes[i].pose = pose.copy()
                
                print("✅ Sim3 Optimization complete")
                return True
            except Exception as e:
                print(f"⚠️ Sim3 Optimization failed: {e}")
        
        # 3. Fallback Open3D Global Optimization (only if GTSAM absent)
        if not (GTSAM_AVAILABLE and self.isam_healthy) and len(self.pose_graph.nodes) > 2:
             print(f"📐 Global Optimization (Open3D Fallback)...")
             try:
                 option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=self.icp_threshold, edge_prune_threshold=0.25, preference_loop_closure=1.0, reference_node=0)
                 o3d.pipelines.registration.global_optimization(
                    self.pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
                 # Update poses
                 for i, node in enumerate(self.pose_graph.nodes):
                     if i in self.keyframes: self.keyframes[i].pose = node.pose.copy()
             except Exception as e:
                 print(f"⚠️ Optimization failed: {e}")

        print("✅ Optimization complete")
        return True

    def train_vocabulary(self, descriptors_list: List[np.ndarray]):
        self.bow.train_vocabulary(descriptors_list)
        self.bow_trained = True

    def _create_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
        """Create point cloud from RGB-D data."""
        h, w = depth.shape[:2]
        depth_min = self.intrinsics.get('depth_min', 0.2)
        depth_max = self.intrinsics.get('depth_max', 5.0)
        mask = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
        if np.sum(mask) < 50: return o3d.geometry.PointCloud()
        
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_filtered = depth.copy().astype(np.float32); depth_filtered[~mask] = 0
        depth_o3d = o3d.geometry.Image(depth_filtered)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, 1.0, depth_max, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy'])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        if self.voxel_size > 0 and len(pcd.points) > 100: pcd = pcd.voxel_down_sample(self.voxel_size)
        return pcd

    def _compute_confidence_map(self, kf, height, width):
        """[学术核心] 构建 2D 特征置信度场"""
        confidence_map = np.full((height, width), 0.1, dtype=np.float32)
        if kf.keypoints and len(kf.keypoints) > 0:
            coords = []
            scores = []
            for kp in kf.keypoints:
                pt = getattr(kp, 'pt', None)
                if pt is None and hasattr(kp, 'point_3d'): continue
                response = getattr(kp, 'response', 1.0)
                coords.append([int(pt[1]), int(pt[0])])
                scores.append(response)
            if coords:
                coords = np.array(coords)
                scores = np.array(scores)
                mask = np.zeros((height, width), dtype=np.float32)
                valid_y = np.clip(coords[:, 0], 1, height-2)
                valid_x = np.clip(coords[:, 1], 1, width-2)
                mask[valid_y, valid_x] = scores
                mask[valid_y+1, valid_x] = scores * 0.8; mask[valid_y-1, valid_x] = scores * 0.8
                mask[valid_y, valid_x+1] = scores * 0.8; mask[valid_y, valid_x-1] = scores * 0.8
                confidence_map = np.maximum(confidence_map, mask)
        return confidence_map

    def _check_geometric_consistency(self, curr_pcd_world, curr_kf_id, check_window=2, dist_thres=0.05):
        """[学术核心] 几何一致性检查"""
        if len(curr_pcd_world.points) == 0: return []
        neighbor_pcds = []
        for i in range(1, check_window + 1):
            prev_id = curr_kf_id - i
            if prev_id in self.keyframes:
                kf_prev = self.keyframes[prev_id]
                if kf_prev.pointcloud is not None and len(kf_prev.pointcloud.points) > 0:
                    pcd_prev = copy.deepcopy(kf_prev.pointcloud)
                    pcd_prev.transform(kf_prev.pose)
                    neighbor_pcds.append(pcd_prev)
        if not neighbor_pcds: return list(range(len(curr_pcd_world.points)))
        valid_mask = np.zeros(len(curr_pcd_world.points), dtype=bool)
        for nb_pcd in neighbor_pcds:
            dists = np.asarray(curr_pcd_world.compute_point_cloud_distance(nb_pcd))
            valid_mask |= (dists < dist_thres)
        return np.where(valid_mask)[0]

    def build_global_map(self, filter_noise: bool = True, strict_mode: bool = True) -> o3d.geometry.PointCloud:
        """构建全局地图 (保留所有清洗逻辑)"""
        print(f"\n🗺️  正在构建高精度全局地图 (Strict Mode: {strict_mode})...")
        global_map = o3d.geometry.PointCloud()
        sorted_ids = sorted(self.keyframes.keys())
        total_points_raw = 0; total_points_kept = 0
        
        for i, kf_id in enumerate(sorted_ids):
            kf = self.keyframes[kf_id]
            if kf.pointcloud is None or len(kf.pointcloud.points) == 0: continue
            
            pcd_world = copy.deepcopy(kf.pointcloud)
            pcd_world.transform(kf.pose)
            n_raw = len(pcd_world.points)
            total_points_raw += n_raw
            
            if strict_mode and i > 0:
                valid_idx = self._check_geometric_consistency(pcd_world, kf_id, check_window=3, dist_thres=0.08)
                if len(valid_idx) < 10: continue
                pcd_world = pcd_world.select_by_index(valid_idx)
            
            global_map += pcd_world
            total_points_kept += len(pcd_world.points)
            if i % 10 == 0: print(f"   处理帧 {i}/{len(sorted_ids)} | 保留率: {len(pcd_world.points)/n_raw*100:.1f}%")
            if i % 50 == 0: global_map = global_map.voxel_down_sample(self.voxel_size)

        print(f"   📉 初始清洗完成: {total_points_raw} -> {total_points_kept} 点")
        if filter_noise and len(global_map.points) > 0:
            print("   🧹 执行最终统计滤波 (SOR)...")
            cl, ind = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            global_map = global_map.select_by_index(ind)
            cl, ind = global_map.remove_radius_outlier(nb_points=10, radius=0.05)
            global_map = global_map.select_by_index(ind)
        print(f"   ✅ 地图构建完成! 最终点数: {len(global_map.points)}")
        return global_map

    def save_trajectory(self, filepath: str, format: str = 'tum'):
        """Save trajectory (Updated to accept format argument)."""
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
        """Save keyframe data."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Saving {len(self.keyframes)} keyframes...")
        for kf_id, kf in self.keyframes.items():
            if kf.rgb_image is not None:
                cv2.imwrite(os.path.join(output_dir, f"frame_{kf_id:06d}.jpg"), kf.rgb_image)
            with open(os.path.join(output_dir, f"frame_{kf_id:06d}.json"), 'w') as f:
                json.dump({'id': kf_id, 'timestamp': kf.timestamp, 'pose': kf.pose.flatten().tolist()}, f)
        print("✅ Keyframes saved")

    def get_statistics(self) -> Dict:
        """Get optimizer statistics."""
        def safe_mean(lst): return np.mean(lst[-100:]) if lst else 0.0
        stats = {
            'num_keyframes': len(self.keyframes), 'num_nodes': len(self.nodes),
            'num_edges': len(self.edges), 'num_loops': self.total_loop_closures,
            'avg_feature_extraction_ms': safe_mean(self.timing_stats['feature_extraction']) * 1000,
            'avg_pointcloud_creation_ms': safe_mean(self.timing_stats['pointcloud_creation']) * 1000,
            'avg_loop_detection_ms': safe_mean(self.timing_stats['loop_detection']) * 1000,
            'avg_optimization_ms': safe_mean(self.timing_stats['optimization']) * 1000,
        }
        if self.use_enhanced_lc and self.loop_detector is not None:
            stats.update({'lc_' + k: v for k, v in self.loop_detector.get_statistics().items()})
        return stats

    # pose_graph_enhanced.py
    def get_latest_bias(self):
        if not GTSAM_AVAILABLE or self.isam is None or not self.isam_healthy:
            return None
        if len(self.keyframes) == 0:
            return None

        latest_id = max(self.keyframes.keys())
        try:
            if self.isam.valueExists(B(latest_id)):
                return self.isam.calculateEstimateConstantBias(B(latest_id))
        except Exception:
            pass
        return None

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        print("\n" + "="*60 + "\n📊 PoseGraphOptimizer Statistics\n" + "="*60)
        print(f"  Keyframes:         {stats['num_keyframes']}")
        print(f"  Loop closures:     {stats['num_loops']}")
        print("-"*60)
        print(f"  Feature extraction: {stats['avg_feature_extraction_ms']:.1f} ms")
        print(f"  Optimization:       {stats['avg_optimization_ms']:.1f} ms")
        print("="*60 + "\n")

    # Legacy placeholders just in case
    def _add_node_legacy(self, kid, p): self._add_node(kid, p)
    def _add_edge_legacy(self, s, t, tr, inf, l): self._add_edge(s, t, tr, inf, l)

def create_pose_graph_optimizer(config_path=None, use_enhanced_lc=True, hef_path=None, onnx_path=None, vdevice=None):
    return PoseGraphOptimizer(config_path, use_enhanced_lc, hef_path, onnx_path, vdevice)