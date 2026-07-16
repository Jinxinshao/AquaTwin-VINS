#!/usr/bin/env python3
"""
================================================================================
PiSLAM Enhanced Visual Odometry (visual_odometry_enhanced.py)
================================================================================

增强版视觉里程计模块 (Academic Edition v2.2 - IMU Fusion Ready)
集成 XFeat 极速深度特征前端，支持多模态特征提取与自适应几何求解。
新增：IMU 紧耦合初始猜测 (Loose-Coupled IMU Initialization)

System Architecture:
--------------------
1. Primary Frontend (XFeat): CVPR 2024 SOTA 轻量级特征提取器。
2. Geometric Verification: Adaptive PnP (P3P/EPnP/Iterative) + RANSAC。
3. Motion Prediction: 
   - IMU Rotation Integration (Primary)
   - Constant Velocity Model (Secondary/Fallback)

Author: PiSLAM Research Team
================================================================================
"""

import numpy as np
import cv2
import open3d as o3d
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Union
import yaml
import time
import os
import sys

# 引入 Scipy 处理四元数 (关键新增)
from scipy.spatial.transform import Rotation as R
import torch
# 在文件顶部
from scale_fusion_bayesian import WeightedPnPSolver

# =============================================================================
# 模块导入与环境检查
# =============================================================================

from modules.xfeat import XFeat
# 尝试导入 LighterGlue
try:
    from modules.lighterglue import LighterGlue
    HAS_LIGHTERGLUE = True
except ImportError:
    HAS_LIGHTERGLUE = False

HAS_XFEAT = False
HAS_ONNX = True

# 尝试导入 ONNXRuntime (备选方案)
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    pass

# 导入增强特征模块
from features_enhanced import (
    EnhancedFeatureManager, 
    EnhancedKeyPoint, 
    EnhancedMatch
)
from scale_corrector import DepthCorrector


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class EnhancedOdometryResult:
    """
    增强版里程计结果数据类
    """
    success: bool
    pose: np.ndarray
    # 统计指标
    inliers: int = 0
    fitness: float = 0.0
    rmse: float = float('inf')
    num_features: int = 0
    processing_time: float = 0.0
    # 详细诊断信息
    num_matches: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = float('inf')
    depth_consistency_ratio: float = 0.0
    method_used: str = "none"
    feature_time: float = 0.0
    matching_time: float = 0.0
    pnp_time: float = 0.0
    # 透传特征点用于可视化
    current_keypoints: Optional[np.ndarray] = None
    # 🟢 [新增] 特征追踪长度 (用于保守建图筛选)
    track_lengths: Optional[np.ndarray] = None


# =============================================================================
# 辅助类：ONNX 前端封装 (Legacy Support)
# =============================================================================

class SuperPointFrontend:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_tensor = (img.astype(np.float32) / 255.0)[None, None, :, :]
        outs = self.session.run(None, {self.input_name: img_tensor})
        return outs[0][0], outs[2][0], outs[1][0] # kpts, descs, scores

class SuperGlueMatcher:
    def __init__(self, model_path: str, match_threshold: float = 0.2):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.match_threshold = match_threshold
        self.input_names = [i.name for i in self.session.get_inputs()]

    def match(self, kpts0, desc0, scores0, kpts1, desc1, scores1):
        if len(kpts0) == 0 or len(kpts1) == 0: return [], []
        inputs = {
            self.input_names[0]: kpts0[None, :, :], self.input_names[1]: desc0[None, :, :],
            self.input_names[2]: scores0[None, :],   self.input_names[3]: kpts1[None, :, :],
            self.input_names[4]: desc1[None, :, :],   self.input_names[5]: scores1[None, :]
        }
        outs = self.session.run(None, inputs)
        return outs[0][0], outs[1][0]


# =============================================================================
# 核心类：自适应 PnP 求解器
# =============================================================================

class AdaptivePnPSolver:
    """
    自适应 PnP 求解器 (Robust Geometric Verification)
    """
    def __init__(self, config: Optional[Dict] = None):
        self.ransac_iterations = 1000
        self.ransac_reproj_error = 2.0  # 像素阈值
        self.ransac_confidence = 0.999
        self.use_refinement = True
        self.min_points_for_epnp = 30
        
        if config:
            self.ransac_iterations = config.get('ransac_iterations', self.ransac_iterations)
            self.ransac_reproj_error = config.get('ransac_reprojection_error', self.ransac_reproj_error)

    def solve(self, object_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray,
              initial_rvec: Optional[np.ndarray] = None, initial_tvec: Optional[np.ndarray] = None):
        
        n_points = len(object_points)
        info = {'method': 'none', 'iterations': 0}
        
        if n_points < 4:
            return False, None, None, None, info
            
        object_points = np.ascontiguousarray(object_points, dtype=np.float64)
        image_points = np.ascontiguousarray(image_points, dtype=np.float64)
        
        # 策略选择
        if n_points < self.min_points_for_epnp:
            pnp_method = cv2.SOLVEPNP_P3P
            info['method'] = 'P3P'
        else:
            pnp_method = cv2.SOLVEPNP_EPNP
            info['method'] = 'EPnP'
            
        use_guess = initial_rvec is not None and initial_tvec is not None
        
        # ⚠️ 关键：如果没有猜测值，必须传入零向量，否则 solvePnPRansac 会报错
        rvec_init = initial_rvec if use_guess else np.zeros((3,1), dtype=np.float64)
        tvec_init = initial_tvec if use_guess else np.zeros((3,1), dtype=np.float64)
        
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, camera_matrix, None,
                rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=use_guess,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.ransac_reproj_error,
                confidence=self.ransac_confidence,
                flags=pnp_method
            )
        except Exception as e:
            print(f"⚠️ [PnP] Solver Exception: {e}")
            return False, None, None, None, info
            
        if not success or inliers is None:
            return False, None, None, None, info
            
        inliers = inliers.flatten()
        info['mean_reproj_error'] = 0.0 
        
        # 迭代优化
        if self.use_refinement and len(inliers) >= 6:
            try:
                success_ref, rvec, tvec = cv2.solvePnP(
                    object_points[inliers], image_points[inliers], camera_matrix, None,
                    rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success_ref: info['refined'] = True
            except: pass
            
        return True, rvec, tvec, inliers, info


# =============================================================================
# 主类：增强版视觉里程计 (Multi-Backend)
# =============================================================================

class EnhancedVisualOdometry:
    """
    增强版视觉里程计 (Academic Edition)
    支持 IMU 辅助与 XFeat/LighterGlue 前端
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._load_config(config_path)
        print("\n" + "="*60)
        print("🚀 Initializing Enhanced Visual Odometry Pipeline (IMU Ready)")
        print("="*60)

        # 1. Frontend Initialization Strategy
        self.frontend_mode = "none"
        self.xfeat = None
        self.sp_frontend = None

        # [新增] 记录上一帧的 IMU 状态
        self.prev_imu_quat = None # [w, x, y, z]
        self.use_imu_prediction = True
        # 🟢 [新增] 上一帧特征点的追踪长度缓存
        self.prev_track_lengths = None
        
        # # 策略 A: XFeat + LighterGlue
        # xfeat_weights = "./weights/xfeat-lighterglue.pt"

        # if HAS_XFEAT and os.path.exists(xfeat_weights) and HAS_LIGHTERGLUE:
        #     print(f"🔄 Loading XFeat from {xfeat_weights}...")
        #     self.xfeat = XFeat(weights_path=xfeat_weights, top_k=2000)
            
        #     print(f"🔄 Loading LighterGlue...")
        #     self.matcher = LighterGlue() 
        #     self.matcher.eval()

        #     self.frontend_mode = "xfeat_lg"
        #     print("✅ [VO] Mode Active: XFeat + LighterGlue (Deep Matching)")
            
        #     # Tensor 缓存
        #     self.prev_feats_tensor = None

        # # 策略 A: XFeat + BFMatcher (Modified: Remove LighterGlue)
        # xfeat_weights = "./weights/xfeat-lighterglue.pt"

        # # 🟢 [修改] 即使有 LighterGlue 也不加载，强制使用 XFeat + BFMatcher
        # # if HAS_XFEAT and os.path.exists(xfeat_weights) and HAS_LIGHTERGLUE:
        # if HAS_XFEAT and os.path.exists(xfeat_weights):
        #     print(f"🔄 Loading XFeat from {xfeat_weights}...")
        #     self.xfeat = XFeat(weights_path=xfeat_weights, top_k=1000) # 降一点点数量提速
            
        #     # print(f"🔄 Loading LighterGlue...")
        #     # self.matcher = LighterGlue() 
        #     # self.matcher.eval()
        #     self.matcher = None # 禁用深度匹配器

        #     self.frontend_mode = "xfeat_lg" # 保持模式名称不变，但在处理函数里改逻辑
        #     print("✅ [VO] Mode Active: XFeat + BFMatcher (High Speed)")
            
        #     # Tensor 缓存 (不再需要)
        #     self.prev_feats_tensor = None

        

        # 策略 B: SuperPoint + SuperGlue
        if self.frontend_mode == "none" and HAS_ONNX:
            sp_path = "superpoint.onnx"
            sg_path = "superglue.onnx"
            if os.path.exists(sp_path) and os.path.exists(sg_path):
                try:
                    self.sp_frontend = SuperPointFrontend(sp_path)
                    self.sg_matcher = SuperGlueMatcher(sg_path)
                    self.frontend_mode = "onnx"
                    print("✅ [VO] Mode Active: SuperPoint + SuperGlue (ONNX)")
                except Exception as e:
                    print(f"❌ [VO] ONNX Init Failed: {e}")

        # 策略 C: Classic ORB
        self.feature_manager = EnhancedFeatureManager(config_path)
        if self.frontend_mode == "none":
            self.frontend_mode = "classic"
            print("⚠️ [VO] Mode Active: Classic ORB (Fallback)")

        # # 2. Backend Initialization
        # self.pnp_solver = AdaptivePnPSolver({
        #     'ransac_iterations': self.ransac_iterations,
        #     'ransac_reprojection_error': self.ransac_reproj_error
        # })

        # 初始化加权 PnP 求解器
        self.pnp_solver = WeightedPnPSolver(
            ransac_iterations=self.ransac_iterations,
            reproj_threshold=self.ransac_reproj_error,
            depth_sigma_coeff=0.02 # 深度误差系数，可调
        )
        print("✅ [VO] Using Weighted PnP Solver (Uncertainty-Aware)")
        
        self.motion_model = MotionModel(decay=self.motion_decay)
        
        # 深度校正
        if self.enable_depth_correction:
            try:
                self.depth_corrector = DepthCorrector("depth_anything_v2_vits.onnx", target_interval=10.0)
                print("✅ [VO] Depth Corrector Initialized (Interval: 10.0s)")
            except:
                self.depth_corrector = None
                print("⚠️ [VO] Depth Corrector Failed (Optional)")
        else:
            self.depth_corrector = None

        # State Variables
        self.current_pose = np.eye(4)
        self.is_initialized = False
        self.tracking_lost = False
        self.frame_count = 0
        
        # Cache
        self.prev_rgb = None
        self.prev_depth = None
        self.prev_kpts = None
        self.prev_descs = None
        self.prev_scores = None
        self.prev_kpts_3d = [] # List[EnhancedKeyPoint]
        self.camera_matrix = None

    def _load_config(self, config_path):
        self.intrinsics = {'fx': 500.0, 'fy': 500.0, 'cx': 320.0, 'cy': 240.0, 'depth_min': 0.1, 'depth_max': 10.0}
        self.ransac_iterations = 1000
        self.ransac_reproj_error = 2.0
        self.motion_decay = 0.8
        self.use_motion_model = True
        self.min_inliers = 12
        self.enable_depth_correction = True
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                cam = cfg.get('camera', {})
                self.intrinsics.update(cam)
                odom = cfg.get('odometry', {})
                self.ransac_iterations = odom.get('ransac_iterations', self.ransac_iterations)
                self.use_motion_model = odom.get('use_motion_model', self.use_motion_model)
            except: pass

    def set_intrinsics(self, fx, fy, cx, cy):
        self.intrinsics.update({'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # 🟢 [关键修改]：增加 imu_orientation 参数
    def process_frame(self, rgb: np.ndarray, depth: np.ndarray, 
                      imu_orientation: Optional[np.ndarray] = None) -> EnhancedOdometryResult:
        """Main Processing Loop with IMU Fusion"""
        start_time = time.time()
        self.frame_count += 1
        
        # 1. Preprocessing
        h, w = rgb.shape[:2]  # 获取当前实际输入的分辨率 (例如 640x480)

        # =====================================================================
        # 🟢 [修复] 内参自动适配 (Auto-Scaling Intrinsics)
        # =====================================================================
        # 如果配置文件里的 cx (光心) 偏离了当前图像中心太远 (超过 20%)
        # 说明内参是针对其他分辨率配置的，需要自动缩放

        if abs(self.intrinsics['cx'] - w / 2) > w * 0.2:
            print(f"⚠️ [VO] Intrinsics mismatch detected! Config: {self.intrinsics['width']}x{self.intrinsics['height']}, Input: {w}x{h}")
            
            # 计算缩放比例：将光心强行对齐到当前图像中心
            scale_x = (w / 2.0) / self.intrinsics['cx']
            scale_y = (h / 2.0) / self.intrinsics['cy']
            
            # 应用缩放
            print(f"   🔄 Auto-scaling intrinsics by factor: {scale_x:.2f}")
            self.intrinsics['fx'] *= scale_x
            self.intrinsics['fy'] *= scale_y
            self.intrinsics['cx'] *= scale_x
            self.intrinsics['cy'] *= scale_y
            
            # 更新 width/height 记录
            self.intrinsics['width'] = w
            self.intrinsics['height'] = h
            
            # 更新内部矩阵
            self.set_intrinsics(self.intrinsics['fx'], self.intrinsics['fy'], 
                                self.intrinsics['cx'], self.intrinsics['cy'])
        # =====================================================================

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if len(rgb.shape)==3 else rgb.copy()
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
            
        # 2. Depth Correction
        scale = 1.0
        if self.depth_corrector:
            self.depth_corrector.update_request(rgb, depth)
            scale = self.depth_corrector.get_scale_factor()
        corrected_depth = depth * scale

        # =====================================================================
        # 🟢 [新增]：IMU 旋转预测 (Rotation Prediction)
        # =====================================================================
        imu_pred_rot_matrix = None # 这是用来辅助 PnP 的猜测值 (R_curr_prev)

        if self.use_imu_prediction and imu_orientation is not None and self.prev_imu_quat is not None:
            try:
                # 假设输入 imu_orientation 是 [w, x, y, z] (Hamilton)
                # Scipy 需要 [x, y, z, w]
                q_curr_scipy = np.roll(imu_orientation, -1)
                q_prev_scipy = np.roll(self.prev_imu_quat, -1)
                
                rot_curr = R.from_quat(q_curr_scipy)
                rot_prev = R.from_quat(q_prev_scipy)
                
                # 计算从上一帧坐标系到当前帧坐标系的相对旋转
                # 我们假设 IMU 坐标系与 Camera 坐标系重合 (Rigid transform 忽略)
                # T_wc_k = T_wc_k-1 * T_rel
                # R_wc_k = R_wc_k-1 * R_rel
                # => R_rel = R_wc_k-1^T * R_wc_k
                # 注意：solvePnP 需要的是将 Previous Frame 的点变换到 Current Frame 的 R, t
                # 即 P_curr = R_guess * P_prev + t
                # 也就是我们需要 R_curr_prev
                # R_curr_prev = R_wc_k^T * R_wc_k-1  (World-to-Body inverse logic)
                
                # 简化理解：
                # R_prev: Body(k-1) -> World
                # R_curr: Body(k) -> World
                # R_diff = R_curr.inv() * R_prev (World消掉，剩下 Prev -> Curr)
                
                delta_R = rot_curr.inv() * rot_prev
                imu_pred_rot_matrix = delta_R.as_matrix()
                
            except Exception as e:
                print(f"⚠️ [VO] IMU Math Error: {e}")

        # 更新上一帧 IMU
        if imu_orientation is not None:
            self.prev_imu_quat = imu_orientation.copy()

        # =====================================================================
        
        # 3. Dispatch to Frontend (传入 imu_pred_rot_matrix)
        if self.frontend_mode == "xfeat_lg":
            return self._process_xfeat(rgb, gray, corrected_depth, start_time, imu_pred_rot_matrix)
        elif self.frontend_mode == "onnx":
            return self._process_onnx(rgb, gray, corrected_depth, start_time, imu_pred_rot_matrix)
        else:
            return self._process_classic(rgb, gray, corrected_depth, start_time, imu_pred_rot_matrix)

    # def _process_xfeat(self, rgb, gray, depth, start_time, r_pred_guess=None):
    #     t0 = time.time()

    #     # 1. 特征提取
    #     kpts, descs, scores = self.xfeat.detectAndCompute(gray)

    #     # LighterGlue 数据准备
    #     h, w = gray.shape
    #     kpts_norm = kpts / np.array([w, h], dtype=np.float32)
    #     feats_tensor = {
    #         'keypoints': torch.from_numpy(kpts_norm).float()[None],
    #         'descriptors': torch.from_numpy(descs).float()[None],
    #         'image_size': torch.tensor([(w, h)]).float()[None]
    #     }
    #     feat_time = time.time() - t0

    #     # 2. 初始化检查
    #     if not self.is_initialized:
    #         self.prev_feats_tensor = feats_tensor
    #         res = self._initialize_system(rgb, depth, kpts, descs, scores, start_time, "xfeat_init")
    #         res.current_keypoints = kpts
    #         return res

    #     # 3. 深度匹配
    #     t1 = time.time()
    #     data = {
    #         'keypoints0': self.prev_feats_tensor['keypoints'],
    #         'descriptors0': self.prev_feats_tensor['descriptors'],
    #         'image_size0': self.prev_feats_tensor['image_size'],
    #         'keypoints1': feats_tensor['keypoints'],
    #         'descriptors1': feats_tensor['descriptors'],
    #         'image_size1': feats_tensor['image_size']
    #     }
    #     with torch.no_grad():
    #         out = self.matcher(data)

    #     matches_idx = out['matches0'][0].cpu().numpy()
    #     scores_lg = out['matching_scores0'][0].cpu().numpy()

    #     matches = []
    #     valid_indices = np.where(matches_idx > -1)[0]
    #     for idx0 in valid_indices:
    #         idx1 = matches_idx[idx0]
    #         if scores_lg[idx0] > 0.5: # 适度降低阈值，依赖 PnP 剔除
    #             matches.append(cv2.DMatch(idx0, int(idx1), 1.0 - scores_lg[idx0]))

    #     match_time = time.time() - t1
        
    #     # 4. 3D-2D 关联
    #     object_points, image_points = [], []
    #     for m in matches:
    #         if m.queryIdx < len(self.prev_kpts_3d):
    #             p3d = self.prev_kpts_3d[m.queryIdx].point_3d
    #             if p3d is not None:
    #                 object_points.append(p3d)
    #                 image_points.append(kpts[m.trainIdx])
                    
    #     # 5. Pose Estimation (传入 IMU Guess)
    #     result = self._solve_pose(object_points, image_points, len(kpts), len(matches), 
    #                              feat_time, match_time, start_time, 
    #                              r_pred_guess=r_pred_guess) # <--- Pass guess

    #     result.current_keypoints = kpts
        
    #     if result.success:
    #         self.prev_feats_tensor = feats_tensor
    #         self._update_history(rgb, depth, kpts, descs, scores)

    #     return result

    def _process_xfeat(self, rgb, gray, depth, start_time, r_pred_guess=None):
        t0 = time.time()

        # 1. 特征提取 (XFeat 本身很快，保留)
        kpts, descs, scores = self.xfeat.detectAndCompute(gray)
        # 🟢 [新增] 默认追踪长度为 1 (新特征点)
        curr_track_lengths = np.ones(len(kpts), dtype=np.int32)
        feat_time = time.time() - t0

        # 2. 初始化检查
        if not self.is_initialized:
            # 只需要存 numpy 数组
            self._update_history(rgb, depth, kpts, descs, scores)
            self.is_initialized = True
            
            # 构造返回结果
            res = EnhancedOdometryResult(True, self.current_pose.copy(), num_features=len(kpts), 
                                      processing_time=time.time()-start_time, method_used="xfeat_init")
            res.current_keypoints = kpts
            res.track_lengths = curr_track_lengths # 返回长度
            return res

        # 3. 传统匹配 (BFMatcher L2)
        # 🟢 [核心修改] 替换 LighterGlue 为 OpenCV BFMatcher
        t1 = time.time()
        
        # XFeat 输出是浮点数描述子，必须用 NORM_L2 (ORB才用HAMMING)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = []
        if self.prev_descs is not None and descs is not None and len(self.prev_descs) > 0 and len(descs) > 0:
            try:
                matches = bf.match(self.prev_descs, descs)
            except Exception as e:
                print(f"Matcher Error: {e}")
                matches = []
        
        match_time = time.time() - t1
        
        # 4. 3D-2D 关联 (逻辑不变)
        object_points, image_points = [], []
        for m in matches:
            # 简单的距离过滤，剔除太离谱的匹配 (可选)
            # if m.distance > 0.8: continue 
            
            if m.queryIdx < len(self.prev_kpts_3d):
                p3d = self.prev_kpts_3d[m.queryIdx].point_3d
                if p3d is not None:
                    object_points.append(p3d)
                    image_points.append(kpts[m.trainIdx])
                    
        # 5. Pose Estimation (传入 IMU Guess)
        result = self._solve_pose(object_points, image_points, len(kpts), len(matches), 
                                 feat_time, match_time, start_time, 
                                 r_pred_guess=r_pred_guess)
        # 更新历史缓存
        if result.success:
            self.prev_track_lengths = curr_track_lengths # 缓存给下一帧
            self._update_history(...)

        # 返回结果带上 track_lengths
        result.track_lengths = curr_track_lengths
        result.current_keypoints = kpts
        
        if result.success:
            # 不再需要缓存 Tensor
            self._update_history(rgb, depth, kpts, descs, scores)

        return result

    def _process_onnx(self, rgb, gray, depth, start_time, r_pred_guess=None):
        t0 = time.time()
        kpts, descs, scores = self.sp_frontend.run(gray)
        feat_time = time.time() - t0
        
        if not self.is_initialized:
            return self._initialize_system(rgb, depth, kpts, descs.T, scores, start_time, "onnx_init")
            
        t1 = time.time()
        matches_idx, scores_idx = self.sg_matcher.match(
            self.prev_kpts, self.prev_descs.T, self.prev_scores,
            kpts, descs, scores
        )
        match_time = time.time() - t1
        
        valid = matches_idx > -1
        object_points, image_points = [], []
        valid_indices = np.where(valid)[0]
        for idx_prev in valid_indices:
            idx_curr = matches_idx[idx_prev]
            if idx_prev < len(self.prev_kpts_3d):
                p3d = self.prev_kpts_3d[idx_prev].point_3d
                if p3d is not None:
                    object_points.append(p3d)
                    image_points.append(kpts[idx_curr])
                    
        result = self._solve_pose(object_points, image_points, len(kpts), len(valid_indices),
                                 feat_time, match_time, start_time, r_pred_guess)
        
        if result.success:
            self._update_history(rgb, depth, kpts, descs.T, scores)
        return result

    def _process_classic(self, rgb, gray, depth, start_time, r_pred_guess=None):
        t0 = time.time()
        kpts_cv, descs = self.feature_manager.process_frame(rgb, depth, self.intrinsics, return_matches=False)
        
        if len(kpts_cv) > 0:
            kpts = np.array([k.pt for k in kpts_cv], dtype=np.float32)
            scores = np.array([k.response for k in kpts_cv], dtype=np.float32)
        else:
            kpts = np.empty((0, 2), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            descs = np.empty((0, 32), dtype=np.uint8)

        feat_time = time.time() - t0
        
        if not self.is_initialized:
            return self._initialize_system(rgb, depth, kpts, descs, scores, start_time, "classic_init")
            
        t1 = time.time()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if self.prev_descs is not None and descs is not None and len(self.prev_descs)>0 and len(descs)>0:
            matches = bf.match(self.prev_descs, descs)
        else: matches = []
        match_time = time.time() - t1
        
        object_points, image_points = [], []
        for m in matches:
            if m.queryIdx < len(self.prev_kpts_3d):
                p3d = self.prev_kpts_3d[m.queryIdx].point_3d
                if p3d is not None:
                    object_points.append(p3d)
                    image_points.append(kpts[m.trainIdx])
                    
        result = self._solve_pose(object_points, image_points, len(kpts), len(matches),
                                 feat_time, match_time, start_time, r_pred_guess)
        
        if result.success:
            self._update_history(rgb, depth, kpts, descs, scores)
        return result

    # =========================================================================
    # Common Core Logic
    # =========================================================================
    
    def _initialize_system(self, rgb, depth, kpts, descs, scores, start_time, method):
        self._update_history(rgb, depth, kpts, descs, scores)
        self.is_initialized = True
        print(f"✅ [VO] System Initialized via {method} with {len(kpts)} features")
        return EnhancedOdometryResult(True, self.current_pose.copy(), num_features=len(kpts), 
                                      processing_time=time.time()-start_time, method_used=method)

    # 🟢 [关键修改]：接受 r_pred_guess (旋转矩阵)
    # def _solve_pose(self, object_points, image_points, num_feat, num_match, t_feat, t_match, start_time, r_pred_guess=None):
    #     """Common PnP Solver and Pose Update with IMU Guess"""
    #     obj_pts = np.array(object_points, dtype=np.float64)
    #     img_pts = np.array(image_points, dtype=np.float64)
        
    #     if len(obj_pts) < self.min_inliers:
    #         self._update_tracking_state(False)
    #         return EnhancedOdometryResult(False, self.current_pose.copy(), num_features=num_feat, 
    #                                       num_matches=num_match, method_used="insufficient_inliers")
        
    #     if self.camera_matrix is None:
    #         self.set_intrinsics(self.intrinsics['fx'], self.intrinsics['fy'], 
    #                             self.intrinsics['cx'], self.intrinsics['cy'])

    #     # =====================================================================
    #     # 🟢 构造 PnP 初始猜测 (Fusion Strategy)
    #     # =====================================================================
    #     r_init, t_init = None, None
        
    #     # 优先级 1: IMU 旋转预测
    #     if r_pred_guess is not None:
    #         # r_pred_guess 是旋转矩阵 R (Prev -> Curr)
    #         r_init, _ = cv2.Rodrigues(r_pred_guess)
    #         # 对于平移，如果没有 IMU 加速度计二次积分，最好假设为 0 或沿用上一次速度
    #         # 这里简单混合：使用运动模型预测平移
    #         if self.use_motion_model:
    #             delta = self.motion_model.predict()
    #             t_init = delta[:3, 3:4]
    #         else:
    #             t_init = np.zeros((3,1), dtype=np.float64)
                
    #     # 优先级 2: 纯视觉恒速模型
    #     elif self.use_motion_model:
    #         delta = self.motion_model.predict()
    #         r_init, _ = cv2.Rodrigues(delta[:3,:3])
    #         t_init = delta[:3, 3:4]

    #     t_pnp_start = time.time()
        
    #     # 调用求解器 (自动处理 Extrinsic Guess)
    #     success, rvec, tvec, inliers, info = self.pnp_solver.solve(
    #         obj_pts, img_pts, self.camera_matrix, r_init, t_init
    #     )
    #     t_pnp = time.time() - t_pnp_start

    #     if not success:
    #         self._update_tracking_state(False)
    #         return EnhancedOdometryResult(False, self.current_pose.copy(), num_features=num_feat, 
    #                                       num_matches=num_match, method_used="pnp_failed")

    #     # 恢复位姿 T_curr_prev (因为 solvePnP 算的是 Obj->Cam, 也就是 Prev->Curr)
    #     R, _ = cv2.Rodrigues(rvec)
    #     T_local = np.eye(4)
    #     T_local[:3, :3] = R
    #     T_local[:3, 3] = tvec.flatten()
        
    #     # 计算全局位姿增量 T_prev_curr (是 T_local 的逆吗？)
    #     # solvePnP: P_curr = R * P_prev + t  => P_curr = T_local * P_prev
    #     # 我们维护的是 T_world_cam (Camera to World)
    #     # T_wc_curr = T_wc_prev * T_prev_curr
    #     # P_world = T_wc_curr * P_curr  (通常 T_wc 是相机中心在世界坐标，所以是 P_world = T_wc * P_cam ? 不，通常是 Pose matrix)
    #     # 严谨定义：self.current_pose 是 T_world_from_camera (相机在世界坐标系下的位姿)
    #     # P_world = T_world_from_camera * P_camera
    #     # 而 solvePnP 给出的是 T_camera_from_prev (把上一帧点转到当前帧)
    #     # P_curr = T_cf_p * P_prev
    #     # => P_prev = T_cf_p^-1 * P_curr
    #     # => P_world = T_wc_prev * P_prev = T_wc_prev * (T_cf_p^-1 * P_curr)
    #     # => T_wc_curr = T_wc_prev * T_cf_p^-1
        
    #     try:
    #         delta_T = np.linalg.inv(T_local) # 即 T_prev_from_curr
    #     except:
    #          return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="inv_failed")

    #     # 跳变检测
    #     trans_mag = np.linalg.norm(delta_T[:3, 3])
    #     if trans_mag > 1.5: # 放宽一点阈值，以免快速运动被误杀
    #         print(f"⚠️ [VO] Large Jump: {trans_mag:.2f}m. IMU={r_pred_guess is not None}")
    #         if r_pred_guess is None: # 如果没有 IMU 护体，就拒绝
    #              self._update_tracking_state(False)
    #              return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="large_jump")
    #         # 如果有 IMU，我们选择信任（或者至少不直接 kill），除非真的太离谱

    #     self.current_pose = self.current_pose @ delta_T
    #     if self.use_motion_model: self.motion_model.update(delta_T)
    #     self._update_tracking_state(True)
        
    #     return EnhancedOdometryResult(
    #         success=True, pose=self.current_pose.copy(),
    #         num_features=num_feat, num_matches=num_match,
    #         inliers=len(inliers), inlier_ratio=len(inliers)/len(obj_pts),
    #         method_used=self.frontend_mode + ("+IMU" if r_pred_guess is not None else ""),
    #         processing_time=time.time()-start_time,
    #         feature_time=t_feat, matching_time=t_match, pnp_time=t_pnp
    #     )


    # 请定位到 visual_odometry_enhanced.py 约 390 行的 _solve_pose 函数
    # 用以下代码替换原函数：

    # def _solve_pose(self, object_points, image_points, num_feat, num_match, t_feat, t_match, start_time, r_pred_guess=None):
    #     """
    #     [Academic Optimization] Pose Solver with Enforced IMU Fusion
    #     """
    #     obj_pts = np.array(object_points, dtype=np.float64)
    #     img_pts = np.array(image_points, dtype=np.float64)
        
    #     # 1. 安全检查
    #     if len(obj_pts) < self.min_inliers:
    #         self._update_tracking_state(False)
    #         return EnhancedOdometryResult(False, self.current_pose.copy(), num_features=num_feat, 
    #                                       num_matches=num_match, method_used="insufficient_inliers")
        
    #     if self.camera_matrix is None:
    #         self.set_intrinsics(self.intrinsics['fx'], self.intrinsics['fy'], 
    #                             self.intrinsics['cx'], self.intrinsics['cy'])

    #     # 2. 构造初始猜测 (PnP Guess)
    #     r_init, t_init = None, None
        
    #     if r_pred_guess is not None:
    #         r_init, _ = cv2.Rodrigues(r_pred_guess)
    #         # 平移猜测：使用恒速模型或保持静止
    #         if self.use_motion_model:
    #             delta = self.motion_model.predict()
    #             t_init = delta[:3, 3:4]
    #         else:
    #             t_init = np.zeros((3,1), dtype=np.float64)
                
    #     elif self.use_motion_model:
    #         delta = self.motion_model.predict()
    #         r_init, _ = cv2.Rodrigues(delta[:3,:3])
    #         t_init = delta[:3, 3:4]

    #     t_pnp_start = time.time()
        
    #     # 3. 核心解算 (PnP RANSAC)
    #     # 注意：这里我们依然让 PnP 优化旋转，以获得正确的内点(Inliers)和位移(tvec)
    #     try:
    #         success, rvec, tvec, inliers, info = self.pnp_solver.solve(
    #             obj_pts, img_pts, self.camera_matrix, r_init, t_init
    #         )
    #     except Exception as e:
    #         print(f"PnP Error: {e}")
    #         success = False

    #     t_pnp = time.time() - t_pnp_start

    #     if not success:
    #         self._update_tracking_state(False)
    #         return EnhancedOdometryResult(False, self.current_pose.copy(), num_features=num_feat, 
    #                                       num_matches=num_match, method_used="pnp_failed")

    #     # =====================================================================
    #     # [学术优化核心]：强制 IMU 姿态锁定 (Enforced Rotation)
    #     # =====================================================================
    #     # 原理：PnP 算出的位移 tvec 是可靠的，但旋转 rvec 容易受特征点分布影响而漂移。
    #     # 如果我们有高频 IMU (r_pred_guess)，直接信任 IMU 的旋转。
        
    #     T_local = np.eye(4)
        
    #     if r_pred_guess is not None:
    #         # 🟢 方案 A: 强制使用 IMU 旋转 (最稳，消除旋转漂移)
    #         # r_pred_guess 是从上一帧到当前帧的相对旋转 (R_curr_prev)
    #         R_final = r_pred_guess
    #         method_tag = self.frontend_mode + "+IMU_Locked"
    #     else:
    #         # ⚪ 方案 B: 纯视觉旋转 (无 IMU 时回退)
    #         R_final, _ = cv2.Rodrigues(rvec)
    #         method_tag = self.frontend_mode
            
    #     T_local[:3, :3] = R_final
    #     T_local[:3, 3] = tvec.flatten() # 保持 PnP 算出的位移
        
    #     # =====================================================================

    #     # 4. 计算全局位姿更新
    #     # T_local 是 T_curr_prev (把上一帧点转到当前帧)
    #     # 我们需要 delta_T = T_prev_curr = inv(T_local)
    #     try:
    #         delta_T = np.linalg.inv(T_local)
    #     except:
    #          return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="inv_failed")

    #     # 5. 跳变检测与更新
    #     trans_mag = np.linalg.norm(delta_T[:3, 3])
    #     if trans_mag > 1.5: 
    #         print(f"⚠️ [VO] Large Jump: {trans_mag:.2f}m.")
    #         # 如果有 IMU 锁定旋转，我们对大位移可以宽容一点，因为旋转没得跑
    #         if r_pred_guess is None: 
    #              self._update_tracking_state(False)
    #              return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="large_jump")

    #     self.current_pose = self.current_pose @ delta_T
    #     if self.use_motion_model: self.motion_model.update(delta_T)
    #     self._update_tracking_state(True)
        
    #     return EnhancedOdometryResult(
    #         success=True, pose=self.current_pose.copy(),
    #         num_features=num_feat, num_matches=num_match,
    #         inliers=len(inliers) if inliers is not None else 0, 
    #         inlier_ratio=len(inliers)/len(obj_pts) if inliers is not None else 0,
    #         method_used=method_tag,
    #         processing_time=time.time()-start_time,
    #         feature_time=t_feat, matching_time=t_match, pnp_time=t_pnp,
    #         # 传递当前特征点用于可视化
    #         current_keypoints=None 
    #     )


    # 定位到 visual_odometry_enhanced.py 中的 _solve_pose 函数
    # 用以下代码完全替换该函数

    # def _solve_pose(self, object_points, image_points, num_feat, num_match, t_feat, t_match, start_time, r_pred_guess=None):
    #     """
    #     [Academic Correction] Soft-Coupled Solver (IMU Guess + PnP Refinement)
    #     修复：不再强制锁定IMU旋转，而是将其作为PnP的初始猜测，允许视觉自动修正外参误差。
    #     """
    #     obj_pts = np.array(object_points, dtype=np.float64)
    #     img_pts = np.array(image_points, dtype=np.float64)
        
    #     # 1. 基础检查
    #     if len(obj_pts) < self.min_inliers:
    #         self._update_tracking_state(False)
    #         return EnhancedOdometryResult(False, self.current_pose.copy(), num_features=num_feat, 
    #                                       num_matches=num_match, method_used="insufficient_inliers")
        
    #     if self.camera_matrix is None:
    #         self.set_intrinsics(self.intrinsics['fx'], self.intrinsics['fy'], 
    #                             self.intrinsics['cx'], self.intrinsics['cy'])

    #     # 2. 构造初始猜测 (Initial Guess) - 软耦合的核心
    #     r_init, t_init = None, None
    #     use_guess = False
        
    #     # 优先使用 IMU 预测作为 PnP 的起跑线
    #     if r_pred_guess is not None:
    #         r_init, _ = cv2.Rodrigues(r_pred_guess) # 将 IMU 旋转转为向量
    #         t_init = np.zeros((3,1), dtype=np.float64) # 平移暂时给0
    #         use_guess = True
    #     elif self.use_motion_model:
    #         delta = self.motion_model.predict()
    #         r_init, _ = cv2.Rodrigues(delta[:3,:3])
    #         t_init = delta[:3, 3:4]
    #         use_guess = True

    #     # 3. 核心解算 (Iterative PnP with Extrinsic Guess)
    #     # 关键修改：使用 SOLVEPNP_ITERATIVE 并开启 useExtrinsicGuess
    #     # 这会自动利用 r_init 修正方向，但最终结果由视觉决定，从而消除安装误差
    #     try:
    #         # 必须保证 r_init 格式正确
    #         if use_guess:
    #              if r_init is None: r_init = np.zeros((3,1))
    #              if t_init is None: t_init = np.zeros((3,1))

    #         # success, rvec, tvec, inliers = cv2.solvePnPRansac(
    #         #     obj_pts, img_pts, self.camera_matrix, None,
    #         #     rvec=r_init, tvec=t_init, 
    #         #     useExtrinsicGuess=use_guess, # 🟢 启用猜测
    #         #     iterationsCount=self.ransac_iterations,
    #         #     reprojectionError=self.ransac_reproj_error, # 默认2.0，严格控制精度
    #         #     confidence=0.999,
    #         #     flags=cv2.SOLVEPNP_ITERATIVE # 🟢 迭代法精度最高
    #         # )

    #         # 尝试获取特征追踪长度（如果在 process_xfeat 中计算了的话）
    #         # 即使没有 track_lengths，WeightedPnPSolver 也能通过深度计算权重
    #         track_lens = None
    #         # 如果您在 EnhancedOdometryResult 或 context 里存了 track_lengths，这里可以传入
    #         # 例如：track_lens = self.prev_track_lengths (需要您在类里维护这个变量)

    #         success, rvec, tvec, inliers, info = self.pnp_solver.solve(
    #             obj_pts, img_pts, self.camera_matrix, 
    #             track_lengths=track_lens, # 传入追踪长度
    #             initial_rvec=r_init, initial_tvec=t_init # 注意参数名可能需要对应
    #         )
                        
    #         # 构造返回信息
    #         info = {'method': 'PnP+IMU' if r_pred_guess is not None else 'PnP+Motion'}
            
    #     except Exception as e:
    #         print(f"PnP Error: {e}")
    #         success = False
    #         inliers = []

    #     if not success or inliers is None:
    #         self._update_tracking_state(False)
    #         return EnhancedOdometryResult(False, self.current_pose.copy(), num_features=num_feat, 
    #                                       num_matches=num_match, method_used="pnp_failed")

    #     # 4. 计算位姿增量
    #     # 此时 rvec 是经过视觉修正后的准确旋转
    #     R, _ = cv2.Rodrigues(rvec)
    #     T_local = np.eye(4)
    #     T_local[:3, :3] = R
    #     T_local[:3, 3] = tvec.flatten()
        
    #     try:
    #         delta_T = np.linalg.inv(T_local)
    #     except:
    #          return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="inv_failed")

    #     # 5. 简单的跳变检测 (防飞)
    #     if np.linalg.norm(delta_T[:3, 3]) > 2.0: 
    #          self._update_tracking_state(False)
    #          return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="large_jump")

    #     # 6. 更新全局位姿
    #     self.current_pose = self.current_pose @ delta_T
    #     if self.use_motion_model: self.motion_model.update(delta_T)
    #     self._update_tracking_state(True)
        
    #     # 计算耗时统计
    #     t_pnp = 0.0 # 简化统计
        
    #     return EnhancedOdometryResult(
    #         success=True, pose=self.current_pose.copy(),
    #         num_features=num_feat, num_matches=num_match,
    #         inliers=len(inliers), inlier_ratio=len(inliers)/len(obj_pts),
    #         method_used=info['method'],
    #         processing_time=time.time()-start_time,
    #         feature_time=t_feat, matching_time=t_match, pnp_time=t_pnp
    #     )

    def _solve_pose(self, object_points, image_points, num_feat, num_match, t_feat, t_match, start_time, r_pred_guess=None):
        """
        [Academic Optimization] Hybrid PnP Solver (EPnP + Iterative Refinement)
        """
        obj_pts = np.array(object_points, dtype=np.float64)
        img_pts = np.array(image_points, dtype=np.float64)
        
        # 1. 安全检查：点数过少直接丢弃，防止计算出奇异解
        if len(obj_pts) < max(self.min_inliers, 6): # 至少需要6个点才能解出唯一解
            self._update_tracking_state(False)
            return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="insufficient_points")

        if self.camera_matrix is None:
            self.set_intrinsics(self.intrinsics['fx'], self.intrinsics['fy'], 
                                self.intrinsics['cx'], self.intrinsics['cy'])

        # 2. 策略分歧：有 IMU 猜测 vs 无 IMU 猜测
        use_guess = False
        r_vec_init, t_vec_init = None, None
        
        # 如果有 IMU 预测，我们把这个预测转换成向量
        if r_pred_guess is not None:
            r_vec_init, _ = cv2.Rodrigues(r_pred_guess)
            t_vec_init = np.zeros((3, 1), dtype=np.float64) # 平移很难预测，设为0
            use_guess = True
        
        # 3. 核心解算：两阶段法 (Two-Stage Solver)
        # 阶段 A: 鲁棒估计 (RANSAC) - 这一步不过分依赖初值，用来剔除外点
        # 重点：如果点很少(<50)，EPnP 更稳；如果点很多，Iterative 更快。这里我们强制用 EPnP 做 RANSAC 核心。
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.camera_matrix, None,
                rvec=r_vec_init, tvec=t_vec_init,
                useExtrinsicGuess=use_guess, 
                iterationsCount=self.ransac_iterations, # 建议设为 1000
                reprojectionError=2.0,                  # 严格阈值：2像素
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP                 # [关键] 使用 EPnP 防止陷入局部最优
            )
        except Exception as e:
            print(f"PnP RANSAC Error: {e}")
            success = False

        if not success or inliers is None or len(inliers) < self.min_inliers:
            self._update_tracking_state(False)
            return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="pnp_ransac_failed")
            
        # 阶段 B: 局部精化 (Refinement) - 只使用内点 (Inliers)
        # 使用 LM 算法进一步优化位姿，减少抖动
        inliers_idx = inliers.flatten()
        try:
            # 只有内点足够多才做精化
            if len(inliers_idx) > 10:
                success_refine, rvec, tvec = cv2.solvePnP(
                    obj_pts[inliers_idx], img_pts[inliers_idx], 
                    self.camera_matrix, None,
                    rvec=rvec, tvec=tvec, 
                    useExtrinsicGuess=True, 
                    flags=cv2.SOLVEPNP_ITERATIVE # 使用 LM 迭代法精化
                )
        except:
            pass # 如果精化失败，就保留 RANSAC 的结果

        # 4. 几何一致性检查 (Sanity Check)
        # 如果这一帧算出来的位移大得离谱（比如 0.1秒 移动了 2米），肯定是算错了
        t_norm = np.linalg.norm(tvec)
        if t_norm > 5.0: # 设定一个物理上限
             return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="motion_insanity")

        # 5. 更新位姿
        R, _ = cv2.Rodrigues(rvec)
        T_local = np.eye(4)
        T_local[:3, :3] = R
        T_local[:3, 3] = tvec.flatten()
        
        # 计算 T_world
        # 注意：solvePnP 求出的是 T_camera_from_world (将世界点转到相机)
        # 还是 T_camera_from_prev (将上一帧点转到当前帧)？
        # 代码逻辑中，object_points 是上一帧的相机坐标系下的 3D 点。
        # 所以 solvePnP 求出的是 T_curr_prev。
        # P_curr = T_curr_prev * P_prev
        # 我们维护的 self.current_pose 通常是 T_world_curr (相机在世界中的位姿)
        # T_world_curr = T_world_prev * T_prev_curr
        #              = T_world_prev * (T_curr_prev)^-1
        
        try:
            T_curr_prev = T_local
            T_prev_curr = np.linalg.inv(T_curr_prev)
            self.current_pose = self.current_pose @ T_prev_curr
        except:
            return EnhancedOdometryResult(False, self.current_pose.copy(), method_used="matrix_inv_failed")

        self._update_tracking_state(True)
        
        return EnhancedOdometryResult(
            success=True, pose=self.current_pose.copy(),
            num_features=num_feat, num_matches=num_match,
            inliers=len(inliers_idx), 
            inlier_ratio=len(inliers_idx)/len(obj_pts),
            method_used="EPnP+Refine",
            processing_time=time.time()-start_time
        )


    def _update_history(self, rgb, depth, kpts, descs, scores):
        """Update historical data cache with 3D projection"""
        self.prev_rgb = rgb
        self.prev_depth = depth
        self.prev_kpts = kpts
        self.prev_descs = descs
        self.prev_scores = scores

        depth_grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        depth_grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = cv2.magnitude(depth_grad_x, depth_grad_y)
        grad_threshold = 0.5 # 阈值需要根据深度图数值范围调整，如果 depth 是米，0.5m的跳变认为是边缘
        
        self.prev_kpts_3d = []
        h, w = depth.shape
        fx, fy, cx, cy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy']
        d_min, d_max = self.intrinsics['depth_min'], self.intrinsics['depth_max']
        
        for i, pt in enumerate(kpts):
            u, v = int(round(pt[0])), int(round(pt[1]))
            p3d = None
            # if 0 <= u < w and 0 <= v < h:
            #     d = depth[v, u]
            #     if d_min < d < d_max:
            #         z = float(d)
            #         x = (pt[0] - cx) * z / fx
            #         y = (pt[1] - cy) * z / fy
            #         p3d = np.array([x, y, z], dtype=np.float32)
            
            # 边界检查
            if 10 <= u < w-10 and 10 <= v < h-10: # 去掉图像最边缘的区域
                d = depth[v, u]
                g = depth_grad[v, u]
                
                # [核心优化] 仅当深度在合理范围，且不是边缘点时，才信任它
                if d_min < d < d_max and g < grad_threshold:
                    z = float(d)
                    x = (pt[0] - cx) * z / fx
                    y = (pt[1] - cy) * z / fy
                    p3d = np.array([x, y, z], dtype=np.float32)
            
            score_val = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            self.prev_kpts_3d.append(EnhancedKeyPoint(
                pt=tuple(pt),
                size=1.0, angle=0.0, response=score_val, octave=0,
                point_3d=p3d
            ))
            
        self.feature_manager.prev_keypoints = self.prev_kpts_3d
        self.feature_manager.prev_descriptors = self.prev_descs

    def _update_tracking_state(self, success: bool):
        if success:
            self.tracking_lost = False
        elif self.frame_count > 10:
            self.tracking_lost = True
            self.motion_model.reset()

    def get_current_pose(self): return self.current_pose.copy()
    def get_position(self): return self.current_pose[:3, 3].copy()
    def reset(self):
        self.current_pose = np.eye(4)
        self.is_initialized = False
        self.motion_model.reset()
        print("Visual Odometry Reset")

# =============================================================================
# 运动模型 (Simplified Constant Velocity)
# =============================================================================

class MotionModel:
    def __init__(self, decay=0.85):
        self.decay = decay
        self.last_delta = None
        
    def update(self, delta_T):
        if self.last_delta is None: self.last_delta = delta_T
        else:
            t_new = delta_T[:3, 3] * (1-self.decay) + self.last_delta[:3, 3] * self.decay
            T_new = np.eye(4)
            T_new[:3, 3] = t_new
            # 旋转部分不混合，太复杂，直接用新的或IMU覆盖
            T_new[:3, :3] = delta_T[:3, :3]
            self.last_delta = T_new
            
    def predict(self):
        return self.last_delta.copy() if self.last_delta is not None else np.eye(4)
    
    def reset(self): self.last_delta = None

# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    print("Test: Initializing VO...")
    vo = EnhancedVisualOdometry()
    img = np.random.randint(0,255,(480,640,3),dtype=np.uint8)
    depth = np.ones((480,640),dtype=np.float32)*2.0
    res = vo.process_frame(img, depth)
    print(f"Result: Success={res.success}, Method={res.method_used}")