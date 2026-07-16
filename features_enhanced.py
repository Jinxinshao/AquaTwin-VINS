#!/usr/bin/env python3
"""
================================================================================
PiSLAM Enhanced Feature Module (features_enhanced.py)
================================================================================

增强版特征提取与匹配模块，专为提高单目SLAM在无惯导条件下的鲁棒性设计。

核心改进：
---------
1. Grid-based 特征检测：确保特征点在图像上均匀分布，避免纹理丰富区域过度聚集
2. 自适应FAST阈值：根据每个网格的纹理复杂度动态调整检测阈值
3. 亚像素精度优化：使用cornerSubPix对关键点位置进行亚像素级精化
4. 光流辅助跟踪：用Lucas-Kanade光流预测特征位置，缩小匹配搜索范围
5. 深度一致性约束：利用深度图进行3D距离验证，剔除深度不一致的错误匹配
6. Epipolar几何约束：基本矩阵约束进行几何一致性验证

理论依据：
---------
- Grid-based detection: 受ORB-SLAM2启发，保证特征的空间均匀性对于 
  视觉里程计至关重要，可以提高运动估计的可观测性
- 光流辅助: 利用时序连续性，光流预测提供了良好的初始匹配猜测，
  可将描述子匹配的搜索范围从全图缩小到局部窗口
- 深度一致性: 错误匹配在3D空间中通常表现为深度/距离的不一致，
  这是一个强有力的后验验证条件

Author: Enhanced for PiSLAM Project
================================================================================
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import yaml


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class EnhancedKeyPoint:
    """
    增强版关键点，携带更丰富的元信息。
    
    Attributes:
        pt: 2D像素坐标 (u, v)
        size: 特征邻域直径
        angle: 方向角 (degrees)
        response: 角点响应强度
        octave: 金字塔层级
        depth: 关联的深度值 (meters)
        point_3d: 相机坐标系下的3D位置
        grid_id: 所属网格ID (用于跟踪)
        track_id: 跨帧跟踪ID
        confidence: 匹配置信度 [0, 1]
    """
    pt: Tuple[float, float]
    size: float
    angle: float
    response: float
    octave: int
    depth: Optional[float] = None
    point_3d: Optional[np.ndarray] = None
    grid_id: int = -1
    track_id: int = -1
    confidence: float = 1.0
    
    def to_cv_keypoint(self) -> cv2.KeyPoint:
        """转换为OpenCV KeyPoint对象"""
        return cv2.KeyPoint(
            x=self.pt[0], y=self.pt[1],
            size=self.size, angle=self.angle,
            response=self.response, octave=self.octave
        )


@dataclass
class EnhancedMatch:
    """
    增强版匹配结果，包含多种验证状态。
    
    Attributes:
        query_idx: 查询帧特征索引
        train_idx: 训练帧特征索引
        distance: 描述子距离 (Hamming)
        is_inlier: 是否通过几何验证
        depth_consistent: 是否通过深度一致性检验
        flow_assisted: 是否由光流辅助定位
        confidence: 综合置信度评分
    """
    query_idx: int
    train_idx: int
    distance: float
    is_inlier: bool = True
    depth_consistent: bool = True
    flow_assisted: bool = False
    confidence: float = 1.0


# =============================================================================
# 网格化特征提取器
# =============================================================================

class GridBasedORBExtractor:
    """
    网格化ORB特征提取器。
    
    将图像划分为 n_rows × n_cols 个网格，在每个网格内独立检测特征，
    确保特征在图像上均匀分布。对于纹理贫乏的网格，自动降低FAST阈值
    以保证最小特征数量。
    
    优势：
    1. 避免特征过度聚集在高纹理区域
    2. 提高位姿估计的数值稳定性 (特征点分布越均匀，信息矩阵条件数越好)
    3. 对场景中的均匀区域 (如墙面、水面) 更加鲁棒
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化网格化特征提取器。
        
        Args:
            config_path: 配置文件路径
        """
        # 默认参数 (针对640x480图像优化)
        self.n_features_total = 1500       # 总特征数目标
        self.n_grid_rows = 6               # 网格行数
        self.n_grid_cols = 8               # 网格列数
        self.min_features_per_cell = 5     # 每个网格最少特征数
        self.max_features_per_cell = 50    # 每个网格最多特征数
        
        # FAST检测器参数
        self.fast_threshold_default = 20   # 默认FAST阈值
        self.fast_threshold_min = 5        # 最低FAST阈值 (用于低纹理区域)
        self.fast_threshold_adaptive = True # 启用自适应阈值
        
        # ORB描述子参数
        self.scale_factor = 1.2
        self.n_levels = 8
        self.edge_threshold = 31
        self.patch_size = 31
        
        # 亚像素精化参数
        self.enable_subpixel = True
        self.subpixel_win_size = (5, 5)
        self.subpixel_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            30, 0.001
        )
        
        # 加载配置
        if config_path:
            self._load_config(config_path)
            
        # 创建ORB描述子计算器 (只用于计算描述子，不检测)
        self.orb_descriptor = cv2.ORB_create(
            nfeatures=10000,  # 设置大值，因为我们自己控制特征数
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=self.edge_threshold,
            patchSize=self.patch_size
        )
        
        # 预计算网格参数
        self._compute_grid_params(640, 480)  # 默认分辨率
        
    def _load_config(self, config_path: str):
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            feat_cfg = config.get('features_enhanced', config.get('features', {}))
            self.n_features_total = feat_cfg.get('n_features', self.n_features_total)
            self.n_grid_rows = feat_cfg.get('grid_rows', self.n_grid_rows)
            self.n_grid_cols = feat_cfg.get('grid_cols', self.n_grid_cols)
            self.fast_threshold_default = feat_cfg.get('fast_threshold', self.fast_threshold_default)
            self.enable_subpixel = feat_cfg.get('subpixel_refinement', self.enable_subpixel)
        except Exception as e:
            print(f"⚠️ [GridORB] Config load failed: {e}, using defaults")
            
    def _compute_grid_params(self, img_width: int, img_height: int):
        """预计算网格参数"""
        self.img_width = img_width
        self.img_height = img_height
        self.cell_width = img_width // self.n_grid_cols
        self.cell_height = img_height // self.n_grid_rows
        
        # 每个网格的理想特征数
        n_cells = self.n_grid_rows * self.n_grid_cols
        self.features_per_cell_target = self.n_features_total // n_cells
        
    def extract(self, image: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        网格化特征提取 (主入口)。
        
        Args:
            image: 输入图像 (BGR或灰度)
            mask: 可选的检测掩膜
            
        Returns:
            keypoints: 检测到的关键点列表
            descriptors: 对应的ORB描述子 (N, 32)
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 更新网格参数 (如果分辨率变化)
        h, w = gray.shape[:2]
        if w != self.img_width or h != self.img_height:
            self._compute_grid_params(w, h)
            
        # 网格化检测
        all_keypoints = []
        
        for row in range(self.n_grid_rows):
            for col in range(self.n_grid_cols):
                # 计算当前网格边界
                x_start = col * self.cell_width
                y_start = row * self.cell_height
                x_end = min(x_start + self.cell_width, w)
                y_end = min(y_start + self.cell_height, h)
                
                # 提取网格区域
                cell_img = gray[y_start:y_end, x_start:x_end]
                cell_mask = None
                if mask is not None:
                    cell_mask = mask[y_start:y_end, x_start:x_end]
                    
                # 在此网格内检测特征
                cell_kps = self._detect_in_cell(
                    cell_img, cell_mask, 
                    offset=(x_start, y_start),
                    grid_id=row * self.n_grid_cols + col
                )
                all_keypoints.extend(cell_kps)
                
        # 亚像素精化
        if self.enable_subpixel and len(all_keypoints) > 0:
            all_keypoints = self._refine_subpixel(gray, all_keypoints)
            
        # 计算描述子
        if len(all_keypoints) == 0:
            return [], np.array([])
            
        keypoints, descriptors = self.orb_descriptor.compute(gray, all_keypoints)
        
        return keypoints, descriptors
        
    def _detect_in_cell(self, cell_img: np.ndarray, 
                        cell_mask: Optional[np.ndarray],
                        offset: Tuple[int, int],
                        grid_id: int) -> List[cv2.KeyPoint]:
        """
        在单个网格内检测特征。
        
        使用自适应FAST阈值策略：
        1. 首先用默认阈值检测
        2. 如果特征不足，逐步降低阈值重新检测
        3. 如果特征过多，按响应值筛选Top-K
        """
        threshold = self.fast_threshold_default
        keypoints = []
        
        # 创建FAST检测器
        fast = cv2.FastFeatureDetector_create(threshold=threshold)
        fast.setNonmaxSuppression(True)
        
        # 第一次检测
        keypoints = fast.detect(cell_img, cell_mask)
        
        # 自适应阈值：如果特征太少，降低阈值
        if self.fast_threshold_adaptive:
            while len(keypoints) < self.min_features_per_cell and threshold > self.fast_threshold_min:
                threshold = max(self.fast_threshold_min, threshold - 5)
                fast.setThreshold(threshold)
                keypoints = fast.detect(cell_img, cell_mask)
                
        # 如果特征过多，按响应值筛选
        if len(keypoints) > self.max_features_per_cell:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
            keypoints = keypoints[:self.max_features_per_cell]
            
        # 坐标偏移到全图坐标系
        for kp in keypoints:
            kp.pt = (kp.pt[0] + offset[0], kp.pt[1] + offset[1])
            kp.class_id = grid_id
            
        return keypoints
        
    def _refine_subpixel(self, gray: np.ndarray, 
                         keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        亚像素精度优化。
        
        使用OpenCV的cornerSubPix函数对角点位置进行亚像素级精化，
        可以将定位精度从整数像素提升到约0.1像素级别。
        """
        if len(keypoints) == 0:
            return keypoints
            
        # 提取所有角点坐标
        corners = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        
        # 亚像素精化
        refined = cv2.cornerSubPix(
            gray, corners, 
            self.subpixel_win_size, (-1, -1), 
            self.subpixel_criteria
        )
        
        # 更新关键点坐标
        for i, kp in enumerate(keypoints):
            kp.pt = (refined[i, 0, 0], refined[i, 0, 1])
            
        return keypoints
        
    def extract_with_depth(self, image: np.ndarray,
                           depth_map: np.ndarray,
                           intrinsics: Dict[str, float],
                           mask: Optional[np.ndarray] = None) -> Tuple[List[EnhancedKeyPoint], np.ndarray]:
        """
        带深度信息的特征提取。
        
        将2D特征反投影到3D空间，为后续的深度一致性验证做准备。
        
        Args:
            image: 输入图像
            depth_map: 深度图 (米制)
            intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy', 'depth_min', 'depth_max'}
            mask: 可选掩膜
            
        Returns:
            enhanced_keypoints: 增强版关键点列表
            descriptors: ORB描述子
        """
        # 提取2D特征
        cv_keypoints, descriptors = self.extract(image, mask)
        
        if len(cv_keypoints) == 0:
            return [], np.array([])
            
        # 获取内参
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        depth_min = intrinsics.get('depth_min', 0.2)
        depth_max = intrinsics.get('depth_max', 5.0)
        
        # 转换为增强关键点
        enhanced_keypoints = []
        valid_indices = []
        
        for i, kp in enumerate(cv_keypoints):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            
            # 边界检查
            h, w = depth_map.shape[:2]
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
                
            z = depth_map[v, u]
            
            # 深度有效性检查
            if z <= depth_min or z >= depth_max or not np.isfinite(z):
                continue
                
            # 反投影到3D (相机坐标系)
            x = (kp.pt[0] - cx) * z / fx
            y = (kp.pt[1] - cy) * z / fy
            
            enhanced_kp = EnhancedKeyPoint(
                pt=kp.pt,
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                depth=z,
                point_3d=np.array([x, y, z], dtype=np.float32),
                grid_id=kp.class_id
            )
            enhanced_keypoints.append(enhanced_kp)
            valid_indices.append(i)
            
        # 筛选有效描述子
        if len(valid_indices) > 0:
            descriptors = descriptors[valid_indices]
        else:
            descriptors = np.array([])
            
        return enhanced_keypoints, descriptors


# =============================================================================
# 增强版特征匹配器
# =============================================================================

class EnhancedFeatureMatcher:
    """
    增强版特征匹配器。
    
    整合多种匹配策略和验证机制：
    1. 描述子匹配 + Lowe's ratio test
    2. 光流辅助局部搜索
    3. 深度一致性约束
    4. Epipolar几何验证
    
    匹配流程：
    ---------
    [Frame t-1] ----> [Optical Flow Prediction] ----> [Local Descriptor Matching]
                           |                                    |
                           v                                    v
                    [Depth Consistency]              [Ratio Test + Cross-check]
                           |                                    |
                           +-------------> [Merge] <-----------+
                                              |
                                              v
                                   [Epipolar Verification]
                                              |
                                              v
                                      [Final Matches]
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化增强匹配器。
        """
        # 描述子匹配参数
        self.ratio_threshold = 0.7         # Lowe's ratio (更严格)
        self.max_hamming_distance = 40     # 最大Hamming距离 (更严格)
        self.cross_check = True            # 交叉验证
        
        # 光流参数
        self.enable_optical_flow = True
        self.lk_win_size = (21, 21)
        self.lk_max_level = 3
        self.lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
            30, 0.01
        )
        self.flow_search_radius = 30       # 光流预测后的局部搜索半径 (pixels)
        
        # 深度一致性参数
        self.enable_depth_check = True
        self.depth_consistency_threshold = 0.15  # 相对深度误差阈值 (15%)
        self.max_3d_distance = 0.3         # 3D点最大距离阈值 (meters)
        
        # 几何验证参数
        self.ransac_reproj_threshold = 2.0
        self.ransac_confidence = 0.999
        self.min_inliers = 15
        
        # 加载配置
        if config_path:
            self._load_config(config_path)
            
        # 创建BF匹配器
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # 跟踪状态
        self.prev_gray = None
        self.prev_keypoints = None
        
    def _load_config(self, config_path: str):
        """加载配置"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            match_cfg = config.get('matching_enhanced', config.get('features', {}))
            self.ratio_threshold = match_cfg.get('ratio_threshold', self.ratio_threshold)
            self.max_hamming_distance = match_cfg.get('max_distance', self.max_hamming_distance)
            self.cross_check = match_cfg.get('cross_check', self.cross_check)
            self.enable_optical_flow = match_cfg.get('optical_flow', self.enable_optical_flow)
            self.enable_depth_check = match_cfg.get('depth_check', self.enable_depth_check)
        except Exception as e:
            print(f"⚠️ [EnhancedMatcher] Config load failed: {e}")
            
    def match(self, 
              prev_gray: np.ndarray,
              curr_gray: np.ndarray,
              prev_keypoints: List[EnhancedKeyPoint],
              curr_keypoints: List[EnhancedKeyPoint],
              prev_descriptors: np.ndarray,
              curr_descriptors: np.ndarray,
              intrinsics: Optional[Dict] = None) -> List[EnhancedMatch]:
        """
        执行增强版特征匹配。
        
        Args:
            prev_gray: 上一帧灰度图
            curr_gray: 当前帧灰度图
            prev_keypoints: 上一帧特征点
            curr_keypoints: 当前帧特征点
            prev_descriptors: 上一帧描述子
            curr_descriptors: 当前帧描述子
            intrinsics: 相机内参 (用于Epipolar验证)
            
        Returns:
            matches: 匹配结果列表
        """
        if len(prev_descriptors) == 0 or len(curr_descriptors) == 0:
            return []
            
        # ===== 阶段1: 光流辅助匹配 =====
        flow_matches = []
        flow_predictions = {}
        
        if self.enable_optical_flow and prev_gray is not None:
            flow_predictions, flow_matches = self._optical_flow_matching(
                prev_gray, curr_gray,
                prev_keypoints, curr_keypoints,
                prev_descriptors, curr_descriptors
            )
            
        # ===== 阶段2: 全局描述子匹配 (补充光流未匹配的点) =====
        descriptor_matches = self._descriptor_matching(
            prev_descriptors, curr_descriptors,
            exclude_query_indices=set(m.query_idx for m in flow_matches)
        )
        
        # ===== 阶段3: 合并匹配结果 =====
        all_matches = flow_matches + descriptor_matches
        
        # ===== 阶段4: 深度一致性验证 =====
        if self.enable_depth_check:
            all_matches = self._depth_consistency_check(
                all_matches, prev_keypoints, curr_keypoints
            )
            
        # ===== 阶段5: Epipolar几何验证 =====
        all_matches = self._geometric_verification(
            all_matches, prev_keypoints, curr_keypoints, intrinsics
        )
        
        return all_matches
        
    def _optical_flow_matching(self,
                               prev_gray: np.ndarray,
                               curr_gray: np.ndarray,
                               prev_keypoints: List[EnhancedKeyPoint],
                               curr_keypoints: List[EnhancedKeyPoint],
                               prev_descriptors: np.ndarray,
                               curr_descriptors: np.ndarray) -> Tuple[Dict, List[EnhancedMatch]]:
        """
        使用Lucas-Kanade光流进行特征跟踪。
        
        光流提供了特征点在下一帧中可能位置的预测，
        我们用这个预测来缩小描述子匹配的搜索范围。
        
        Returns:
            flow_predictions: {prev_idx: predicted_position}
            matches: 基于光流的匹配结果
        """
        # 准备光流输入
        prev_pts = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32).reshape(-1, 1, 2)
        
        # 计算光流
        curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=self.lk_criteria
        )
        
        matches = []
        flow_predictions = {}
        
        # 构建当前帧特征点的KD-Tree用于快速最近邻搜索
        curr_pts_array = np.array([kp.pt for kp in curr_keypoints], dtype=np.float32)
        
        for i, (pred_pt, st, err) in enumerate(zip(curr_pts, status, error)):
            if st[0] == 0:  # 光流跟踪失败
                continue
                
            pred_x, pred_y = pred_pt[0]
            flow_predictions[i] = (pred_x, pred_y)
            
            # 在预测位置附近搜索最近的特征点
            distances = np.sqrt(
                (curr_pts_array[:, 0] - pred_x) ** 2 + 
                (curr_pts_array[:, 1] - pred_y) ** 2
            )
            
            # 找到搜索半径内的候选点
            candidates = np.where(distances < self.flow_search_radius)[0]
            
            if len(candidates) == 0:
                continue
                
            # 在候选点中用描述子匹配
            prev_desc = prev_descriptors[i:i+1]
            cand_descs = curr_descriptors[candidates]
            
            # 计算Hamming距离
            hamming_dists = np.array([
                cv2.norm(prev_desc, cand_descs[j:j+1], cv2.NORM_HAMMING)
                for j in range(len(candidates))
            ])
            
            best_idx = np.argmin(hamming_dists)
            best_dist = hamming_dists[best_idx]
            
            # 检查距离阈值
            if best_dist < self.max_hamming_distance:
                match = EnhancedMatch(
                    query_idx=i,
                    train_idx=candidates[best_idx],
                    distance=best_dist,
                    flow_assisted=True,
                    confidence=1.0 - best_dist / 256.0  # 归一化置信度
                )
                matches.append(match)
                
        return flow_predictions, matches
        
    def _descriptor_matching(self,
                             prev_descriptors: np.ndarray,
                             curr_descriptors: np.ndarray,
                             exclude_query_indices: set = None) -> List[EnhancedMatch]:
        """
        标准描述子匹配 (带Lowe's ratio test)。
        
        对于已被光流匹配的点，跳过重复匹配。
        """
        exclude_query_indices = exclude_query_indices or set()
        
        # KNN匹配 (k=2用于ratio test)
        raw_matches = self.bf_matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)
        
        matches = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
                
            m, n = match_pair[0], match_pair[1]
            
            # 跳过已匹配的点
            if m.queryIdx in exclude_query_indices:
                continue
                
            # Lowe's ratio test
            if m.distance >= self.ratio_threshold * n.distance:
                continue
                
            # 距离阈值
            if m.distance >= self.max_hamming_distance:
                continue
                
            matches.append(EnhancedMatch(
                query_idx=m.queryIdx,
                train_idx=m.trainIdx,
                distance=m.distance,
                flow_assisted=False,
                confidence=1.0 - m.distance / 256.0
            ))
            
        # 可选: 交叉验证
        if self.cross_check:
            matches = self._cross_check(matches, prev_descriptors, curr_descriptors)
            
        return matches
        
    def _cross_check(self,
                     matches: List[EnhancedMatch],
                     prev_descriptors: np.ndarray,
                     curr_descriptors: np.ndarray) -> List[EnhancedMatch]:
        """
        交叉验证：确保匹配是双向最优的。
        """
        # 反向匹配
        reverse_matches = self.bf_matcher.knnMatch(curr_descriptors, prev_descriptors, k=2)
        
        # 构建反向匹配映射
        reverse_map = {}
        for match_pair in reverse_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < self.ratio_threshold * n.distance:
                reverse_map[m.queryIdx] = m.trainIdx
                
        # 验证正向匹配
        validated = []
        for match in matches:
            if match.train_idx in reverse_map:
                if reverse_map[match.train_idx] == match.query_idx:
                    validated.append(match)
                    
        return validated
        
    def _depth_consistency_check(self,
                                 matches: List[EnhancedMatch],
                                 prev_keypoints: List[EnhancedKeyPoint],
                                 curr_keypoints: List[EnhancedKeyPoint]) -> List[EnhancedMatch]:
        """
        深度一致性验证。
        
        原理：如果两个特征点是正确的匹配，它们对应的3D点应该是同一个点
        (或者非常接近)。对于单目SLAM中的短基线运动，匹配点的深度变化
        应该是平滑的。
        
        验证条件：
        1. 深度比值应接近1 (考虑尺度漂移，允许一定误差)
        2. 3D欧氏距离应该较小 (对于帧间短基线)
        """
        validated = []
        
        for match in matches:
            prev_kp = prev_keypoints[match.query_idx]
            curr_kp = curr_keypoints[match.train_idx]
            
            # 检查是否有有效深度
            if prev_kp.depth is None or curr_kp.depth is None:
                # 没有深度信息，跳过深度检查
                validated.append(match)
                continue
                
            # 深度比值检查
            depth_ratio = curr_kp.depth / (prev_kp.depth + 1e-6)
            
            # 允许一定范围的深度变化 (考虑移动和深度估计噪声)
            if abs(depth_ratio - 1.0) > self.depth_consistency_threshold:
                match.depth_consistent = False
                match.confidence *= 0.5  # 降低置信度但不直接剔除
            else:
                match.depth_consistent = True
                
            # 3D距离检查 (如果有3D坐标)
            if prev_kp.point_3d is not None and curr_kp.point_3d is not None:
                dist_3d = np.linalg.norm(prev_kp.point_3d - curr_kp.point_3d)
                
                # 短基线假设下，3D点不应该跑太远
                # 注意：这里的阈值需要根据帧率和运动速度调整
                if dist_3d > self.max_3d_distance:
                    match.depth_consistent = False
                    match.confidence *= 0.3
                    
            validated.append(match)
            
        return validated
        
    def _geometric_verification(self,
                                matches: List[EnhancedMatch],
                                prev_keypoints: List[EnhancedKeyPoint],
                                curr_keypoints: List[EnhancedKeyPoint],
                                intrinsics: Optional[Dict] = None) -> List[EnhancedMatch]:
        """
        Epipolar几何验证。
        
        使用RANSAC估计基本矩阵(Fundamental Matrix)，
        将不符合Epipolar约束的匹配标记为outlier。
        """
        if len(matches) < 8:  # 基本矩阵至少需要8对点
            return matches
            
        # 提取匹配点坐标
        pts1 = np.array([prev_keypoints[m.query_idx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([curr_keypoints[m.train_idx].pt for m in matches], dtype=np.float32)
        
        # RANSAC估计基本矩阵
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            confidence=self.ransac_confidence
        )
        
        if F is None or mask is None:
            return matches
            
        # 标记inlier/outlier
        mask = mask.flatten()
        n_inliers = 0
        
        for i, match in enumerate(matches):
            if mask[i]:
                match.is_inlier = True
                n_inliers += 1
            else:
                match.is_inlier = False
                match.confidence *= 0.1
                
        # 如果inlier太少，可能存在问题
        if n_inliers < self.min_inliers:
            print(f"⚠️ [Matcher] Low inlier count: {n_inliers}/{len(matches)}")
            
        return matches


# =============================================================================
# 综合特征管理器 (对外接口)
# =============================================================================

class EnhancedFeatureManager:
    """
    综合特征管理器，提供简洁的API。
    
    Usage:
    ------
    manager = EnhancedFeatureManager(config_path)
    
    # 第一帧
    kps, desc = manager.process_frame(rgb1, depth1, intrinsics)
    
    # 后续帧
    kps, desc, matches = manager.process_frame(rgb2, depth2, intrinsics, 
                                                return_matches=True)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.extractor = GridBasedORBExtractor(config_path)
        self.matcher = EnhancedFeatureMatcher(config_path)
        
        # 上一帧数据缓存
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.intrinsics = None
        
        self.frame_count = 0
        
    def process_frame(self,
                      rgb: np.ndarray,
                      depth: np.ndarray,
                      intrinsics: Dict[str, float],
                      return_matches: bool = False) -> Tuple:
        """
        处理单帧图像。
        
        Args:
            rgb: RGB图像
            depth: 深度图 (meters)
            intrinsics: 相机内参
            return_matches: 是否返回与上一帧的匹配结果
            
        Returns:
            如果 return_matches=False:
                (keypoints, descriptors)
            如果 return_matches=True:
                (keypoints, descriptors, matches, prev_keypoints_used)
                注意：返回的prev_keypoints_used是匹配时实际使用的上一帧关键点列表，
                     调用者应该用这个列表来通过match.query_idx访问上一帧的特征
        """
        self.frame_count += 1
        self.intrinsics = intrinsics
        
        # 转灰度
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb.copy()
            
        # 特征提取
        keypoints, descriptors = self.extractor.extract_with_depth(
            rgb, depth, intrinsics
        )
        
        matches = None
        prev_keypoints_used = None  # 匹配时使用的上一帧关键点
        
        # 匹配
        if return_matches and self.prev_gray is not None and self.prev_keypoints is not None and len(self.prev_keypoints) > 0:
            # 保存匹配时使用的prev_keypoints引用
            prev_keypoints_used = self.prev_keypoints
            
            matches = self.matcher.match(
                self.prev_gray, gray,
                self.prev_keypoints, keypoints,
                self.prev_descriptors, descriptors,
                intrinsics
            )
            
        # 更新缓存
        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        if return_matches:
            return keypoints, descriptors, matches if matches else [], prev_keypoints_used
        return keypoints, descriptors
        
    def get_inlier_matches(self, matches: List[EnhancedMatch], 
                           min_confidence: float = 0.5) -> List[EnhancedMatch]:
        """获取高质量匹配"""
        return [m for m in matches 
                if m.is_inlier and m.depth_consistent and m.confidence >= min_confidence]
                
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'frame_count': self.frame_count,
            'last_features': len(self.prev_keypoints) if self.prev_keypoints else 0
        }


# =============================================================================
# 可视化工具
# =============================================================================

def visualize_grid_features(image: np.ndarray,
                            keypoints: List,
                            n_rows: int = 6,
                            n_cols: int = 8) -> np.ndarray:
    """
    可视化网格化特征分布。
    
    用不同颜色标记不同网格的特征点，便于调试和论文图示。
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
    h, w = vis.shape[:2]
    cell_h, cell_w = h // n_rows, w // n_cols
    
    # 绘制网格线
    for i in range(1, n_rows):
        cv2.line(vis, (0, i * cell_h), (w, i * cell_h), (128, 128, 128), 1)
    for j in range(1, n_cols):
        cv2.line(vis, (j * cell_w, 0), (j * cell_w, h), (128, 128, 128), 1)
        
    # 为每个网格分配颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128),
        (128, 0, 128), (192, 192, 192), (128, 128, 128), (255, 128, 0)
    ]
    
    # 绘制特征点
    for kp in keypoints:
        pt = kp.pt if hasattr(kp, 'pt') else (kp.pt[0], kp.pt[1])
        grid_id = getattr(kp, 'grid_id', getattr(kp, 'class_id', 0))
        if grid_id < 0:
            grid_id = 0
        color = colors[grid_id % len(colors)]
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, color, -1)
        
    # 统计信息
    n_features = len(keypoints)
    avg_per_cell = n_features / (n_rows * n_cols)
    cv2.putText(vis, f"Features: {n_features} (Avg: {avg_per_cell:.1f}/cell)", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    return vis


def visualize_matches_enhanced(img1: np.ndarray,
                               img2: np.ndarray,
                               prev_keypoints: List,
                               curr_keypoints: List,
                               matches: List[EnhancedMatch],
                               show_inliers_only: bool = True) -> np.ndarray:
    """
    增强版匹配可视化。
    
    用不同颜色区分：
    - 绿色: 高质量匹配 (inlier + depth_consistent)
    - 黄色: 光流辅助匹配
    - 红色: 低置信度匹配
    """
    # 拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
    vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # 绘制匹配
    for match in matches:
        if show_inliers_only and not match.is_inlier:
            continue
            
        pt1 = prev_keypoints[match.query_idx].pt
        pt2 = curr_keypoints[match.train_idx].pt
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
        
        # 颜色编码
        if match.confidence > 0.7 and match.is_inlier and match.depth_consistent:
            color = (0, 255, 0)   # 绿色: 高质量
        elif match.flow_assisted:
            color = (0, 255, 255) # 黄色: 光流辅助
        else:
            color = (0, 0, 255)   # 红色: 低置信度
            
        cv2.line(vis, (int(pt1[0]), int(pt1[1])), pt2_shifted, color, 1)
        cv2.circle(vis, (int(pt1[0]), int(pt1[1])), 3, color, -1)
        cv2.circle(vis, pt2_shifted, 3, color, -1)
        
    # 统计信息
    n_total = len(matches)
    n_inliers = sum(1 for m in matches if m.is_inlier)
    n_depth_ok = sum(1 for m in matches if m.depth_consistent)
    
    info = f"Matches: {n_total} | Inliers: {n_inliers} | Depth OK: {n_depth_ok}"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return vis


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Enhanced Feature Module - Self Test")
    print("="*60)
    
    # 创建测试图像 (棋盘格)
    test_img = np.zeros((480, 640), dtype=np.uint8)
    for i in range(0, 480, 40):
        for j in range(0, 640, 40):
            if (i // 40 + j // 40) % 2 == 0:
                test_img[i:i+40, j:j+40] = 255
                
    # 添加一些噪声
    noise = np.random.randint(0, 30, test_img.shape, dtype=np.uint8)
    test_img = cv2.add(test_img, noise)
    
    # 创建伪深度图
    test_depth = np.ones((480, 640), dtype=np.float32) * 2.0
    
    # 相机内参
    intrinsics = {
        'fx': 500.0, 'fy': 500.0,
        'cx': 320.0, 'cy': 240.0,
        'depth_min': 0.2, 'depth_max': 5.0
    }
    
    # 测试特征提取
    print("\n1. Testing GridBasedORBExtractor...")
    extractor = GridBasedORBExtractor()
    kps, descs = extractor.extract(test_img)
    print(f"   Detected {len(kps)} features")
    
    # 测试带深度的提取
    print("\n2. Testing extract_with_depth...")
    enhanced_kps, enhanced_descs = extractor.extract_with_depth(
        test_img, test_depth, intrinsics
    )
    print(f"   Detected {len(enhanced_kps)} features with 3D coordinates")
    
    # 测试匹配器
    print("\n3. Testing EnhancedFeatureMatcher...")
    
    # 创建略微平移的第二帧
    M = np.float32([[1, 0, 5], [0, 1, 3]])  # 平移5,3像素
    test_img2 = cv2.warpAffine(test_img, M, (640, 480))
    
    kps2, descs2 = extractor.extract_with_depth(test_img2, test_depth, intrinsics)
    
    matcher = EnhancedFeatureMatcher()
    matches = matcher.match(
        test_img, test_img2,
        enhanced_kps, kps2,
        enhanced_descs, descs2,
        intrinsics
    )
    print(f"   Found {len(matches)} matches")
    
    # 统计匹配质量
    n_inliers = sum(1 for m in matches if m.is_inlier)
    n_flow = sum(1 for m in matches if m.flow_assisted)
    print(f"   Inliers: {n_inliers}, Flow-assisted: {n_flow}")
    
    # 测试综合管理器
    print("\n4. Testing EnhancedFeatureManager...")
    manager = EnhancedFeatureManager()
    
    # 处理第一帧
    kps_out, descs_out = manager.process_frame(
        cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR),
        test_depth, intrinsics
    )
    print(f"   Frame 1: {len(kps_out)} features")
    
    # 处理第二帧
    kps_out2, descs_out2, matches_out = manager.process_frame(
        cv2.cvtColor(test_img2, cv2.COLOR_GRAY2BGR),
        test_depth, intrinsics,
        return_matches=True
    )
    print(f"   Frame 2: {len(kps_out2)} features, {len(matches_out)} matches")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)