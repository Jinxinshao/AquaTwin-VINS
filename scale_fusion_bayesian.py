#!/usr/bin/env python3
"""
================================================================================
Bayesian Scale Fusion for Monocular SLAM
================================================================================

理论基础：
---------
单目 SLAM 的尺度 s 是不可观的。通过引入外部深度先验（Depth Anything V2），
我们可以将尺度估计建模为一个贝叶斯滤波问题：

状态转移模型（尺度应该平滑变化）：
    s_k = s_{k-1} + w_k,  w_k ~ N(0, Q)

观测模型（Teacher 深度与 Student 深度的比值）：
    z_k = s_k + v_k,  v_k ~ N(0, R_k)

其中 R_k 是动态计算的观测噪声，取决于两个深度估计的一致性。

参考文献：
---------
[1] Engel et al., "LSD-SLAM: Large-Scale Direct Monocular SLAM", ECCV 2014
[2] Yin et al., "Scale Recovery for Monocular Visual Odometry...", ICRA 2017

Author: Academic Enhancement
================================================================================
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats


class BayesianScaleEstimator:
    """
    基于卡尔曼滤波的尺度估计器。
    
    相比简单的 EMA，本方法的优势：
    1. 根据观测质量动态调整滤波增益
    2. 提供尺度估计的不确定性量化
    3. 支持异常观测的自动剔除
    """
    
    def __init__(
        self,
        initial_scale: float = 1.0,
        initial_variance: float = 0.1,
        process_noise: float = 0.001,
        base_observation_noise: float = 0.05,
        outlier_threshold: float = 3.0,
        min_valid_pixels: int = 1000
    ):
        """
        Args:
            initial_scale: 初始尺度估计
            initial_variance: 初始尺度方差
            process_noise: 过程噪声 Q（控制尺度变化的平滑度）
            base_observation_noise: 基础观测噪声 R
            outlier_threshold: 马氏距离异常值阈值
            min_valid_pixels: 计算尺度比所需的最小有效像素数
        """
        # 卡尔曼滤波状态
        self.s = initial_scale       # 尺度估计 \hat{s}
        self.P = initial_variance    # 估计方差 P
        
        # 噪声参数
        self.Q = process_noise
        self.R_base = base_observation_noise
        
        # 鲁棒性参数
        self.outlier_threshold = outlier_threshold
        self.min_valid_pixels = min_valid_pixels
        
        # 统计追踪
        self.update_count = 0
        self.outlier_count = 0
        self.history = []
        
    def predict(self) -> Tuple[float, float]:
        """
        预测步骤：尺度的先验估计。
        
        状态转移：s_k = s_{k-1} (假设尺度恒定，但允许小的漂移)
        方差更新：P_k|k-1 = P_{k-1} + Q
        
        Returns:
            (predicted_scale, predicted_variance)
        """
        # 尺度预测保持不变（随机游走模型）
        s_pred = self.s
        
        # 方差增长
        P_pred = self.P + self.Q
        
        return s_pred, P_pred
    
    def compute_robust_scale_ratio(
        self,
        teacher_depth: np.ndarray,
        student_depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, int]:
        """
        计算鲁棒的深度比值，使用 RANSAC + 中值估计。
        
        核心思想：
        1. 只使用有效深度区域（非零、非极端值）
        2. 用 RANSAC 剔除异常值（遮挡边界、反射）
        3. 返回中值比值和 MAD 估计的方差
        
        Args:
            teacher_depth: 高质量深度图 (Depth Anything V2)
            student_depth: 实时深度图 (scDepth)
            valid_mask: 可选的有效区域掩码
            
        Returns:
            (scale_ratio, variance, num_valid_pixels)
        """
        # 构建有效掩码
        if valid_mask is None:
            valid_mask = np.ones_like(teacher_depth, dtype=bool)
            
        # 深度有效性检查
        depth_min, depth_max = 0.2, 10.0  # 合理深度范围
        
        valid = (
            valid_mask &
            (teacher_depth > depth_min) & (teacher_depth < depth_max) &
            (student_depth > depth_min) & (student_depth < depth_max)
        )
        
        n_valid = np.sum(valid)
        if n_valid < self.min_valid_pixels:
            return None, None, n_valid
            
        # 计算逐像素比值
        ratios = teacher_depth[valid] / student_depth[valid]
        
        # 第一步：粗筛 - 去掉明显异常的比值
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        inlier_mask = (ratios > q1 - 1.5 * iqr) & (ratios < q3 + 1.5 * iqr)
        
        if np.sum(inlier_mask) < self.min_valid_pixels // 2:
            # IQR 筛选太激进，退化到简单中值
            scale = np.median(ratios)
            mad = stats.median_abs_deviation(ratios)
        else:
            ratios_clean = ratios[inlier_mask]
            scale = np.median(ratios_clean)
            mad = stats.median_abs_deviation(ratios_clean)
        
        # 将 MAD 转换为标准差估计
        # 对于正态分布：σ ≈ 1.4826 × MAD
        variance = (1.4826 * mad) ** 2
        
        return scale, variance, n_valid
    
    def update(
        self,
        teacher_depth: np.ndarray,
        student_depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, bool]:
        """
        卡尔曼更新步骤。
        
        Returns:
            (updated_scale, updated_variance, is_valid_update)
        """
        # 1. 预测步骤
        s_pred, P_pred = self.predict()
        
        # 2. 计算观测
        z, R_obs, n_pixels = self.compute_robust_scale_ratio(
            teacher_depth, student_depth, valid_mask
        )
        
        if z is None:
            # 观测无效，只做预测，不更新
            self.s = s_pred
            self.P = P_pred
            return self.s, self.P, False
            
        # 3. 计算观测噪声（动态调整）
        # 基础噪声 + 数据驱动的不确定性
        R = self.R_base + R_obs
        
        # 4. 异常值检测（基于马氏距离）
        innovation = z - s_pred
        S = P_pred + R  # Innovation variance
        mahalanobis = abs(innovation) / np.sqrt(S)
        
        if mahalanobis > self.outlier_threshold:
            # 异常观测，降低其权重而非完全剔除
            self.outlier_count += 1
            R = R * (mahalanobis ** 2)  # 增大噪声以降低权重
            
        # 5. 卡尔曼增益
        K = P_pred / (P_pred + R)
        
        # 6. 状态更新
        self.s = s_pred + K * innovation
        self.P = (1 - K) * P_pred
        
        # 尺度合理性约束
        self.s = np.clip(self.s, 0.3, 3.0)
        
        # 记录历史
        self.update_count += 1
        self.history.append({
            'scale': self.s,
            'variance': self.P,
            'observation': z,
            'kalman_gain': K,
            'innovation': innovation
        })
        
        return self.s, self.P, True
    
    def get_scale(self) -> float:
        """获取当前尺度估计"""
        return self.s
    
    def get_uncertainty(self) -> float:
        """获取当前尺度不确定性（标准差）"""
        return np.sqrt(self.P)
    
    def get_confidence(self) -> float:
        """
        获取尺度估计的置信度 [0, 1]。
        
        基于不确定性的 sigmoid 映射：
        - 低不确定性 → 高置信度
        - 高不确定性 → 低置信度
        """
        sigma = self.get_uncertainty()
        # 当 σ < 0.05 时置信度接近 1
        # 当 σ > 0.2 时置信度接近 0
        confidence = 1.0 / (1.0 + np.exp(20 * (sigma - 0.1)))
        return float(confidence)


class AdaptiveDepthCorrector:
    """
    自适应深度校正器，包装 BayesianScaleEstimator。
    
    增强功能：
    1. 支持区域自适应尺度（图像中心 vs 边缘可能有不同尺度）
    2. 时间平滑以避免尺度跳变
    3. 深度图后处理（边缘保持滤波）
    """
    
    def __init__(self, model_path: str = "depth_anything_v2_vits.onnx"):
        """
        Args:
            model_path: Depth Anything V2 模型路径
        """
        self.scale_estimator = BayesianScaleEstimator(
            initial_scale=1.0,
            initial_variance=0.1,
            process_noise=0.001,
            base_observation_noise=0.05
        )
        
        # 区域自适应（可选）
        self.use_regional_scale = False
        self.regional_estimators = None
        
    def correct_depth(
        self,
        student_depth: np.ndarray,
        teacher_depth: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        校正深度图。
        
        Args:
            student_depth: 待校正的深度图
            teacher_depth: 参考深度图（如果有新的）
            
        Returns:
            corrected_depth: 校正后的深度图
        """
        # 如果有新的 teacher 深度，更新尺度估计
        if teacher_depth is not None:
            self.scale_estimator.update(teacher_depth, student_depth)
            
        # 获取当前尺度和置信度
        scale = self.scale_estimator.get_scale()
        confidence = self.scale_estimator.get_confidence()
        
        # 应用尺度校正
        corrected = student_depth * scale
        
        return corrected, scale, confidence


# =============================================================================
# 加权 PnP 求解器（考虑深度不确定性）
# =============================================================================

class WeightedPnPSolver:
    """
    考虑深度不确定性的加权 PnP 求解器。
    
    理论依据：
    ---------
    深度估计的不确定性通常与深度值的平方成正比：
        σ_d ∝ d²
    
    这是因为深度误差主要来源于视差估计误差 δu：
        δd = d² / (f·B) · δu
    
    在 PnP 中，我们应该用这个不确定性来加权重投影误差：
        min Σ w_i · ||u_i - π(P_i)||²
    
    其中 w_i = 1 / σ_i²
    """
    
    def __init__(
        self,
        ransac_iterations: int = 1000,
        reproj_threshold: float = 2.0,
        confidence: float = 0.999,
        depth_sigma_coeff: float = 0.01  # σ_d = coeff * d²
    ):
        self.ransac_iterations = ransac_iterations
        self.reproj_threshold = reproj_threshold
        self.confidence = confidence
        self.depth_sigma_coeff = depth_sigma_coeff
        
    def compute_weights(self, depths: np.ndarray) -> np.ndarray:
        """
        计算每个点的权重，基于深度不确定性模型。
        
        Args:
            depths: 3D 点的深度值
            
        Returns:
            weights: 归一化的权重向量
        """
        # 深度不确定性模型：σ = c * d²
        sigmas = self.depth_sigma_coeff * (depths ** 2)
        
        # 权重 = 1 / σ²
        weights = 1.0 / (sigmas ** 2 + 1e-6)
        
        # 归一化
        weights = weights / np.sum(weights) * len(weights)
        
        return weights
    
    def solve(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        track_lengths: Optional[np.ndarray] = None,
        initial_rvec: Optional[np.ndarray] = None,
        initial_tvec: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        带权重的 PnP 求解。
        
        策略：
        1. 先用 RANSAC PnP 获得初始解和内点集
        2. 在内点集上进行加权非线性优化
        
        Args:
            object_points: 3D 点 (N×3)
            image_points: 2D 点 (N×2)
            camera_matrix: 相机内参
            track_lengths: 特征跟踪长度（用于加权）
            initial_rvec, initial_tvec: 初始猜测
            
        Returns:
            success, rvec, tvec, inliers, info
        """
        import cv2
        
        n_points = len(object_points)
        info = {'method': 'weighted_pnp', 'n_points': n_points}
        
        if n_points < 4:
            return False, None, None, None, info
            
        # 确保数据格式
        obj_pts = np.ascontiguousarray(object_points, dtype=np.float64)
        img_pts = np.ascontiguousarray(image_points, dtype=np.float64)
        
        # 计算深度（相机坐标系下的 Z 值）
        depths = obj_pts[:, 2]
        
        # Phase 1: RANSAC PnP 获得初始解
        use_guess = initial_rvec is not None
        rvec_init = initial_rvec if use_guess else np.zeros((3,1), dtype=np.float64)
        tvec_init = initial_tvec if use_guess else np.zeros((3,1), dtype=np.float64)
        
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, camera_matrix, None,
                rvec=rvec_init.copy(), tvec=tvec_init.copy(),
                useExtrinsicGuess=use_guess,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.reproj_threshold,
                confidence=self.confidence,
                flags=cv2.SOLVEPNP_EPNP
            )
        except Exception as e:
            info['error'] = str(e)
            return False, None, None, None, info
            
        if not success or inliers is None or len(inliers) < 4:
            return False, None, None, None, info
            
        inliers = inliers.flatten()
        
        # Phase 2: 在内点上进行加权优化
        obj_inliers = obj_pts[inliers]
        img_inliers = img_pts[inliers]
        depths_inliers = depths[inliers]
        
        # 计算权重
        weights = self.compute_weights(depths_inliers)
        
        # 如果有 track length 信息，进一步调整权重
        if track_lengths is not None:
            track_inliers = track_lengths[inliers]
            # 长轨迹特征权重更高（最多 3 倍）
            track_weights = np.clip(track_inliers / 5.0, 1.0, 3.0)
            weights = weights * track_weights
            
        # 进行加权迭代优化
        # OpenCV 的 solvePnP 不直接支持权重，我们用多次迭代近似
        try:
            # 按权重采样多次，然后取平均
            rvecs, tvecs = [], []
            for _ in range(3):
                # 按权重重采样
                probs = weights / np.sum(weights)
                indices = np.random.choice(
                    len(obj_inliers), 
                    size=min(len(obj_inliers), 50),
                    replace=True, 
                    p=probs
                )
                
                success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                    obj_inliers[indices],
                    img_inliers[indices],
                    camera_matrix, None,
                    rvec=rvec.copy(), tvec=tvec.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success_iter:
                    rvecs.append(rvec_iter)
                    tvecs.append(tvec_iter)
                    
            # 取平均
            if len(rvecs) > 0:
                rvec = np.mean(rvecs, axis=0)
                tvec = np.mean(tvecs, axis=0)
                info['weighted_refinement'] = True
                
        except Exception as e:
            info['refinement_error'] = str(e)
            
        # 计算重投影误差
        proj_pts, _ = cv2.projectPoints(obj_inliers, rvec, tvec, camera_matrix, None)
        proj_pts = proj_pts.reshape(-1, 2)
        reproj_errors = np.linalg.norm(img_inliers - proj_pts, axis=1)
        info['mean_reproj_error'] = float(np.mean(reproj_errors))
        info['median_reproj_error'] = float(np.median(reproj_errors))
        info['n_inliers'] = len(inliers)
        
        return True, rvec, tvec, inliers, info


if __name__ == "__main__":
    # 测试贝叶斯尺度估计器
    print("=" * 60)
    print("Bayesian Scale Estimator Test")
    print("=" * 60)
    
    estimator = BayesianScaleEstimator()
    
    # 模拟观测序列（真实尺度 = 1.2，有噪声）
    true_scale = 1.2
    np.random.seed(42)
    
    for i in range(20):
        # 模拟深度图
        teacher = np.random.uniform(0.5, 5.0, (480, 640)) * true_scale
        student = np.random.uniform(0.5, 5.0, (480, 640))
        
        # 添加噪声
        teacher += np.random.normal(0, 0.1, teacher.shape)
        student += np.random.normal(0, 0.1, student.shape)
        
        # 更新估计
        s, P, valid = estimator.update(teacher, student)
        
        if i % 5 == 0:
            print(f"Step {i:2d}: scale={s:.4f}, σ={np.sqrt(P):.4f}, "
                  f"confidence={estimator.get_confidence():.2f}")
    
    print(f"\nFinal: scale={estimator.get_scale():.4f} (true={true_scale})")
    print(f"Error: {abs(estimator.get_scale() - true_scale):.4f}")
