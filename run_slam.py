#!/usr/bin/env python3
"""
================================================================================
PiSLAM: A Lightweight Visual SLAM System for Edge Devices
Main System Module (Optimized for Hailo NPU)
- Academic UI Workspace (OpenCV only, no extra deps)
- Save/Quit buttons + keep hotkeys (S/Q)
================================================================================
"""

import numpy as np
import cv2
import open3d as o3d
import time
import os
import sys
import yaml
import argparse
import shutil
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# ==============================================================================
# 1. 导入增强版模块 (修复 ImportError)
# ==============================================================================
from visual_odometry_enhanced import EnhancedVisualOdometry as VisualOdometry
# ✅ 从增强版直接导入，避免兼容层带来的命名空间混淆
from pose_graph_enhanced import PoseGraphOptimizer, Keyframe

# 1. 导入 IMU 接收器
from imu_receiver import IMUReceiver
from imu_preintegrator import IMUPreintegrator
# [新增] 导入水下增强模块
from underwater_enhancer import UnderwaterEnhancer

# ==============================================================================
# 2. Hailo SDK 导入检查
# ==============================================================================
HAILO_AVAILABLE = False
try:
    # from hailo_platform import (HEF, VDevice, ConfigureParams, InferVStreams,
    #                             InputVStreamParams, OutputVStreamParams,
    #                             FormatType, HailoStreamInterface)

    from hailo_platform import (HEF, VDevice, ConfigureParams, InferVStreams,
                                InputVStreamParams, OutputVStreamParams,
                                FormatType, HailoStreamInterface, 
                                HailoSchedulingAlgorithm) # <--- 新增这项

    HAILO_AVAILABLE = True
except ImportError:
    print("⚠️  Hailo runtime not available. NPU features disabled.")




class DepthEstimator:
    """深度估计抽象基类"""
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HailoDepthEstimator(DepthEstimator):
    """
    使用 Hailo NPU 进行深度估计
    ✅ 修复：接受共享的 vdevice 实例，并正确管理推理上下文
    """

    def __init__(self, hef_path: str, vdevice,
                 input_width: int = 512,
                 input_height: int = 160,
                 depth_scale: float = 10.0):

        self.input_width = input_width
        self.input_height = input_height
        self.depth_scale = depth_scale
        self.hailo_available = HAILO_AVAILABLE

        if not self.hailo_available:
            print("⚠️  Hailo SDK not installed. Using dummy depth.")
            return

        if vdevice is None:
            raise RuntimeError("Shared VDevice is required for HailoDepthEstimator")

        print(f"🔌 [Depth] Initializing with shared VDevice using {hef_path}")

        # 加载模型
        self.hef = HEF(hef_path)

        # 使用传入的共享 vdevice
        self.target = vdevice

        # 配置参数
        self.configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )

        # 配置网络组
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_params = self.network_group.create_params()

        # 创建流参数
        self.input_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # ======================================================================
        # 🔧 关键修复：手动激活并保持上下文 (Persistent Context)
        # ======================================================================
        self._network_context = self.network_group.activate(self.network_params)
        self._network_context.__enter__()

        self._infer_context = InferVStreams(
            self.network_group, self.input_params, self.output_params
        )
        self.infer_pipeline = self._infer_context.__enter__()

        print("✅ [Depth] Hailo NPU initialized (Network Activated)")

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        if not self.hailo_available:
            h, w = rgb.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 2.0

        # 预处理
        input_frame = cv2.resize(rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(input_frame, axis=0)

        # 推理
        result = self.infer_pipeline.infer(input_data)
        raw_depth = list(result.values())[0].squeeze()

        # 转换深度
        abs_depth = np.abs(raw_depth)
        depth_metric = self.depth_scale / (abs_depth + 1e-5)
        return depth_metric

    def __del__(self):
        # 析构时尝试退出上下文（不释放 shared_vdevice）
        if hasattr(self, '_infer_context') and self._infer_context:
            try:
                self._infer_context.__exit__(None, None, None)
            except Exception:
                pass
        if hasattr(self, '_network_context') and self._network_context:
            try:
                self._network_context.__exit__(None, None, None)
            except Exception:
                pass


class DatasetDepthLoader(DepthEstimator):
    """从数据集加载预计算的深度图"""
    def __init__(self, depth_dir: str, depth_scale: float = 5000.0, depth_format: str = 'png'):
        self.depth_dir = Path(depth_dir)
        self.depth_scale = depth_scale
        self.depth_format = depth_format
        self.depth_files = sorted(self.depth_dir.glob(f'*.{depth_format}'))
        self.current_idx = 0
        print(f"📁 Found {len(self.depth_files)} depth images")

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        if self.current_idx >= len(self.depth_files):
            raise StopIteration("No more depth images")

        depth_path = self.depth_files[self.current_idx]
        self.current_idx += 1

        if self.depth_format == 'png':
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth = depth_raw.astype(np.float32) / self.depth_scale
        elif self.depth_format == 'exr':
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            raise ValueError(f"Unknown depth format: {self.depth_format}")

        return depth


class KeyframeSelector:
    """关键帧选择器"""
    def __init__(self, config_path: Optional[str] = None):
        self.min_time_interval = 1.0
        self.min_translation = 0.1
        self.min_rotation = 0.1
        self.min_covisibility = 0.5
        self.last_keyframe_time = 0.0
        self.last_keyframe_pose = np.eye(4)

    def should_create_keyframe(self, current_pose, current_time, num_tracked_features=100, num_covisible_features=50):
        time_elapsed = current_time - self.last_keyframe_time
        if time_elapsed < self.min_time_interval:
            return False

        translation = np.linalg.norm(current_pose[:3, 3] - self.last_keyframe_pose[:3, 3])
        R_rel = self.last_keyframe_pose[:3, :3].T @ current_pose[:3, :3]
        rotation_angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))

        covisibility = num_covisible_features / num_tracked_features if num_tracked_features > 0 else 0.0

        if (translation >= self.min_translation or rotation_angle >= self.min_rotation or covisibility < self.min_covisibility):
            return True
        return False

    def update(self, pose, timestamp):
        self.last_keyframe_pose = pose.copy()
        self.last_keyframe_time = timestamp


class PiSLAM:
    """
    PiSLAM 系统主类
    ✅ 修复：实现 VDevice 单例模式，统一管理 NPU 资源
    ✅ 新增：Academic UI Workspace（OpenCV 按钮 + 顶栏）
    """

    UI_WINDOW_NAME = "PiSLAM Academic Workspace"

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self._load_config()

        print("\n" + "=" * 60)
        print("🚀 Initializing PiSLAM System")
        print("=" * 60)

        # [新增] 初始化 IMU 接收器
        print("🔌 [System] Connecting to IMU Stream...")
        self.imu_receiver = IMUReceiver(port=9999)  # 确保端口和 run_pislam.sh 一致
        self.imu_receiver.start()

        # 初始化 buffer / phase
        self.initialization_buffer: List = []
        self.init_phase = True

        # [新增] 初始化 IMU 预积分器
        if 'imu' in self.config:
            self.imu_preintegrator = IMUPreintegrator(self.config['imu'])
            print("✅ IMU Preintegrator Initialized")
        else:
            print("⚠️ No IMU config found, tight coupling disabled.")
            self.imu_preintegrator = None

        # ==========================================================
        # 1. 创建全局共享 VDevice
        # ==========================================================
        self.shared_vdevice = None
        if HAILO_AVAILABLE:
            try:
                # print("🔌 [System] Creating shared Hailo VDevice...")
                # self.shared_vdevice = VDevice()
                # print("✅ [System] Shared VDevice created")

                print("🔌 [System] Creating shared Hailo VDevice (Scheduler Mode)...")
                
                # 🟢 [修复开始] 启用 Round-Robin 调度器
                params = VDevice.create_params()
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                self.shared_vdevice = VDevice(params=params)
                # 🟢 [修复结束]
                
                print("✅ [System] Shared VDevice created")
            except Exception as e:
                print(f"❌ [System] Failed to create VDevice: {e}")
                print("   NPU features will be disabled.")
        
        # ==========================================================
        # [新增] 初始化水下增强器
        # ==========================================================
        self.enhancer = None
        # 假设 config 中有配置，或者我们硬编码路径
        enhancer_hef = "implicit_physics_npu.hef" 
        if self.shared_vdevice and os.path.exists(enhancer_hef):
            try:
                self.enhancer = UnderwaterEnhancer(enhancer_hef, self.shared_vdevice)
                print("🌊 [System] Underwater Enhancer Attached.")
            except Exception as e:
                print(f"⚠️ Failed to load enhancer: {e}")
        else:
            print("⚠️ Enhancer HEF not found or NPU unavailable. Running in RAW mode.")

        # ==========================================================

        # ==========================================================
        # 2. 初始化模块
        # ==========================================================
        self.visual_odometry = VisualOdometry(config_path)

        print("\n🔗 Initializing Pose Graph (CPU Mode for Stability)...")
        self.pose_graph = PoseGraphOptimizer(
            config_path,
            use_enhanced_lc=False  # ✅ 必须设为 False，避免 NPU 资源冲突
        )

        self.keyframe_selector = KeyframeSelector(config_path)
        self.depth_estimator: Optional[DepthEstimator] = None

        # 状态
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0.0

        self.output_dir = os.path.expanduser(self.save_directory)

        # --------------------------
        # UI State (OpenCV only)
        # --------------------------
        self._ui_window_ready = False
        self._ui_buttons: Dict[str, Tuple[int, int, int, int]] = {}
        self._ui_mouse_xy = (-1, -1)
        self._ui_hover: Optional[str] = None
        self._ui_pressed: Optional[str] = None
        self._ui_actions: set = set()
        self._ui_last_status_msg = ""
        self._ui_last_status_ts = 0.0
        self._ui_fps = 0.0

        # IMU frame anchor (epoch seconds)
        self.last_frame_time_epoch = -1.0

    # ==============================================================================
    # UI Helpers (OpenCV mouse-driven buttons)
    # ==============================================================================
    def _ensure_ui_window(self):
        if self._ui_window_ready:
            return
        cv2.namedWindow(self.UI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.UI_WINDOW_NAME, self._on_mouse_event)
        self._ui_window_ready = True

    def _hit_test_button(self, x: int, y: int) -> Optional[str]:
        for name, (x1, y1, x2, y2) in self._ui_buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def _on_mouse_event(self, event, x, y, flags, userdata=None):
        self._ui_mouse_xy = (x, y)
        self._ui_hover = self._hit_test_button(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._ui_pressed = self._hit_test_button(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            hit = self._hit_test_button(x, y)
            if hit is not None and hit == self._ui_pressed:
                self._ui_actions.add(hit)  # 'save' or 'quit'
            self._ui_pressed = None

    def _consume_ui_actions(self) -> set:
        actions = set(self._ui_actions)
        self._ui_actions.clear()
        return actions

    def _ui_set_status(self, msg: str, duration_s: float = 2.0):
        self._ui_last_status_msg = msg
        self._ui_last_status_ts = time.time() + duration_s

    # ==============================================================================
    # Drawing helpers (Academic theme)
    # ==============================================================================
    @staticmethod
    def _draw_panel_header(img: np.ndarray, text: str, bg=(30, 30, 30), fg=(220, 220, 220)):
        """绘制带背景的标题栏（panel 内部）"""
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 30), bg, -1)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, fg, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _draw_button(img: np.ndarray, rect: Tuple[int, int, int, int], label: str,
                     hotkey: str, hovered: bool, pressed: bool,
                     fill=(45, 45, 45), fill_hover=(60, 60, 60), fill_pressed=(25, 25, 25),
                     border=(130, 130, 130), accent=(0, 180, 255), text=(235, 235, 235)):
        x1, y1, x2, y2 = rect
        bg = fill_pressed if pressed else (fill_hover if hovered else fill)
        cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)

        # accent strip
        cv2.rectangle(img, (x1, y1), (x1 + 4, y2), accent, -1)

        # label
        cy = (y1 + y2) // 2
        cv2.putText(img, label, (x1 + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text, 1, cv2.LINE_AA)

        # hotkey badge
        badge = f"[{hotkey.upper()}]"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bx2 = x2 - 10
        bx1 = bx2 - tw - 10
        by1 = cy - 12
        by2 = cy + 10
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (90, 90, 90), 1)
        cv2.putText(img, badge, (bx1 + 5, by2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_trajectory_widget(self, size=(240, 320)):
        """绘制简易的 2D 轨迹小地图"""
        canvas = np.ones((size[0], size[1], 3), dtype=np.uint8) * 35  # 深灰背景

        poses = [kf.pose for kf in self.pose_graph.keyframes.values()]
        if len(poses) < 2 or not hasattr(self.visual_odometry, 'current_pose') or self.visual_odometry.current_pose is None:
            return self._draw_panel_header(canvas, "Trajectory (Top-View)")

        pos_list = [p[:3, 3] for p in poses]
        pos_arr = np.array(pos_list)
        curr_pos = self.visual_odometry.current_pose[:3, 3]

        min_x, max_x = float(np.min(pos_arr[:, 0])), float(np.max(pos_arr[:, 0]))
        min_z, max_z = float(np.min(pos_arr[:, 2])), float(np.max(pos_arr[:, 2]))

        # 自动缩放
        margin = 1.0
        range_x = max(max_x - min_x, margin)
        range_z = max(max_z - min_z, margin)
        scale = min((size[1] - 40) / range_x, (size[0] - 40) / range_z)

        # 以第一帧为原点（稳定）
        ox, oz = float(pos_arr[0, 0]), float(pos_arr[0, 2])
        cx, cy = size[1] // 2, size[0] // 2

        # 背景网格（更像学术软件）
        for gx in range(20, size[1], 40):
            cv2.line(canvas, (gx, 30), (gx, size[0] - 1), (50, 50, 50), 1)
        for gy in range(50, size[0], 40):
            cv2.line(canvas, (0, gy), (size[1] - 1, gy), (50, 50, 50), 1)

        # 轨迹点
        for x, y, z in pos_list:
            u = int((x - ox) * scale + cx)
            v = int((z - oz) * scale + cy)
            if 0 <= u < size[1] and 30 <= v < size[0]:
                cv2.circle(canvas, (u, v), 1, (200, 200, 200), -1)

        # 当前点
        cur_u = int((curr_pos[0] - ox) * scale + cx)
        cur_v = int((curr_pos[2] - oz) * scale + cy)
        if 0 <= cur_u < size[1] and 30 <= cur_v < size[0]:
            cv2.circle(canvas, (cur_u, cur_v), 4, (0, 0, 255), -1)
            cv2.circle(canvas, (cur_u, cur_v), 8, (0, 0, 255), 1)

        return self._draw_panel_header(canvas, "Trajectory (Top-View)")

    # ==============================================================================
    # Config / init
    # ==============================================================================
    def _load_config(self):
        self.save_directory = "~/UW_pislam_output"
        self.visualization = True
        self.config = {}

        if self.config_path is not None and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            output_cfg = self.config.get('output', {})
            self.save_directory = output_cfg.get('save_directory', self.save_directory)
            self.visualization = output_cfg.get('visualization', self.visualization)
        else:
            print(f"⚠️ Config file not found: {self.config_path}, using defaults.")

    def initialize_depth_estimator(self, mode: str, **kwargs):
        """初始化深度估计，注入共享 VDevice"""
        if mode == 'hailo':
            hef_path = kwargs.get('hef_path', 'scdepthv3.hef')
            if self.shared_vdevice is None:
                raise RuntimeError("Cannot use Hailo mode: VDevice creation failed.")
            self.depth_estimator = HailoDepthEstimator(hef_path, vdevice=self.shared_vdevice)

        elif mode == 'dataset':
            depth_dir = kwargs.get('depth_dir', './depth')
            self.depth_estimator = DatasetDepthLoader(depth_dir)
        else:
            raise ValueError(f"Unknown depth mode: {mode}")

    # ==============================================================================
    # Core pipeline
    # ==============================================================================
    def process_frame(self, rgb: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        if timestamp is None:
            timestamp = time.time() - self.start_time  # relative seconds since start

        self.frame_count += 1

        # ==========================================================
        # [核心修改] 图像增强流水线
        # ==========================================================
        original_rgb = rgb.copy() # 备份原图用于对比显示
        enhanced_rgb = rgb        # 默认等于原图
        # enhanced_rgb = self.enhancer.enhance(rgb)

        if self.enhancer:
            # 执行增强
            # [调试] 强制画个框证明代码走到这里了
            # cv2.rectangle(enhanced_rgb, (10, 10), (50, 50), (255, 0, 255), -1)
            t0 = time.time()
            enhanced_rgb = self.enhancer.enhance(rgb)
            dt_ms = (time.time() - t0) * 1000
            
            # [验证代码] 计算平均像素差异 (Mean Absolute Difference)
            diff = np.mean(np.abs(enhanced_rgb.astype(float) - original_rgb.astype(float)))
            
            # 只有当差异 > 0 时，才说明增强生效了
            print(f"\r[Enhancer] Latency: {dt_ms:.1f}ms | Pixel Diff: {diff:.4f}  ", end="")
            
            # 如果 Diff 始终是 0.0000，说明模型输出无效或没生效

        # 1) depth
        if self.depth_estimator is None:
            raise RuntimeError("Depth estimator not initialized")

        try:
            depth = self.depth_estimator.estimate(enhanced_rgb)
        except StopIteration:
            return {'success': False, 'reason': 'end_of_sequence'}

        if depth.shape[:2] != enhanced_rgb.shape[:2]:
            depth = cv2.resize(depth, (enhanced_rgb.shape[1], enhanced_rgb.shape[0]))

        # 2) init warm-up
        if self.init_phase:
            if self.frame_count < 5:
                return {
                    'success': False,
                    'reason': 'warming_up',
                    'frame': self.frame_count,
                    'timestamp': timestamp,
                    'pose': np.eye(4),
                    'position': np.zeros(3),
                    'num_features': 0,
                    'inliers': 0,
                    'is_keyframe': False,
                    'loop_detected': False,
                    'depth': depth,
                    'current_keypoints': [],
                    # [新增] 将原图和增强图放入 result 以便 UI 显示
                    'raw_image': original_rgb,
                    'enhanced_image': enhanced_rgb
                    #'current_keypoints': []
                }
            else:
                self.init_phase = False

        # 3) imu orientation prior
        imu_quat = self.imu_receiver.get_quaternion()  # [w, x, y, z]

        # 4) visual odometry
        odom_result = self.visual_odometry.process_frame(enhanced_rgb, depth, imu_orientation=imu_quat)

        result = {
            'success': odom_result.success,
            'frame': self.frame_count,
            'timestamp': timestamp,
            'pose': odom_result.pose,
            'position': odom_result.pose[:3, 3],
            'num_features': odom_result.num_features,
            'inliers': odom_result.inliers,
            'is_keyframe': False,
            'loop_detected': False,
            'depth': depth,
            'current_keypoints': getattr(odom_result, 'current_keypoints', None),
            'raw_image': original_rgb,
            'enhanced_image': enhanced_rgb

        }

        if not odom_result.success:
            return result

        # 5) keyframe / backend
        if self.keyframe_selector.should_create_keyframe(
            odom_result.pose, timestamp, odom_result.num_features
        ):
            prev_kf_pose = self.keyframe_selector.last_keyframe_pose
            odom_transform = np.linalg.inv(prev_kf_pose) @ odom_result.pose

            current_scale = 1.0
            if hasattr(self.visual_odometry, 'depth_corrector') and self.visual_odometry.depth_corrector:
                current_scale = self.visual_odometry.depth_corrector.get_scale_factor()

            pim = None
            if self.imu_preintegrator is not None:
                pim = self.imu_preintegrator.get_factor()
                self.imu_preintegrator.reset()

            kf_id, loop_detected = self.pose_graph.add_keyframe(
                rgb=enhanced_rgb,
                depth=depth,
                pose=odom_result.pose,
                timestamp=timestamp,
                odometry_transform=odom_transform,
                scale_factor=current_scale,
                pim=pim
            )

            if self.imu_preintegrator is not None:
                latest_bias = self.pose_graph.get_latest_bias()
                if latest_bias is not None:
                    self.imu_preintegrator.update_bias(latest_bias)

            self.keyframe_selector.update(odom_result.pose, timestamp)
            result['is_keyframe'] = True
            result['keyframe_id'] = kf_id
            result['loop_detected'] = loop_detected

        return result

    # ==============================================================================
    # Run modes
    # ==============================================================================
    def _prepare_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'keyframes'), exist_ok=True)

    def run_live(self, camera_id: int = 0):
        print("\n" + "=" * 60)
        print("PiSLAM - Live Mode (Academic Workspace UI)")
        print("=" * 60)

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        self._prepare_output_dir()
        self.start_time = time.time()
        self.is_running = True

        # epoch anchor for IMU pull
        self.last_frame_time_epoch = time.time()

        try:
            while self.is_running:
                loop_start = time.time()

                # 1) frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera stream ended.")
                    break

                curr_time_epoch = time.time()

                # 2) IMU feed (epoch domain)
                if self.imu_preintegrator is not None:
                    imu_packets = self.imu_receiver.get_imu_data_since(self.last_frame_time_epoch)
                    for ts, acc, gyro in imu_packets:
                        self.imu_preintegrator.add_imu_measurement(ts, acc, gyro)
                    self.last_frame_time_epoch = curr_time_epoch

                # 3) process (relative timestamp domain)
                timestamp_rel = curr_time_epoch - self.start_time
                result = self.process_frame(frame, timestamp=timestamp_rel)

                # FPS
                dt = max(time.time() - loop_start, 1e-6)
                self._ui_fps = 1.0 / dt

                # 4) visualization + input
                if self.visualization:
                    try:
                        depth_map = result.get('depth', None)
                        if depth_map is None:
                            h, w = frame.shape[:2]
                            depth_map = np.zeros((h, w), dtype=np.float32)

                        self._visualize_frame(frame, depth_map, result)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self._ui_set_status("Quit requested (hotkey).")
                            self.is_running = False
                        elif key == ord('s'):
                            self._ui_set_status("Saving results (hotkey)...")
                            self._save_results()

                        # buttons
                        for act in self._consume_ui_actions():
                            if act == 'quit':
                                self._ui_set_status("Quit requested (button).")
                                self.is_running = False
                            elif act == 'save':
                                self._ui_set_status("Saving results (button)...")
                                self._save_results()

                    except cv2.error as e:
                        print(f"\n⚠️ [GUI Error] Disabling visualization: {e}")
                        self.visualization = False

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Runtime Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
            
            # [新增] 关闭增强器
            if self.enhancer:
                self.enhancer.close()

            cv2.destroyAllWindows()

            print("💾 Saving final trajectory...")
            self._save_results()

            self.shared_vdevice = None
            print("👋 System shutdown complete.")

    def run_dataset(self, rgb_dir: str, depth_dir: Optional[str] = None):
        """
        Dataset mode runner
        - Uses existing process_frame() interface.
        - Buttons + hotkeys still work if visualization enabled.
        """
        print("\n" + "=" * 60)
        print("PiSLAM - Dataset Mode (Academic Workspace UI)")
        print("=" * 60)

        rgb_path = Path(rgb_dir)
        if not rgb_path.exists():
            raise RuntimeError(f"RGB dir not found: {rgb_dir}")

        # common image extensions
        rgb_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            rgb_files.extend(sorted(rgb_path.glob(ext)))
        rgb_files = sorted(rgb_files)

        if len(rgb_files) == 0:
            raise RuntimeError(f"No RGB images found in: {rgb_dir}")

        self._prepare_output_dir()
        self.start_time = time.time()
        self.is_running = True
        self.last_frame_time_epoch = time.time()

        # dataset timestamp: use index / fps-like
        fps_assumed = 10.0
        t0 = 0.0

        try:
            for idx, img_path in enumerate(rgb_files):
                if not self.is_running:
                    break

                loop_start = time.time()

                frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"⚠️ Failed to read image: {img_path}")
                    continue

                # IMU feed (still attempt if receiver provides, epoch domain)
                curr_time_epoch = time.time()
                if self.imu_preintegrator is not None:
                    imu_packets = self.imu_receiver.get_imu_data_since(self.last_frame_time_epoch)
                    for ts, acc, gyro in imu_packets:
                        self.imu_preintegrator.add_imu_measurement(ts, acc, gyro)
                    self.last_frame_time_epoch = curr_time_epoch

                timestamp_rel = t0 + idx / fps_assumed
                result = self.process_frame(frame, timestamp=timestamp_rel)

                dt = max(time.time() - loop_start, 1e-6)
                self._ui_fps = 1.0 / dt

                if self.visualization:
                    try:
                        depth_map = result.get('depth', None)
                        if depth_map is None:
                            h, w = frame.shape[:2]
                            depth_map = np.zeros((h, w), dtype=np.float32)

                        self._visualize_frame(frame, depth_map, result)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self._ui_set_status("Quit requested (hotkey).")
                            self.is_running = False
                        elif key == ord('s'):
                            self._ui_set_status("Saving results (hotkey)...")
                            self._save_results()

                        for act in self._consume_ui_actions():
                            if act == 'quit':
                                self._ui_set_status("Quit requested (button).")
                                self.is_running = False
                            elif act == 'save':
                                self._ui_set_status("Saving results (button)...")
                                self._save_results()

                    except cv2.error as e:
                        print(f"\n⚠️ [GUI Error] Disabling visualization: {e}")
                        self.visualization = False

                if not result.get('success', True) and result.get('reason') == 'end_of_sequence':
                    break

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            cv2.destroyAllWindows()
            print("💾 Saving final trajectory...")
            self._save_results()

    # ==============================================================================
    # Academic UI Workspace (2x2 + top bar buttons)
    # ==============================================================================
    # def _visualize_frame(self, frame: np.ndarray, depth: np.ndarray, result: Dict):
    #     """
    #     学术级仪表盘可视化 (Workspace Layout)
    #     [Top Bar]: Title + Buttons (Save/Quit) + FPS + short status toast
    #     [Row 1]: RGB+Features | Depth Heatmap
    #     [Row 2]: Trajectory   | System Panel
    #     """
    #     self._ensure_ui_window()

    #     H, W = frame.shape[:2]
    #     bottom_h = 240
    #     topbar_h = 44

    #     # ---------------------------------------------------------
    #     # Panel A: RGB + features
    #     # ---------------------------------------------------------
    #     vis_rgb = frame.copy()

    #     kpts_to_draw = result.get('current_keypoints')
    #     if kpts_to_draw is None and hasattr(self.visual_odometry, 'prev_kpts'):
    #         kpts_to_draw = self.visual_odometry.prev_kpts

    #     if kpts_to_draw is not None:
    #         color = (0, 255, 0) if result.get('success', False) else (0, 0, 255)
    #         for i, pt in enumerate(kpts_to_draw):
    #             if i % 2 == 0:
    #                 cv2.circle(vis_rgb, (int(pt[0]), int(pt[1])), 2, color, -1)
    #         cv2.putText(vis_rgb, f"Feats: {len(kpts_to_draw)}", (10, 55),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #     if result.get('loop_detected', False):
    #         cv2.rectangle(vis_rgb, (0, 0), (W - 1, H - 1), (0, 0, 255), 4)
    #         cv2.putText(vis_rgb, "LOOP DETECTED!", (W // 2 - 140, H // 2),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    #     vis_rgb = self._draw_panel_header(vis_rgb, "Input RGB + Feature Tracking (XFeat)")

    #     # ---------------------------------------------------------
    #     # Panel B: Depth heatmap
    #     # ---------------------------------------------------------
    #     if depth is not None:
    #         depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    #         depth_norm = depth_norm.astype(np.uint8)
    #         vis_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    #     else:
    #         vis_depth = np.zeros_like(vis_rgb)
    #         cv2.putText(vis_depth, "Initializing depth...", (W // 2 - 120, H // 2),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    #     vis_depth = self._draw_panel_header(vis_depth, "Dense Depth Map (SC-DepthV3 @ NPU)")

    #     # ensure sizes
    #     if vis_depth.shape[:2] != vis_rgb.shape[:2]:
    #         vis_depth = cv2.resize(vis_depth, (W, H))

    #     # ---------------------------------------------------------
    #     # Panel C: Trajectory widget
    #     # ---------------------------------------------------------
    #     traj_map = self._draw_trajectory_widget(size=(bottom_h, W))

    #     # ---------------------------------------------------------
    #     # Panel D: Status panel
    #     # ---------------------------------------------------------
    #     status_panel = np.zeros((bottom_h, W, 3), dtype=np.uint8)
    #     status_panel[:] = (30, 30, 30)

    #     pos = result.get('position', np.zeros(3))
    #     backend_str = "Hailo-8L (Depth) + CPU (VO)" if self.depth_estimator is not None else "CPU"
    #     state_str = "RUNNING" if self.is_running else "STOPPED"

    #     lines = [
    #         f"System Status: {state_str}",
    #         f"Frame Index:   {result.get('frame', 0)}",
    #         f"Time (rel):    {result.get('timestamp', 0.0):.2f} s",
    #         f"FPS:           {self._ui_fps:.1f}",
    #         f"Keyframes:     {len(self.pose_graph.keyframes)}",
    #         "-" * 30,
    #         f"Position X:    {pos[0]:.3f} m",
    #         f"Position Y:    {pos[1]:.3f} m",
    #         f"Position Z:    {pos[2]:.3f} m",
    #         "-" * 30,
    #         f"Backend:       {backend_str}",
    #         "Controls:      Save(S)  Quit(Q)"
    #     ]

    #     y_start = 55
    #     for i, line in enumerate(lines):
    #         color = (0, 255, 0) if "RUNNING" in line else (200, 200, 200)
    #         if "Backend" in line:
    #             color = (255, 191, 0)
    #         if "FPS" in line:
    #             color = (170, 220, 255)
    #         cv2.putText(status_panel, line, (24, y_start + i * 18),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    #     status_panel = self._draw_panel_header(status_panel, "System Statistics & Logs")

    #     # ---------------------------------------------------------
    #     # Compose 2x2
    #     # ---------------------------------------------------------
    #     top_row = np.hstack((vis_rgb, vis_depth))
    #     bottom_row = np.hstack((traj_map, status_panel))
    #     workspace = np.vstack((top_row, bottom_row))

    #     # ---------------------------------------------------------
    #     # Top bar with buttons
    #     # ---------------------------------------------------------
    #     full_w = workspace.shape[1]
    #     topbar = np.zeros((topbar_h, full_w, 3), dtype=np.uint8)
    #     topbar[:] = (22, 22, 22)

    #     # Title
    #     cv2.putText(topbar, "PiSLAM Academic Workspace", (14, 28),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    #     # Subtitle / hint
    #     hint = "Mouse buttons: Save / Quit   |   Hotkeys: S / Q"
    #     cv2.putText(topbar, hint, (14, 42),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    #     # Status toast (short-lived)
    #     now = time.time()
    #     if self._ui_last_status_msg and now <= self._ui_last_status_ts:
    #         msg = self._ui_last_status_msg
    #         (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    #         x2 = full_w - 20
    #         x1 = x2 - tw - 22
    #         y1, y2 = 8, 36
    #         cv2.rectangle(topbar, (x1, y1), (x2, y2), (35, 35, 35), -1)
    #         cv2.rectangle(topbar, (x1, y1), (x2, y2), (90, 90, 90), 1)
    #         cv2.putText(topbar, msg, (x1 + 10, y2 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    #     # Buttons (right side)
    #     btn_w, btn_h = 160, 30
    #     gap = 10
    #     yb1 = 7
    #     yb2 = yb1 + btn_h

    #     quit_rect = (full_w - 14 - btn_w, yb1, full_w - 14, yb2)
    #     save_rect = (quit_rect[0] - gap - btn_w, yb1, quit_rect[0] - gap, yb2)

    #     # Update button registry (absolute coords in final image)
    #     self._ui_buttons = {
    #         'save': save_rect,
    #         'quit': quit_rect
    #     }

    #     # Draw buttons with hover/press feedback
    #     self._draw_button(
    #         topbar, save_rect, "Save Results", "S",
    #         hovered=(self._ui_hover == 'save'),
    #         pressed=(self._ui_pressed == 'save'),
    #         accent=(0, 200, 140)
    #     )
    #     self._draw_button(
    #         topbar, quit_rect, "Quit", "Q",
    #         hovered=(self._ui_hover == 'quit'),
    #         pressed=(self._ui_pressed == 'quit'),
    #         accent=(0, 0, 255)
    #     )

    #     final_dashboard = np.vstack((topbar, workspace))

    #     cv2.imshow(self.UI_WINDOW_NAME, final_dashboard)

    # def _visualize_frame(self, frame: np.ndarray, depth: np.ndarray, result: Dict):
    #     """
    #     学术级仪表盘可视化 (UW-SLAM 专用 3-Panel Layout)
    #     布局策略:
    #     [Row 1]: 原始浑浊图像 (Raw) | 物理增强图像 (Enhanced + Feats) | 深度热力图 (Depth)
    #     [Row 2]: 实时轨迹 (Trajectory, 占50%) | 系统状态面板 (Status, 占50%)
    #     """
    #     self._ensure_ui_window()

    #     # =========================================================
    #     # 1. 数据准备与动态尺寸计算
    #     # =========================================================
    #     src_h, src_w = frame.shape[:2]
        
    #     # [学术细节] 为了在屏幕上横向放下3张图，我们统一设定一个合理的显示高度
    #     # 例如高度定为 360px 或 400px，宽度按比例自适应
    #     display_h = 360 
    #     display_w = int(display_h * (src_w / src_h))
        
    #     # 从 result 中获取原始图和增强图 (依赖于 process_frame 的修改)
    #     # 如果尚未集成增强器，get() 会默认返回 frame，保证代码不报错
    #     raw_img = result.get('raw_image', frame).copy()
    #     enh_img = result.get('enhanced_image', frame).copy()
        
    #     # 处理深度图
    #     if depth is not None:
    #         depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #         vis_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    #     else:
    #         vis_depth = np.zeros_like(enh_img)
    #         cv2.putText(vis_depth, "NPU Init...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    #     # =========================================================
    #     # 2. 统一缩放 (关键步骤)
    #     # =========================================================
    #     # 将三张图 resize 到统一的显示尺寸
    #     raw_disp = cv2.resize(raw_img, (display_w, display_h))
    #     enh_disp = cv2.resize(enh_img, (display_w, display_h))
    #     dep_disp = cv2.resize(vis_depth, (display_w, display_h))

    #     # =========================================================
    #     # 3. 在增强图上绘制特征点 (Features on Enhanced)
    #     # =========================================================
    #     # [学术逻辑] SLAM 追踪的是增强后的纹理，所以特征点应该画在增强图上
    #     kpts = result.get('current_keypoints')
    #     if kpts is None and hasattr(self.visual_odometry, 'prev_kpts'):
    #          kpts = self.visual_odometry.prev_kpts

    #     if kpts is not None:
    #         # [重要修正] 因为图像被 resize 了，特征点坐标(x,y)也必须缩放
    #         scale_x = display_w / src_w
    #         scale_y = display_h / src_h
            
    #         # 追踪成功为绿色，失败为红色
    #         color = (0, 255, 0) if result.get('success', False) else (0, 0, 255)
            
    #         for i, pt in enumerate(kpts):
    #             if i % 2 == 0: # 采样绘制，避免太密集遮挡图像
    #                 px = int(pt[0] * scale_x)
    #                 py = int(pt[1] * scale_y)
    #                 cv2.circle(enh_disp, (px, py), 2, color, -1)
            
    #         # 显示特征点数量
    #         cv2.putText(enh_disp, f"Feats: {len(kpts)}", (10, 30), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #     # =========================================================
    #     # 4. 绘制学术标题栏 (Panel Headers)
    #     # =========================================================
    #     raw_disp = self._draw_panel_header(raw_disp, "Raw Input (Degraded)")
    #     enh_disp = self._draw_panel_header(enh_disp, "Physics Enhanced (SLAM Input)")
    #     dep_disp = self._draw_panel_header(dep_disp, "Depth Estimation (NPU)")

    #     # 拼接第一行 (Top Row)
    #     top_row = np.hstack((raw_disp, enh_disp, dep_disp))
    #     total_w = top_row.shape[1]  # 计算总宽度，用于对齐第二行

    #     # =========================================================
    #     # 5. 构建第二行 (Trajectory & Status)
    #     # =========================================================
    #     bottom_h = 240 # 下半部分高度固定
    #     half_w = total_w // 2 # 左右各分一半宽度
        
    #     # (A) 轨迹图
    #     traj_map = self._draw_trajectory_widget(size=(bottom_h, half_w))
        
    #     # (B) 状态面板 (自动填补剩余宽度)
    #     status_panel = np.zeros((bottom_h, total_w - half_w, 3), dtype=np.uint8)
    #     status_panel[:] = (30, 30, 30) # 深灰背景
        
    #     # 状态文字内容
    #     pos = result.get('position', np.zeros(3))
    #     fps = getattr(self, '_ui_fps', 0.0)
        
    #     lines = [
    #         f"System Status: {'RUNNING' if self.is_running else 'STOPPED'}",
    #         f"Frame Index:   {result.get('frame', 0)}",
    #         f"Timestamp:     {result.get('timestamp', 0):.2f} s",
    #         f"FPS:           {fps:.1f}",
    #         "-" * 30,
    #         f"Pos X: {pos[0]:.3f} m",
    #         f"Pos Y: {pos[1]:.3f} m",
    #         f"Pos Z: {pos[2]:.3f} m",
    #         "-" * 30,
    #         f"Method: Physics-Guided Visual-Inertial SLAM"
    #     ]
        
    #     y_start = 50
    #     for i, line in enumerate(lines):
    #         color = (0, 255, 0) if "RUNNING" in line else (220, 220, 220)
    #         if "Method" in line: color = (255, 191, 0) # 橙色高亮核心方法
    #         cv2.putText(status_panel, line, (20, y_start + i*20), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            
    #     status_panel = self._draw_panel_header(status_panel, "System Statistics")

    #     bottom_row = np.hstack((traj_map, status_panel))

    #     # =========================================================
    #     # 6. 最终组合与显示
    #     # =========================================================
    #     final_dashboard = np.vstack((top_row, bottom_row))
        
    #     # 添加 Top Bar (如果您的类中有这个逻辑，这里可以复用，或者直接显示 dashboard)
    #     # 考虑到代码复用，我们这里直接显示 dashboard，或者您可以把之前 Top Bar 的代码加在 final_dashboard 上面
    #     # 这里为了保持函数独立性，直接显示 dashboard
        
    #     cv2.imshow(self.UI_WINDOW_NAME, final_dashboard)
    def _visualize_frame(self, frame: np.ndarray, depth: np.ndarray, result: Dict):
        """
        [Final Fixed] 学术级仪表盘可视化
        - Row 1: Raw | Enhanced | Depth
        - Row 2: Trajectory | Status
        - Top:   Interactive Buttons (Save/Quit)
        """
        self._ensure_ui_window()

        # =========================================================
        # 1. 核心布局内容生成 (3-Panel Layout)
        # =========================================================
        src_h, src_w = frame.shape[:2]
        
        # 设定显示高度，宽度自适应
        display_h = 360 
        display_w = int(display_h * (src_w / src_h))
        
        # 获取图像
        raw_img = result.get('raw_image', frame).copy()
        enh_img = result.get('enhanced_image', frame).copy()
        
        # 深度图处理
        if depth is not None:
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        else:
            vis_depth = np.zeros_like(enh_img)
            cv2.putText(vis_depth, "NPU Init...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 统一缩放
        raw_disp = cv2.resize(raw_img, (display_w, display_h))
        enh_disp = cv2.resize(enh_img, (display_w, display_h))
        dep_disp = cv2.resize(vis_depth, (display_w, display_h))

        # 绘制特征点 (画在增强图上)
        kpts = result.get('current_keypoints')
        if kpts is None and hasattr(self.visual_odometry, 'prev_kpts'):
             kpts = self.visual_odometry.prev_kpts
        
        if kpts is not None:
            scale_x = display_w / src_w
            scale_y = display_h / src_h
            color = (0, 255, 0) if result.get('success', False) else (0, 0, 255)
            for i, pt in enumerate(kpts):
                if i % 2 == 0:
                    px, py = int(pt[0] * scale_x), int(pt[1] * scale_y)
                    cv2.circle(enh_disp, (px, py), 2, color, -1)
            cv2.putText(enh_disp, f"Feats: {len(kpts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 添加标题
        raw_disp = self._draw_panel_header(raw_disp, "Raw Input (Degraded)")
        enh_disp = self._draw_panel_header(enh_disp, "Physics Enhanced (SLAM Input)")
        dep_disp = self._draw_panel_header(dep_disp, "Depth Estimation (NPU)")

        # 拼接 Row 1
        top_row = np.hstack((raw_disp, enh_disp, dep_disp))
        
        # 构建 Row 2
        total_w = top_row.shape[1]
        bottom_h = 240
        half_w = total_w // 2
        
        traj_map = self._draw_trajectory_widget(size=(bottom_h, half_w))
        
        status_panel = np.zeros((bottom_h, total_w - half_w, 3), dtype=np.uint8)
        status_panel[:] = (30, 30, 30)
        
        # 状态文字
        pos = result.get('position', np.zeros(3))
        lines = [
            f"Status: {'RUNNING' if self.is_running else 'STOPPED'}",
            f"Frame:  {result.get('frame', 0)}",
            f"Time:   {result.get('timestamp', 0):.2f} s",
            f"FPS:    {getattr(self, '_ui_fps', 0.0):.1f}",
            "-" * 30,
            f"Pos:    [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]",
            "-" * 30,
            "Enhancer: Active (Shared NPU)"
        ]
        for i, line in enumerate(lines):
            cv2.putText(status_panel, line, (20, 50 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
        status_panel = self._draw_panel_header(status_panel, "System Stats")

        bottom_row = np.hstack((traj_map, status_panel))
        
        # 组合工作区
        workspace = np.vstack((top_row, bottom_row))

        # =========================================================
        # 2. 恢复 TopBar 和 按钮逻辑 (Fix Missing Buttons)
        # =========================================================
        full_w = workspace.shape[1]
        topbar_h = 44
        topbar = np.zeros((topbar_h, full_w, 3), dtype=np.uint8)
        topbar[:] = (22, 22, 22) # 深色背景

        # 标题
        cv2.putText(topbar, "PiSLAM Academic Workspace", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

        # 绘制按钮
        btn_w, btn_h = 160, 30
        gap = 10
        yb1 = 7
        yb2 = yb1 + btn_h
        
        # 计算按钮位置 (右对齐)
        quit_rect = (full_w - 14 - btn_w, yb1, full_w - 14, yb2)
        save_rect = (quit_rect[0] - gap - btn_w, yb1, quit_rect[0] - gap, yb2)

        # 更新鼠标点击区域注册表
        self._ui_buttons = {
            'save': save_rect,
            'quit': quit_rect
        }

        # 绘制按钮实体
        self._draw_button(topbar, save_rect, "Save Results", "S",
                          hovered=(self._ui_hover == 'save'), pressed=(self._ui_pressed == 'save'), accent=(0, 200, 140))
        self._draw_button(topbar, quit_rect, "Quit", "Q",
                          hovered=(self._ui_hover == 'quit'), pressed=(self._ui_pressed == 'quit'), accent=(0, 0, 255))

        # 最终堆叠 (TopBar 在上，Workspace 在下)
        final_dashboard = np.vstack((topbar, workspace))

        cv2.imshow(self.UI_WINDOW_NAME, final_dashboard)

    # ==============================================================================
    # Logging / save
    # ==============================================================================
    def _print_status(self, result: Dict):
        pos = result['position']
        status = (f"\rFrame {result['frame']:5d} | Pos: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | "
                  f"KF: {len(self.pose_graph.keyframes):3d}")
        if result.get('loop_detected', False):
            status += " | 🔗 LOOP!"
        print(status, end='', flush=True)

    def _save_results(self):
        print("\n\n💾 Saving results...")
        self.pose_graph.optimize()

        traj_path = os.path.join(self.output_dir, 'trajectory.txt')
        self.pose_graph.save_trajectory(traj_path, format='tum')

        map_path = os.path.join(self.output_dir, 'map.ply')
        global_map = self.pose_graph.build_global_map(
            filter_noise=True,
            strict_mode=True
        )
        o3d.io.write_point_cloud(map_path, global_map)

        # ==========================================================
        # 🟢 [新增] 4. 保存关键帧数据 (Images + JSON Metadata)
        # ==========================================================
        kf_dir = os.path.join(self.output_dir, 'keyframes')
        if not os.path.exists(kf_dir):
            os.makedirs(kf_dir)
            
        print(f"📸 Dumping keyframes to: {kf_dir} ...")
        # 调用 pose_graph_enhanced.py 中已有的 save_keyframes 方法
        self.pose_graph.save_keyframes(kf_dir)
        # ==========================================================

        print(f"✅ Results saved to: {self.output_dir}")
        self._ui_set_status("Saved ✓", duration_s=2.0)


def main():
    parser = argparse.ArgumentParser(description='PiSLAM System')
    parser.add_argument('--mode', type=str, required=True, choices=['hailo', 'dataset'])
    parser.add_argument('--config', type=str, default='./config/slam_config.yaml')
    parser.add_argument('--hef', type=str, default='scdepthv3.hef')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--rgb-dir', type=str, default=None)
    parser.add_argument('--depth-dir', type=str, default=None)
    parser.add_argument('--no-viz', action='store_true')

    args = parser.parse_args()

    config_path = args.config if os.path.exists(args.config) else None
    slam = PiSLAM(config_path)

    if args.no_viz:
        slam.visualization = False

    if args.mode == 'hailo':
        slam.initialize_depth_estimator('hailo', hef_path=args.hef)
        slam.run_live(camera_id=args.camera)

    elif args.mode == 'dataset':
        if args.rgb_dir is None:
            print("Error: --rgb-dir required for dataset mode")
            sys.exit(1)
        if args.depth_dir is not None:
            slam.initialize_depth_estimator('dataset', depth_dir=args.depth_dir)
        else:
            slam.initialize_depth_estimator('hailo', hef_path=args.hef)
        slam.run_dataset(args.rgb_dir, args.depth_dir)


if __name__ == "__main__":
    main()
