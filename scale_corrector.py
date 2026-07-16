import threading
import time
import numpy as np
import cv2
import onnxruntime as ort
import os
# 在 import os 后面添加
from scale_fusion_bayesian import BayesianScaleEstimator

class DepthCorrector:
    """
    Teacher 模块：使用 Depth Anything V2 (CPU) 在后台提供高质量深度参考。
    """
    def __init__(self, model_path="depth_anything_v2_vits.onnx", target_interval=3.0):
        """
        Args:
            model_path: ONNX 模型路径
            target_interval: 每隔多少秒校正一次 (根据你的实测 2.7s，设定 3.0s 比较稳)
        """
        self.model_path = model_path
        self.target_interval = target_interval
        self.last_run_time = 0
        self.current_scale_factor = 1.0
        self.running = False
        self.lock = threading.Lock()

        # 初始化贝叶斯估计器
        self.bayesian_estimator = BayesianScaleEstimator(
            initial_scale=1.0,
            initial_variance=0.1,  # 初始给一点不确定性
            process_noise=0.001    # 允许尺度随时间缓慢漂移
        )
        
        # 状态容器
        self.latest_image = None
        self.latest_sc_depth = None 
        
        # 初始化 ONNX (CPU)
        print(f"🐢 [Corrector] 正在加载 Depth Anything: {model_path} ...")
        if not os.path.exists(model_path):
            print(f"❌ [Corrector] 找不到模型文件: {model_path}")
            self.ready = False
            return

        try:
            # 限制线程数，防止抢占主线程资源
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.ready = True
            print("✅ [Corrector] 教官已就位 (CPU)")
        except Exception as e:
            print(f"❌ [Corrector] 加载失败: {e}")
            self.ready = False

        # 启动后台线程
        if self.ready:
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.running = True
            self.thread.start()

    def update_request(self, rgb_image, sc_depth):
        """主线程调用：提交作业"""
        if not self.ready: return
        
        # 只有冷却时间到了才接新任务
        if time.time() - self.last_run_time > self.target_interval:
            with self.lock:
                # 存下最新的一帧
                self.latest_image = rgb_image.copy()
                self.latest_sc_depth = sc_depth.copy()

    def get_scale_factor(self):
        """主线程调用：获取最新的修正系数"""
        with self.lock:
            return self.current_scale_factor

    def _worker_loop(self):
        """后台线程：慢工出细活"""
        while self.running:
            img_to_process = None
            sc_depth_ref = None
            
            # 1. 取任务
            with self.lock:
                if self.latest_image is not None:
                    img_to_process = self.latest_image
                    sc_depth_ref = self.latest_sc_depth
                    self.latest_image = None # 清空
            
            if img_to_process is not None:
                start_t = time.time()
                
                # 2. 跑 Depth Anything (耗时 ~2.7s)
                teacher_depth = self._run_inference(img_to_process)
                
                # 3. 计算对齐比例
                if teacher_depth is not None:
                    scale = self._compute_alignment(teacher_depth, sc_depth_ref)
                    
                    with self.lock:
                        # 平滑更新 (EMA) 避免画面闪烁
                        # 0.7 * 旧值 + 0.3 * 新值
                        # self.current_scale_factor = 0.7 * self.current_scale_factor + 0.3 * scale
                        
                        # 使用贝叶斯更新
                        # 注意：这里 teacher_depth 和 sc_depth_ref 需要是 numpy 数组
                        # update 方法会自动处理逐像素的比值分布
                        new_scale, variance, valid = self.bayesian_estimator.update(teacher_depth, sc_depth_ref)

                        if valid:
                            # 只有当贝叶斯估计认为这次观测有效时，才更新全局尺度
                            # 还可以判断置信度：if self.bayesian_estimator.get_confidence() > 0.3:
                            self.current_scale_factor = new_scale
                            self.last_run_time = time.time()

                            # 打印更详细的调试信息
                            conf = self.bayesian_estimator.get_confidence()
                            print(f"⚖️ [Teacher] 贝叶斯校正 | 尺度: x{self.current_scale_factor:.3f} | 置信度: {conf:.2f}")
                        else:
                            print("⚠️ [Teacher] 观测质量不佳，跳过更新")
                        
                        self.last_run_time = time.time()
                    
                    print(f"⚖️ [Teacher] 校正完成 ({time.time()-start_t:.1f}s) | 比例: x{self.current_scale_factor:.3f}")
            
            time.sleep(0.1)

    # def _run_inference(self, frame):
    #     try:
    #         # 预处理 (518x518)
    #         target_size = (518, 518)
    #         h, w = frame.shape[:2]
    #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(img, target_size)
            
    #         img = img.astype(np.float32) / 255.0
    #         mean = np.array([0.485, 0.456, 0.406])
    #         std = np.array([0.229, 0.224, 0.225])
    #         img = (img - mean) / std
    #         img = img.transpose(2, 0, 1)
    #         input_tensor = np.expand_dims(img, axis=0).astype(np.float32)
            
    #         # 推理
    #         outputs = self.session.run(None, {self.input_name: input_tensor})
    #         depth = outputs[0].squeeze()
            
    #         # 还原尺寸
    #         depth = cv2.resize(depth, (w, h))
    #         return depth
    #     except Exception as e:
    #         print(f"⚠️ 推理错误: {e}")
    #         return None

    def _run_inference(self, frame):
        try:
            # 预处理
            target_size = (518, 518)
            h, w = frame.shape[:2]
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            
            # 🟢 修复：强制 float32，避免 float64 导致的 ONNX 崩溃
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)
            
            # 确保输入 Tensor 也是 float32
            input_tensor = np.expand_dims(img, axis=0).astype(np.float32)
            
            # 推理
            outputs = self.session.run(None, {self.input_name: input_tensor})
            depth = outputs[0].squeeze()
            
            # 还原尺寸
            depth = cv2.resize(depth, (w, h))
            return depth
        except Exception as e:
            print(f"⚠️ Teacher 推理错误: {e}")
            return None

    def _compute_alignment(self, teacher_depth, student_depth):
        """
        计算 Student(scDepth) 需要乘多少才能变成 Teacher(DepthAnything)
        """
        # 两个模型输出的可能都是 "相对深度" 或 "视差(1/depth)"
        # 我们假设它们性质类似，直接对齐中位数
        
        mask = (student_depth > 0.1) & (teacher_depth > 0.1)
        if np.sum(mask) < 100:
            return self.current_scale_factor
            
        # 使用中位数对齐 (最稳健，不怕边缘噪声)
        median_student = np.median(student_depth[mask])
        median_teacher = np.median(teacher_depth[mask])
        
        if median_student < 0.001: return 1.0
        
        ratio = median_teacher / median_student
        
        # 安全截断 (防止比例尺乱飞)
        return np.clip(ratio, 0.5, 2.0)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()