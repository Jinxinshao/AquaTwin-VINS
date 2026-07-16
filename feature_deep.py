import numpy as np
import cv2
import onnxruntime as ort
import os

class SuperPointFrontend:
    """SuperPoint ONNX 封装"""
    def __init__(self, model_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.input_name = self.session.get_inputs()[0].name # 'image'

    def run(self, img):
        """
        输入: 灰度图 (H, W)
        输出: keypoints (N, 2), descriptors (256, N), scores (N,)
        """
        # 预处理：归一化 + 增加维度 (1, 1, H, W)
        img_tensor = (img.astype(np.float32) / 255.0)[None, None, :, :]
        
        # 推理
        # 输出顺序由 export 脚本决定，通常是: keypoints, scores, descriptors
        outs = self.session.run(None, {self.input_name: img_tensor})
        
        kpts = outs[0][0]  # (N, 2)
        scores = outs[1][0] # (N,)
        descs = outs[2][0]  # (256, N)
        
        # 后处理：转置描述子为 (N, 256) 以匹配 OpenCV 格式
        descs = descs.transpose(1, 0)
        
        return kpts, descs, scores

    def extract_cv2(self, img):
        """返回 OpenCV 格式的结果，方便无缝替换 ORB"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        kpts, descs, scores = self.run(gray)
        
        # 转换为 cv2.KeyPoint
        cv_kpts = []
        for i in range(len(kpts)):
            pt = kpts[i]
            # 只有分数高的才保留 (Export时可能已经过滤，这里双重保险)
            if scores[i] >= self.conf_thresh:
                kpt = cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0)
                kpt.response = float(scores[i])
                cv_kpts.append(kpt)
                
        # 确保描述子数量与关键点一致（通常是一致的）
        return cv_kpts, descs


class SuperGlueMatcher:
    """SuperGlue ONNX 封装"""
    def __init__(self, model_path, match_threshold=0.2):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.match_threshold = match_threshold
        # 获取输入名称
        inputs = self.session.get_inputs()
        self.input_names = [i.name for i in inputs] # ['kpts0', 'desc0', 'scores0', 'kpts1', 'desc1', 'scores1']

    def match(self, kpts0, desc0, scores0, kpts1, desc1, scores1, img0_shape, img1_shape):
        """
        输入: 原始 numpy 数组
        kpts: (N, 2)
        desc: (N, 256)  <-- 注意 SuperPoint输出通常是 (N, 256)，但 SuperGlue 需要 (1, 256, N)
        scores: (N,)
        """
        # 1. 维度调整 (增加 Batch 维度，调整描述子形状)
        # SuperGlue ONNX 期望: kpts (1, N, 2), desc (1, 256, N), scores (1, N)
        
        # 检查是否为空
        if len(kpts0) == 0 or len(kpts1) == 0:
            return []

        kpts0_t = kpts0[None, :, :]
        # desc 需要转置: (N, 256) -> (256, N) -> (1, 256, N)
        desc0_t = desc0.transpose(1, 0)[None, :, :]
        scores0_t = scores0[None, :]

        kpts1_t = kpts1[None, :, :]
        desc1_t = desc1.transpose(1, 0)[None, :, :]
        scores1_t = scores1[None, :]

        # 2. 推理
        inputs = {
            self.input_names[0]: kpts0_t, # keypoints0
            self.input_names[1]: desc0_t, # descriptors0
            self.input_names[2]: scores0_t, # scores0
            self.input_names[3]: kpts1_t, # keypoints1
            self.input_names[4]: desc1_t, # descriptors1
            self.input_names[5]: scores1_t, # scores1
        }
        
        # 输出: matches0 (1, N), matching_scores0 (1, N)
        outs = self.session.run(None, inputs)
        matches_idx = outs[0][0] # matches0: -1 表示无匹配，否则为 index
        match_scores = outs[1][0]

        # 3. 转换为 cv2.DMatch
        cv_matches = []
        for i, match_idx in enumerate(matches_idx):
            if match_idx > -1 and match_scores[i] >= self.match_threshold:
                # queryIdx=i (img0中的索引), trainIdx=match_idx (img1中的索引)
                m = cv2.DMatch(queryIdx=i, trainIdx=int(match_idx), distance=1.0 - match_scores[i])
                cv_matches.append(m)
        
        return cv_matches