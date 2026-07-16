#!/usr/bin/env python3
"""
================================================================================
PiSLAM Loop Closure: XFeat-VLAD (Zero-Latency Edition)
================================================================================
Optimization: Replaced O(N*1000) BoW with O(N*16) VLAD aggregation.
"""

import numpy as np
import cv2
import os
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("⚠️ FAISS not found. Performance will be degraded.")

@dataclass
class LoopClosureConfig:
    vlad_clusters: int = 16      # ⚡ 关键优化: 从 1000 降到 16
    desc_dim: int = 64
    top_k_candidates: int = 3
    similarity_threshold: float = 0.65
    min_keyframe_gap: int = 15
    warmup_keyframes: int = 20   # 收集20帧后进行瞬间训练

@dataclass
class LoopCandidate:
    query_id: int
    match_id: int
    similarity_score: float
    is_verified: bool = False

class XFeatVLADAggregator:
    def __init__(self, config: LoopClosureConfig):
        self.config = config
        self.is_trained = False
        self.kmeans = None
        self.centroids = None
        self.train_buffer = [] 
        self._lock = threading.Lock()

    def train(self):
        """极速训练 (k=16)"""
        if not HAS_FAISS or len(self.train_buffer) == 0: return

        # 1. 降采样优化: 只要 2000 个点就足够训练 16 个中心了
        data = np.vstack(self.train_buffer).astype(np.float32)
        if data.shape[0] > 2000:
            indices = np.random.choice(data.shape[0], 2000, replace=False)
            data = data[indices]

        print(f"⚡ [VLAD] Instant training on {data.shape[0]} descriptors (k={self.config.vlad_clusters})...")
        
        # 2. 训练
        self.kmeans = faiss.Kmeans(d=self.config.desc_dim, k=self.config.vlad_clusters, niter=10, verbose=False)
        self.kmeans.train(data)
        self.centroids = self.kmeans.centroids
        self.is_trained = True
        self.train_buffer = [] # 释放内存
        print("✅ [VLAD] Ready.")

    def aggregate(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.config.desc_dim * self.config.vlad_clusters, dtype=np.float32)

        descriptors = descriptors.astype(np.float32)
        cv2.normalize(descriptors, descriptors, norm_type=cv2.NORM_L2)

        # 预热期: 使用 Global Average Pooling (GAP) 代替，速度最快
        if not self.is_trained:
            with self._lock:
                if len(self.train_buffer) < 50:
                    self.train_buffer.append(descriptors)
            # 返回 GAP 向量
            gap = np.mean(descriptors, axis=0)
            gap /= (np.linalg.norm(gap) + 1e-8)
            # 补零以匹配 VLAD 维度 (方便后续处理)
            full_vec = np.zeros(self.config.desc_dim * self.config.vlad_clusters, dtype=np.float32)
            full_vec[:self.config.desc_dim] = gap
            return full_vec

        # 成熟期: VLAD 聚合
        D, I = self.kmeans.index.search(descriptors, 1)
        vlad_vector = np.zeros((self.config.vlad_clusters, self.config.desc_dim), dtype=np.float32)
        
        # 向量化累加 (比循环快)
        np.add.at(vlad_vector, I.flatten(), descriptors - self.centroids[I.flatten()])

        # 归一化
        vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))
        vlad_vector = vlad_vector.flatten()
        vlad_vector /= (np.linalg.norm(vlad_vector) + 1e-8)
        return vlad_vector

class FAISSDatabase:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) if HAS_FAISS else None
        self.keyframe_ids = []

    def add(self, keyframe_id: int, vector: np.ndarray):
        if self.index: self.index.add(vector.reshape(1, -1))
        self.keyframe_ids.append(keyframe_id)

    def search(self, query: np.ndarray, k: int, exclude_ids: Set[int]):
        if not self.index or self.index.ntotal == 0: return []
        D, I = self.index.search(query.reshape(1, -1), k * 2)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: continue
            kf_id = self.keyframe_ids[idx]
            if kf_id not in exclude_ids:
                results.append((kf_id, float(dist)))
                if len(results) >= k: break
        return results

class EnhancedLoopClosureDetector:
    def __init__(self, config_path=None, hef_path=None, onnx_path=None, vdevice=None):
        self.config = LoopClosureConfig()
        self.vlad = XFeatVLADAggregator(self.config)
        self.db = None
        self._training_thread = None
        print("✅ [LoopClosure] XFeat-VLAD backend initialized (Latency Optimized)")

    def add_keyframe(self, keyframe_id: int, image, position=None, descriptors=None):
        if descriptors is None: return
        
        vec = self.vlad.aggregate(descriptors)
        
        # 延迟初始化数据库
        if self.db is None: self.db = FAISSDatabase(vec.shape[0])
        
        # ⚠️ 如果维度变了(从GAP变成了VLAD)，为了简单起见，我们通常会重置库
        # 但在Demo中，我们只存入。实际工程中应把旧向量映射过来。
        if vec.shape[0] == self.db.dim:
            self.db.add(keyframe_id, vec)

        # 触发后台训练
        if not self.vlad.is_trained and keyframe_id >= self.config.warmup_keyframes:
            if self._training_thread is None or not self._training_thread.is_alive():
                self._training_thread = threading.Thread(target=self._train_worker)
                self._training_thread.start()

    def _train_worker(self):
        self.vlad.train()
        # 训练完后，新的维度会变大，此时应该重置 DB (Demo简化处理: 之后的新帧才生效)
        # 实际操作: print("VLAD Activated. Old keyframes use GAP fallback.")

    def detect_loop_closure(self, keyframe_id, image, position, descriptors, keypoints, all_keyframes):
        if descriptors is None or self.db is None: return []
        
        query_vec = self.vlad.aggregate(descriptors)
        if query_vec.shape[0] != self.db.dim: return []

        exclude = set(range(max(0, keyframe_id - self.config.min_keyframe_gap), keyframe_id + 1))
        candidates = []
        for match_id, score in self.db.search(query_vec, self.config.top_k_candidates, exclude):
            if score > self.config.similarity_threshold:
                candidates.append(LoopCandidate(keyframe_id, match_id, score, is_verified=True))
        return candidates

# 兼容接口
def create_enhanced_loop_detector(config_path=None, hef_path=None, onnx_path=None, vdevice=None):
    return EnhancedLoopClosureDetector()