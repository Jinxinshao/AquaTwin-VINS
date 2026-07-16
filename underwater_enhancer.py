# -*- coding: utf-8 -*-
"""
================================================================================
GPLMv2 Underwater Enhancer Module (SLAM Integrated)
Author: Academic Research Team
Description: Adapted from rpi_inference_full_npu.py for integration with PiSLAM.
             Supports shared Hailo VDevice execution.
================================================================================
"""

import numpy as np
import cv2
import time
from hailo_platform import (HEF, ConfigureParams, InferVStreams,
                            InputVStreamParams, OutputVStreamParams, FormatType,
                            HailoStreamInterface)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# ============================================================================
# Math Helpers & Kernels (保留您的高效内核)
# ============================================================================

def create_softplus_lut(size: int = 4000, range_val: float = 20.0) -> np.ndarray:
    x = np.linspace(-range_val, range_val, size)
    lut = np.log1p(np.exp(np.clip(x, -50, 20)))
    return lut.astype(np.float32)

SOFTPLUS_LUT = create_softplus_lut()
LUT_SIZE = len(SOFTPLUS_LUT)
LUT_RANGE = 20.0

@jit(nopython=True, parallel=True, fastmath=True)
def enforce_beta_ordering_lowres(params: np.ndarray, lut: np.ndarray, 
                                 lut_range: float, lut_size: int) -> None:
    H, W, C = params.shape
    scale = (lut_size - 1) / (2 * lut_range)
    for i in prange(H):
        for j in range(W):
            b0, b1, b2 = params[i, j, 0], params[i, j, 1], params[i, j, 2]
            
            d1 = b1 - b0
            idx1 = int((d1 + lut_range) * scale)
            idx1 = min(max(idx1, 0), lut_size - 1)
            sp1 = lut[idx1] if -lut_range <= d1 <= lut_range else (d1 if d1 > lut_range else 0.0)
            new_b1 = b0 - sp1
            
            d2 = b2 - new_b1
            idx2 = int((d2 + lut_range) * scale)
            idx2 = min(max(idx2, 0), lut_size - 1)
            sp2 = lut[idx2] if -lut_range <= d2 <= lut_range else (d2 if d2 > lut_range else 0.0)
            new_b2 = new_b1 - sp2
            
            params[i, j, 1] = new_b1
            params[i, j, 2] = new_b2

@jit(nopython=True, parallel=True, fastmath=False)
def fused_render_kernel(img_uint8: np.ndarray, params: np.ndarray, 
                        min_t: float, out_uint8: np.ndarray) -> None:
    H, W, C = img_uint8.shape
    inv255 = 1.0 / 255.0
    for i in prange(H):
        for j in range(W):
            t = params[i, j, 6]
            t_safe = t if t > min_t else min_t
            omt = 1.0 - t
            bi = params[i, j, 3:6]
            
            for c in range(3):
                pix = float(img_uint8[i, j, c]) * inv255
                res = (pix - (bi[c] * omt)) / t_safe
                val = int(min(max(res, 0.0), 1.0) * 255.0)
                out_uint8[i, j, c] = val

# ============================================================================
# Main Class
# ============================================================================

class UnderwaterEnhancer:
    def __init__(self, hef_path: str, vdevice, 
                 input_res=(512, 512), 
                 min_transmission=0.05):
        
        self.input_h, self.input_w = input_res
        self.min_transmission = min_transmission
        
        print(f"🌊 [Enhancer] Initializing Physics Enhancer with {hef_path}")
        
        # 关键修改：使用传入的 Shared VDevice
        if vdevice is None:
            raise RuntimeError("Enhancer requires a shared VDevice")
            
        self.hef = HEF(hef_path)
        self.network_groups = vdevice.configure(self.hef, ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe))
        self.network_group = self.network_groups[0]
        
        self.input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        # 激活网络上下文
        self._ng_context = self.network_group.activate()
        self._ng_context.__enter__()
        
        self._pipe_context = InferVStreams(self.network_group, self.input_params, self.output_params)
        self.pipeline = self._pipe_context.__enter__()
        
        # JIT Warmup
        if NUMBA_AVAILABLE:
            self._warmup_jit()
            
        print("✅ [Enhancer] Ready.")

    def _warmup_jit(self):
        d_u8 = np.zeros((64, 64, 3), dtype=np.uint8)
        d_p = np.zeros((64, 64, 7), dtype=np.float32)
        enforce_beta_ordering_lowres(d_p, SOFTPLUS_LUT, LUT_RANGE, LUT_SIZE)
        fused_render_kernel(d_u8, d_p, 0.1, d_u8)

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        # Preprocessing
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        inp_npu = cv2.resize(image_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(inp_npu, 0)
        
        # Inference
        res = self.pipeline.infer({self.hef.get_input_vstream_infos()[0].name: batch})
        raw_params = list(res.values())[0][0] # (64, 64, 7) assuming fixed shape
        
        # Format check (H, W, C)
        if raw_params.shape[0] == 7: 
            raw_params = np.transpose(raw_params, (1, 2, 0))
            
        # Post-processing (Physics)
        enforce_beta_ordering_lowres(raw_params, SOFTPLUS_LUT, LUT_RANGE, LUT_SIZE)
        
        # Upsample params to original resolution
        params_up = cv2.resize(raw_params, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Render
        out_rgb = np.empty_like(image_rgb)
        fused_render_kernel(image_rgb, params_up, self.min_transmission, out_rgb)
        
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        if hasattr(self, '_pipe_context'): self._pipe_context.__exit__(None, None, None)
        if hasattr(self, '_ng_context'): self._ng_context.__exit__(None, None, None)