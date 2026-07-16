#!/usr/bin/env python3
"""
================================================================================
PiSLAM Pose Graph Module - Compatibility Layer
================================================================================
This file provides backward compatibility by re-exporting classes from the
enhanced implementation.

Original file backed up as: pose_graph_original.py
================================================================================
"""

# Re-export all classes from enhanced implementation
from pose_graph_enhanced import (
    PoseGraphOptimizer,
    Keyframe,
    PoseGraphNode,
    PoseGraphEdge,
    LoopClosureCandidateLegacy as LoopClosureCandidate,
    create_pose_graph_optimizer
)

# Also export the enhanced loop closure components for direct access
try:
    from loop_closure_enhanced import (
        EnhancedLoopClosureDetector,
        LoopClosureConfig,
        LoopCandidate,
        FAISSBagOfWords,
        FAISSDatabase,
        GlobalDescriptorExtractor,
        create_enhanced_loop_detector
    )
except ImportError:
    pass

# For backward compatibility with old BagOfWords
try:
    from loop_closure_enhanced import FAISSBagOfWords as BagOfWords
except ImportError:
    from features import BagOfWords

__all__ = [
    'PoseGraphOptimizer',
    'Keyframe', 
    'PoseGraphNode',
    'PoseGraphEdge',
    'LoopClosureCandidate',
    'create_pose_graph_optimizer',
    'BagOfWords'
]

print("✅ Using Enhanced Loop Closure Detection (FAISS + Global Descriptors)")
