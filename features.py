#!/usr/bin/env python3
"""
================================================================================
PiSLAM Feature Extraction and Matching Module
================================================================================

This module implements ORB (Oriented FAST and Rotated BRIEF) feature detection
and matching for visual odometry and loop closure detection.

Theoretical Background:
-----------------------

ORB Features combine:
1. FAST keypoint detector (Features from Accelerated Segment Test)
2. Oriented BRIEF descriptor (Binary Robust Independent Elementary Features)

FAST Keypoint Detection:
    A pixel p is a corner if N contiguous pixels in its Bresenham circle
    are all brighter or all darker than p by threshold t.
    
    Orientation is computed using intensity centroid:
    θ = atan2(m_01, m_10)
    where m_pq = Σ x^p * y^q * I(x,y) are image moments

BRIEF Descriptor:
    Binary string computed by comparing intensity pairs:
    τ(p; x, y) = 1 if p(x) < p(y), else 0
    
    ORB "steers" BRIEF according to keypoint orientation for
    rotation invariance.

Feature Matching:
    ORB uses Hamming distance: d(a,b) = popcount(a XOR b)
    This is efficiently computed using POPCNT instruction.
    
    Lowe's ratio test rejects ambiguous matches:
    Accept match if d_best / d_second < threshold (typically 0.7-0.8)

References:
    [1] Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011
    [2] Rosten & Drummond, "Machine learning for high-speed corner detection", ECCV 2006
    [3] Lowe, "Distinctive image features from scale-invariant keypoints", IJCV 2004

Author: Academic SLAM Implementation
================================================================================
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import yaml


@dataclass
class KeyPoint:
    """
    Represents a detected feature keypoint with additional metadata.
    
    Attributes:
        pt: 2D location (u, v) in pixels
        size: Diameter of meaningful keypoint neighborhood
        angle: Orientation in degrees [0, 360)
        response: Detector response (corner strength)
        octave: Pyramid octave from which keypoint was extracted
        class_id: Object class (for object recognition)
        depth: Associated depth value (if available)
        point_3d: 3D position in camera frame (if depth available)
    """
    pt: Tuple[float, float]
    size: float
    angle: float
    response: float
    octave: int
    class_id: int = -1
    depth: Optional[float] = None
    point_3d: Optional[np.ndarray] = None


@dataclass
class FeatureMatch:
    """
    Represents a feature match between two frames.
    
    Attributes:
        query_idx: Index of keypoint in query (current) frame
        train_idx: Index of keypoint in train (previous) frame
        distance: Hamming distance between descriptors
        is_inlier: Whether match passed geometric verification
    """
    query_idx: int
    train_idx: int
    distance: float
    is_inlier: bool = True


class ORBExtractor:
    """
    ORB feature extractor with configurable parameters.
    
    This implementation uses a scale pyramid to achieve scale invariance,
    with FAST corner detection at each level and oriented BRIEF descriptors.
    
    The number of features at each pyramid level follows a geometric
    distribution to balance coverage across scales.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ORB extractor from config file or default parameters.
        
        Args:
            config_path: Path to YAML config file. If None, use defaults.
        """
        # Default parameters (optimized for embedded devices)
        self.n_features = 1000
        self.scale_factor = 1.2
        self.n_levels = 8
        self.edge_threshold = 31
        self.first_level = 0
        self.wta_k = 2
        self.patch_size = 31
        self.fast_threshold = 20
        
        # Load from config if provided
        if config_path is not None:
            self._load_config(config_path)
        
        # Create OpenCV ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=self.edge_threshold,
            firstLevel=self.first_level,
            WTA_K=self.wta_k,
            patchSize=self.patch_size,
            fastThreshold=self.fast_threshold
        )
        
        # Pre-compute scale factors for each pyramid level
        self.scale_factors = np.array([
            self.scale_factor ** i for i in range(self.n_levels)
        ])
        self.inv_scale_factors = 1.0 / self.scale_factors
        
        # Pre-compute squared scale factors for covariance
        self.level_sigma2 = self.scale_factors ** 2
        self.inv_level_sigma2 = self.inv_scale_factors ** 2
        
    def _load_config(self, config_path: str):
        """Load parameters from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        features_cfg = config.get('features', {})
        self.n_features = features_cfg.get('n_features', self.n_features)
        self.scale_factor = features_cfg.get('scale_factor', self.scale_factor)
        self.n_levels = features_cfg.get('n_levels', self.n_levels)
        self.edge_threshold = features_cfg.get('edge_threshold', self.edge_threshold)
        self.first_level = features_cfg.get('first_level', self.first_level)
        self.wta_k = features_cfg.get('wta_k', self.wta_k)
        self.patch_size = features_cfg.get('patch_size', self.patch_size)
        self.fast_threshold = features_cfg.get('fast_threshold', self.fast_threshold)
        
    def extract(self, image: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract ORB features from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            mask: Optional mask to restrict detection region
            
        Returns:
            keypoints: List of detected keypoints
            descriptors: Array of shape (N, 32) with binary descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        
        # Handle case of no detections
        if descriptors is None:
            return [], np.array([])
            
        return keypoints, descriptors
    
    def extract_with_depth(self, image: np.ndarray,
                           depth_map: np.ndarray,
                           intrinsics: Dict[str, float],
                           mask: Optional[np.ndarray] = None) -> Tuple[List[KeyPoint], np.ndarray]:
        """
        Extract features with associated 3D positions from depth.
        
        This function back-projects each feature to 3D using the depth map,
        which is essential for 3D-2D PnP pose estimation.
        
        The back-projection equation:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth(u, v)
        
        Args:
            image: Input image
            depth_map: Depth map (in meters)
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy
            mask: Optional detection mask
            
        Returns:
            keypoints: List of KeyPoint objects with 3D positions
            descriptors: Binary descriptors
        """
        # Extract 2D features
        cv_keypoints, descriptors = self.extract(image, mask)
        
        if len(cv_keypoints) == 0:
            return [], np.array([])
        
        # Extract intrinsic parameters
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # Convert to enhanced keypoints with 3D
        keypoints = []
        valid_indices = []
        
        for i, kp in enumerate(cv_keypoints):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            
            # Boundary check
            if u < 0 or u >= depth_map.shape[1] or v < 0 or v >= depth_map.shape[0]:
                continue
                
            z = depth_map[v, u]
            
            # Skip invalid depths
            depth_min = intrinsics.get('depth_min', 0.2)
            depth_max = intrinsics.get('depth_max', 5.0)
            if z <= depth_min or z >= depth_max or not np.isfinite(z):
                continue
            
            # Back-project to 3D
            x = (kp.pt[0] - cx) * z / fx
            y = (kp.pt[1] - cy) * z / fy
            
            keypoint = KeyPoint(
                pt=kp.pt,
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id,
                depth=z,
                point_3d=np.array([x, y, z])
            )
            keypoints.append(keypoint)
            valid_indices.append(i)
        
        # Filter descriptors to match valid keypoints
        if len(valid_indices) > 0:
            descriptors = descriptors[valid_indices]
        else:
            descriptors = np.array([])
            
        return keypoints, descriptors
    
    def compute_features_per_level(self) -> np.ndarray:
        """
        Compute number of features to extract at each pyramid level.
        
        Features are distributed inversely proportional to scale^2,
        so that feature density is roughly constant across scales.
        
        Returns:
            Array of feature counts per level
        """
        # Factor to scale feature count: (1/s)^2 per level
        factor = 1.0 / (self.scale_factor ** 2)
        
        # Compute using geometric series sum
        n_desired_per_level = np.zeros(self.n_levels)
        n_desired_per_level[0] = self.n_features * (1 - factor) / (1 - factor ** self.n_levels)
        
        for level in range(1, self.n_levels):
            n_desired_per_level[level] = n_desired_per_level[level - 1] * factor
            
        return n_desired_per_level.astype(int)


class FeatureMatcher:
    """
    Feature matching with robust outlier rejection.
    
    This class implements:
    1. Brute-force matching with Hamming distance for ORB
    2. Lowe's ratio test for ambiguity rejection
    3. Cross-check (mutual nearest neighbor) filtering
    4. Geometric verification using fundamental/essential matrix RANSAC
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature matcher.
        
        Args:
            config_path: Path to YAML config file
        """
        # Default parameters
        self.ratio_threshold = 0.75
        self.cross_check = True
        self.max_distance = 50
        self.min_matches = 30
        
        # Geometric verification parameters
        self.ransac_reproj_threshold = 3.0
        self.ransac_confidence = 0.99
        
        if config_path is not None:
            self._load_config(config_path)
            
        # Create matchers
        # BFMatcher with Hamming distance for binary descriptors
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN matcher for faster matching (optional, for larger feature sets)
        # FLANN parameters for binary descriptors (LSH index)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,      # Number of hash tables
            key_size=12,         # Key size in bits
            multi_probe_level=1  # Multi-probe level
        )
        search_params = dict(checks=50)  # Number of checks
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    def _load_config(self, config_path: str):
        """Load parameters from config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        features_cfg = config.get('features', {})
        self.ratio_threshold = features_cfg.get('ratio_threshold', self.ratio_threshold)
        self.cross_check = features_cfg.get('cross_check', self.cross_check)
        self.max_distance = features_cfg.get('max_distance', self.max_distance)
        self.min_matches = features_cfg.get('min_matches', self.min_matches)
        
        odom_cfg = config.get('odometry', {})
        self.ransac_reproj_threshold = odom_cfg.get('ransac_reprojection_error', 
                                                     self.ransac_reproj_threshold)
        self.ransac_confidence = odom_cfg.get('ransac_confidence', self.ransac_confidence)
        
    def match(self, 
              desc1: np.ndarray, 
              desc2: np.ndarray,
              use_ratio_test: bool = True) -> List[FeatureMatch]:
        """
        Match features between two descriptor sets.
        
        Args:
            desc1: Descriptors from first image (query)
            desc2: Descriptors from second image (train)
            use_ratio_test: Whether to apply Lowe's ratio test
            
        Returns:
            List of FeatureMatch objects
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) == 0 or len(desc2) == 0:
            return []
            
        matches = []
        
        if use_ratio_test:
            # K-nearest neighbor matching (k=2 for ratio test)
            raw_matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            
            for match_pair in raw_matches:
                # Need at least 2 matches for ratio test
                if len(match_pair) < 2:
                    continue
                    
                m, n = match_pair[0], match_pair[1]
                
                # Lowe's ratio test
                if m.distance < self.ratio_threshold * n.distance:
                    # Additional distance threshold
                    if m.distance < self.max_distance:
                        matches.append(FeatureMatch(
                            query_idx=m.queryIdx,
                            train_idx=m.trainIdx,
                            distance=m.distance
                        ))
        else:
            # Simple nearest neighbor matching
            raw_matches = self.bf_matcher.match(desc1, desc2)
            
            for m in raw_matches:
                if m.distance < self.max_distance:
                    matches.append(FeatureMatch(
                        query_idx=m.queryIdx,
                        train_idx=m.trainIdx,
                        distance=m.distance
                    ))
        
        # Cross-check if enabled
        if self.cross_check and use_ratio_test:
            matches = self._cross_check_matches(matches, desc1, desc2)
            
        return matches
    
    def _cross_check_matches(self, 
                             matches: List[FeatureMatch],
                             desc1: np.ndarray,
                             desc2: np.ndarray) -> List[FeatureMatch]:
        """
        Apply cross-check: only keep mutual nearest neighbors.
        
        For each match (i -> j), verify that j -> i is also the best match.
        This significantly reduces false matches.
        """
        if len(matches) == 0:
            return []
            
        # Match in reverse direction
        reverse_matches = self.bf_matcher.knnMatch(desc2, desc1, k=2)
        
        # Build reverse match map
        reverse_map = {}
        for match_pair in reverse_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < self.ratio_threshold * n.distance:
                reverse_map[m.queryIdx] = m.trainIdx
        
        # Keep only cross-validated matches
        validated = []
        for match in matches:
            if match.train_idx in reverse_map:
                if reverse_map[match.train_idx] == match.query_idx:
                    validated.append(match)
                    
        return validated
    
    def geometric_verification(self,
                               pts1: np.ndarray,
                               pts2: np.ndarray,
                               intrinsics: Optional[Dict[str, float]] = None,
                               method: str = 'fundamental') -> Tuple[np.ndarray, np.ndarray]:
        """
        Verify matches using epipolar geometry.
        
        This function estimates the fundamental/essential matrix and uses it
        to identify geometrically consistent matches (inliers).
        
        The epipolar constraint:
            p2^T * F * p1 = 0  (fundamental matrix)
            p2^T * E * p1 = 0  (essential matrix, calibrated case)
        
        Args:
            pts1: Points from first image, shape (N, 2)
            pts2: Points from second image, shape (N, 2)
            intrinsics: Camera intrinsics (required for essential matrix)
            method: 'fundamental' or 'essential'
            
        Returns:
            matrix: Estimated F or E matrix
            inlier_mask: Boolean mask of inlier matches
        """
        if len(pts1) < 8:
            return None, np.zeros(len(pts1), dtype=bool)
            
        if method == 'essential' and intrinsics is not None:
            # Use essential matrix (requires calibrated camera)
            K = np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ])
            
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=self.ransac_confidence,
                threshold=self.ransac_reproj_threshold
            )
            return E, mask.ravel().astype(bool)
        else:
            # Use fundamental matrix (uncalibrated case)
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
                confidence=self.ransac_confidence
            )
            return F, mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts1), dtype=bool)


class BagOfWords:
    """
    Bag of Visual Words for place recognition in loop closure detection.
    
    The BoW approach:
    1. Build a vocabulary by clustering descriptors from training images
    2. For each image, compute a histogram of visual word occurrences
    3. Compare images using histogram similarity (e.g., L1 norm, cosine)
    
    We use a hierarchical k-means tree for efficient vocabulary construction
    and approximate nearest neighbor lookup.
    
    Reference:
        Gálvez-López & Tardós, "Bags of Binary Words for Fast Place Recognition
        in Image Sequences", IEEE T-RO, 2012
    """
    
    def __init__(self, vocabulary_size: int = 1000, 
                 vocabulary_depth: int = 6):
        """
        Initialize BoW model.
        
        Args:
            vocabulary_size: Number of cluster centers (visual words)
            vocabulary_depth: Depth of vocabulary tree (not used in flat clustering)
        """
        self.vocabulary_size = vocabulary_size
        self.vocabulary_depth = vocabulary_depth
        
        # Vocabulary (cluster centers)
        self.vocabulary: Optional[np.ndarray] = None
        
        # Database of BoW vectors
        self.database: Dict[int, np.ndarray] = {}
        
        # FLANN matcher for fast vocabulary lookup
        FLANN_INDEX_LSH = 6
        self.flann_index = None
        
    def train_vocabulary(self, 
                         descriptors_list: List[np.ndarray],
                         max_descriptors: int = 100000):
        """
        Train vocabulary from a collection of descriptors.
        
        Uses k-means clustering to find cluster centers (visual words).
        For binary descriptors, we use k-medoids or treat them as continuous.
        
        Args:
            descriptors_list: List of descriptor arrays from training images
            max_descriptors: Maximum descriptors to use (for memory)
        """
        print(f"Training BoW vocabulary with k={self.vocabulary_size}...")
        
        # Collect all descriptors
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
                
        if not all_descriptors:
            raise ValueError("No descriptors provided for vocabulary training")
            
        all_descriptors = np.vstack(all_descriptors)
        
        # Subsample if too many
        if len(all_descriptors) > max_descriptors:
            indices = np.random.choice(len(all_descriptors), max_descriptors, replace=False)
            all_descriptors = all_descriptors[indices]
        
        print(f"  Using {len(all_descriptors)} descriptors")
        
        # Convert to float for k-means (ORB descriptors are uint8)
        descriptors_float = all_descriptors.astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS
        
        compactness, labels, centers = cv2.kmeans(
            descriptors_float, 
            self.vocabulary_size,
            None, criteria, 10, flags
        )
        
        self.vocabulary = centers.astype(np.uint8)
        print(f"  Vocabulary trained with {len(self.vocabulary)} words")
        
        # Build FLANN index for fast lookup
        self._build_flann_index()
        
    def _build_flann_index(self):
        """Build FLANN index for vocabulary lookup."""
        if self.vocabulary is None:
            return
            
        # FLANN_INDEX_LSH = 6
        # index_params = dict(
        #     algorithm=FLANN_INDEX_LSH,
        #     table_number=6,
        #     key_size=12,
        #     multi_probe_level=1
        # )

        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=5
        )

        self.flann_index = cv2.flann.Index()
        self.flann_index.build(self.vocabulary.astype(np.float32), index_params)
        
    def compute_bow_vector(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute Bag of Words vector for a set of descriptors.
        
        The BoW vector is a normalized histogram of visual word occurrences.
        We use TF-IDF weighting for better discrimination:
            weight = tf * log(N / df)
        where tf = term frequency, df = document frequency, N = total docs
        
        For simplicity, we use just the normalized frequency (TF) here.
        
        Args:
            descriptors: Array of binary descriptors
            
        Returns:
            Normalized BoW histogram vector
        """
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary not trained. Call train_vocabulary first.")
            
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocabulary_size)
        
        # Find nearest visual word for each descriptor
        # Using brute force Hamming distance for binary descriptors
        bow_vector = np.zeros(self.vocabulary_size)
        
        for desc in descriptors:
            # Compute Hamming distances to all vocabulary words
            distances = np.zeros(len(self.vocabulary))
            for i, word in enumerate(self.vocabulary):
                # Hamming distance: count differing bits
                distances[i] = np.sum(np.unpackbits(desc ^ word))
                
            # Assign to nearest word
            nearest_word = np.argmin(distances)
            bow_vector[nearest_word] += 1
            
        # L1 normalize
        norm = np.sum(bow_vector)
        if norm > 0:
            bow_vector /= norm
            
        return bow_vector
    
    def add_to_database(self, image_id: int, bow_vector: np.ndarray):
        """Add a BoW vector to the database for later querying."""
        self.database[image_id] = bow_vector
        
    def query(self, bow_vector: np.ndarray, 
              top_k: int = 5,
              exclude_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Query database for similar images.
        
        Similarity is computed using L1 distance between BoW vectors.
        Lower distance = more similar.
        
        Args:
            bow_vector: Query BoW vector
            top_k: Number of top matches to return
            exclude_ids: Image IDs to exclude from search
            
        Returns:
            List of (image_id, similarity_score) tuples, sorted by similarity
        """
        if not self.database:
            return []
            
        exclude_ids = exclude_ids or []
        
        # Compute similarity to all database entries
        similarities = []
        for img_id, db_vector in self.database.items():
            if img_id in exclude_ids:
                continue
                
            # L1 distance (lower = more similar)
            # Convert to similarity: 1 - distance/2 (normalized to [0, 1])
            l1_dist = np.sum(np.abs(bow_vector - db_vector))
            similarity = 1.0 - l1_dist / 2.0
            
            similarities.append((img_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file."""
        if self.vocabulary is not None:
            np.save(filepath, self.vocabulary)
            print(f"Vocabulary saved to {filepath}")
            
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file."""
        self.vocabulary = np.load(filepath)
        self._build_flann_index()
        print(f"Vocabulary loaded: {len(self.vocabulary)} words")


def visualize_matches(img1: np.ndarray, kp1: List, 
                      img2: np.ndarray, kp2: List,
                      matches: List[FeatureMatch],
                      show_only_inliers: bool = True) -> np.ndarray:
    """
    Visualize feature matches between two images.
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints from each image
        matches: List of FeatureMatch objects
        show_only_inliers: Only show inlier matches
        
    Returns:
        Visualization image
    """
    # Convert custom KeyPoints to OpenCV KeyPoints if needed
    def to_cv_kp(kp):
        if isinstance(kp, KeyPoint):
            return cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, 
                              kp.response, kp.octave, kp.class_id)
        return kp
    
    cv_kp1 = [to_cv_kp(kp) for kp in kp1]
    cv_kp2 = [to_cv_kp(kp) for kp in kp2]
    
    # Convert matches to OpenCV DMatch
    cv_matches = []
    for m in matches:
        if show_only_inliers and not m.is_inlier:
            continue
        cv_matches.append(cv2.DMatch(m.query_idx, m.train_idx, m.distance))
    
    # Draw matches
    vis = cv2.drawMatches(
        img1, cv_kp1, img2, cv_kp2, cv_matches, None,
        matchColor=(0, 255, 0),      # Green for matches
        singlePointColor=(255, 0, 0), # Blue for unmatched
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return vis


if __name__ == "__main__":
    # Demo: Extract and match features between two test images
    import argparse
    
    parser = argparse.ArgumentParser(description='ORB Feature Extraction Demo')
    parser.add_argument('--image1', type=str, required=True, help='First image')
    parser.add_argument('--image2', type=str, required=True, help='Second image')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    args = parser.parse_args()
    
    # Load images
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    
    if img1 is None or img2 is None:
        print("Failed to load images")
        exit(1)
    
    # Initialize extractor and matcher
    extractor = ORBExtractor(args.config)
    matcher = FeatureMatcher(args.config)
    
    # Extract features
    print("Extracting features...")
    kp1, desc1 = extractor.extract(img1)
    kp2, desc2 = extractor.extract(img2)
    
    print(f"  Image 1: {len(kp1)} features")
    print(f"  Image 2: {len(kp2)} features")
    
    # Match features
    print("Matching features...")
    matches = matcher.match(desc1, desc2)
    print(f"  Initial matches: {len(matches)}")
    
    # Geometric verification
    pts1 = np.array([kp1[m.query_idx].pt for m in matches])
    pts2 = np.array([kp2[m.train_idx].pt for m in matches])
    
    F, inlier_mask = matcher.geometric_verification(pts1, pts2)
    
    for i, m in enumerate(matches):
        m.is_inlier = inlier_mask[i]
        
    n_inliers = np.sum(inlier_mask)
    print(f"  Inlier matches: {n_inliers}")
    
    # Visualize
    vis = visualize_matches(img1, kp1, img2, kp2, matches)
    
    cv2.imshow('Feature Matches', vis)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
