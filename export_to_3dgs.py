#!/usr/bin/env python3
"""
================================================================================
PiSLAM to 3D Gaussian Splatting (COLMAP Format) Converter
================================================================================
"""

import os
import json
import shutil
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Convert PiSLAM results to 3DGS/COLMAP format")
    parser.add_argument("--slam_dir", type=str, default=os.path.expanduser("/home/shaojx8/pislam_output"),
                        help="Path to PiSLAM output directory")
    parser.add_argument("--output_dir", type=str, default="/home/shaojx8/pislam_output/3dgs_data",
                        help="Path where to save the converted dataset")
    # Camera intrinsics (Default for scDepth settings)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    
    args = parser.parse_args()

    # 1. Setup Directories
    kf_dir = os.path.join(args.slam_dir, "keyframes")
    map_path = os.path.join(args.slam_dir, "map.ply")
    
    if not os.path.exists(kf_dir):
        print(f"❌ Error: Keyframes directory not found at {kf_dir}")
        print("   -> Did you run 'run_slam.py' successfully?")
        return

    input_images_dir = os.path.join(args.output_dir, "input")
    sparse_dir = os.path.join(args.output_dir, "sparse", "0")
    
    os.makedirs(input_images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    print(f"🚀 Converting dataset for 3D Gaussian Splatting...")
    print(f"📂 Source: {args.slam_dir}")
    print(f"📂 Target: {args.output_dir}")

    # 2. Write cameras.txt
    print("📸 Writing cameras.txt...")
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {args.width} {args.height} {args.fx} {args.fy} {args.cx} {args.cy}\n")

    # 3. Process Keyframes -> images.txt
    print("🖼️  Processing keyframes & writing images.txt...")
    json_files = sorted([f for f in os.listdir(kf_dir) if f.endswith(".json")])
    
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        for idx, json_file in enumerate(tqdm(json_files)):
            # Load pose
            with open(os.path.join(kf_dir, json_file), "r") as jf:
                meta = json.load(jf)
            
            # SLAM Pose (Camera-to-World) -> COLMAP Pose (World-to-Camera)
            pose_cw = np.array(meta['pose']).reshape(4, 4)
            try:
                pose_wc = np.linalg.inv(pose_cw)
            except np.linalg.LinAlgError:
                continue
                
            # Rotation (Quaternion w,x,y,z) & Translation
            R_wc = pose_wc[:3, :3]
            t_wc = pose_wc[:3, 3]
            r = Rotation.from_matrix(R_wc)
            qx, qy, qz, qw = r.as_quat()
            
            # Copy Image
            img_name = json_file.replace(".json", ".jpg")
            src_img = os.path.join(kf_dir, img_name)
            dst_img = os.path.join(input_images_dir, img_name)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                # ImageID, QW, QX, QY, QZ, TX, TY, TZ, CamID, Name
                f.write(f"{idx+1} {qw} {qx} {qy} {qz} {t_wc[0]} {t_wc[1]} {t_wc[2]} 1 {img_name}\n")
                f.write("\n")

    # 4. Convert Point Cloud -> points3D.txt
    print("☁️  Converting map.ply to points3D.txt...")
    if os.path.exists(map_path):
        # Force CPU rendering to avoid GLX error
        try:
            pcd = o3d.io.read_point_cloud(map_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            if colors.max() <= 1.0: colors = (colors * 255).astype(np.uint8)
            
            with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
                for i in tqdm(range(len(points))):
                    pt = points[i]
                    c = colors[i] if len(colors) > i else [255, 255, 255]
                    f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {c[0]} {c[1]} {c[2]} 0 \n")
        except Exception as e:
            print(f"⚠️ Could not process point cloud (headless mode issue?): {e}")
            # Create empty file as fallback
            with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f: pass
    else:
        print("⚠️ map.ply not found.")
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f: pass

    print("\n✅ Conversion Complete!")
    print(f"📦 Please download this folder to your PC: {args.output_dir}")

if __name__ == "__main__":
    main()