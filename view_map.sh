#!/bin/bash
# ==============================================================================
# PiSLAM: 3D Map & Trajectory Viewer (Robust Version)
# Visualizes 'map.ply' and 'trajectory.txt' using Open3D
# ==============================================================================

# 1. 尝试激活环境 (适配多种安装路径)
# 注意：在脚本中直接用 conda activate 往往无效，需要用 source activate
export LIBGL_ALWAYS_SOFTWARE=1
source activate torch24
cd /home/shaojx8/SLAM_ins
if [ -f "$HOME/miniforge3/bin/activate" ]; then
    source "$HOME/miniforge3/bin/activate" torch24
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    source "$HOME/anaconda3/bin/activate" torch24
else
    echo "⚠️  Conda activation script not found. Assuming environment is already active."
fi

# 2. 定义路径
OUTPUT_DIR="$HOME/pislam_output"
MAP_FILE="$OUTPUT_DIR/map.ply"
TRAJ_FILE="$OUTPUT_DIR/trajectory.txt"

echo "============================================================"
echo "   PiSLAM 3D Viewer (Debug Mode)"
echo "============================================================"

# 3. 检查文件是否存在（但不退出，因为可能只想看轨迹）
if [ ! -f "$MAP_FILE" ]; then
    echo "⚠️  Warning: Map file not found at $MAP_FILE"
fi

echo "✅ Target Map: $MAP_FILE"
echo "✅ Target Traj: $TRAJ_FILE"
echo "🎮 Controls:"
echo "   [Mouse Left]: Rotate  |  [Mouse Right]: Pan"
echo "   [Mouse Wheel]: Zoom   |  [Q]: Quit"

# 4. 生成 Python 可视化脚本
cat <<EOF > .temp_viewer.py
import open3d as o3d
import numpy as np
import sys
import os

def main():
    # 开启详细日志
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    
    map_file = "$MAP_FILE"
    traj_file = "$TRAJ_FILE"
    
    geometries = []
    has_data = False
    
    # --- 1. 加载点云地图 ---
    if os.path.exists(map_file):
        try:
            print(f"-> Loading Map: {map_file}")
            pcd = o3d.io.read_point_cloud(map_file)
            if not pcd.is_empty():
                print(f"   Points: {len(pcd.points)}")
                # 稍微给点颜色增强，方便看清
                if not pcd.has_colors():
                    pcd.paint_uniform_color([0.5, 0.5, 0.5])
                geometries.append(pcd)
                has_data = True
            else:
                print("⚠️  Map is empty (0 points).")
        except Exception as e:
            print(f"❌ Failed to load map: {e}")
    else:
        print("⚠️  Map file does not exist.")

    # --- 2. 加载轨迹 ---
    if os.path.exists(traj_file):
        try:
            print(f"-> Loading Trajectory: {traj_file}")
            lines = open(traj_file).readlines()
            points = []
            for line in lines:
                if line.startswith('#'): continue
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 4: # timestamp tx ty tz ...
                    points.append(parts[1:4])
            
            if len(points) > 1:
                print(f"   Trajectory nodes: {len(points)}")
                points = np.array(points)
                lines_idx = [[i, i+1] for i in range(len(points)-1)]
                colors = [[1, 0, 0] for _ in range(len(lines_idx))] # 红色轨迹
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines_idx)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(line_set)
                has_data = True
            else:
                print("⚠️  Trajectory is too short.")
        except Exception as e:
            print(f"❌ Failed to load trajectory: {e}")

    # --- 3. 兜底处理 (防止闪退) ---
    # 如果没有任何数据，添加一个坐标轴，确保窗口能打开
    if not has_data:
        print("\n⚠️  NO DATA TO SHOW! Adding a coordinate frame so window opens.")
        print("   (This usually means SLAM didn't produce a map or trajectory)")
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        geometries.append(mesh_frame)

    # --- 4. 启动可视化 ---
    print("\n-> Launching Open3D Visualizer...")
    try:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="PiSLAM 3D Result",
            width=1024, height=768,
            left=50, top=50,
            point_show_normal=False
        )
    except Exception as e:
        print(f"\n❌ GUI Crash: {e}")
        print("   Note: Ensure you are running this on a desktop with a display (or VNC).")

if __name__ == "__main__":
    main()
EOF

# 5. 运行 Python 脚本
python3 .temp_viewer.py

# 6. 错误处理
if [ $? -ne 0 ]; then
    echo "------------------------------------------------------------"
    echo "❌ Viewer crashed. Please check the python errors above."
    echo "   If seeing 'Open3D import error', check your environment."
fi

# 清理 (可选：注释掉下面这行以便调试)
rm .temp_viewer.py