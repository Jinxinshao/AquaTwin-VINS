import open3d as o3d
import numpy as np
import sys
import os


# 在导入其他图形库之前设置
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
# ==============================================================================
# 配置区域
# ==============================================================================
MAP_FILE = os.path.expanduser("~/New_qingshui/UW_pislam_output/map.ply")
TRAJ_FILE = os.path.expanduser("~/New_qingshui/UW_pislam_output/trajectory.txt")

def create_gradient_trajectory(points):
    """
    [学术技巧] 生成带有时间梯度的轨迹线
    颜色从 蓝色(Start) -> 青色 -> 黄色 -> 红色(End) 渐变
    """
    num_points = len(points)
    if num_points < 2:
        return None

    lines = [[i, i + 1] for i in range(num_points - 1)]
    colors = []
    
    # 简单的伪彩色映射 (Jet-like colormap simulation)
    for i in range(num_points - 1):
        ratio = i / max(1, num_points - 1)
        # r, g, b
        r = max(0, min(1, 2 * ratio - 0.5)) # Red ramps up late
        b = max(0, min(1, 1.5 - 2 * ratio)) # Blue ramps down early
        g = max(0, min(1, 1 - 2 * abs(ratio - 0.5))) # Green peaks in middle
        colors.append([r, g, b])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main():
    print("🎨 Initializing Academic 3D Viewer...")
    geometries = []

    # 1. 加载地图点云
    if os.path.exists(MAP_FILE):
        print(f"Loading Map: {MAP_FILE}")
        pcd = o3d.io.read_point_cloud(MAP_FILE)
        if not pcd.is_empty():
            geometries.append(pcd)
        else:
            print("⚠️ Map point cloud is empty.")
    
    # 2. 加载并美化轨迹
    if os.path.exists(TRAJ_FILE):
        print(f"Loading Trajectory: {TRAJ_FILE}")
        try:
            data = np.loadtxt(TRAJ_FILE)
            # TUM 格式: timestamp tx ty tz qx qy qz qw
            # 我们只需要 tx, ty, tz (第1, 2, 3列)
            points = data[:, 1:4] 
            
            traj_lines = create_gradient_trajectory(points)
            if traj_lines:
                geometries.append(traj_lines)
                
                # [锦上添花] 添加起点和终点的球体标记
                # 起点：蓝色球
                start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                start_sphere.paint_uniform_color([0, 0, 1])
                start_sphere.translate(points[0])
                geometries.append(start_sphere)
                
                # 终点：红色球
                end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                end_sphere.paint_uniform_color([1, 0, 0])
                end_sphere.translate(points[-1])
                geometries.append(end_sphere)
                
        except Exception as e:
            print(f"❌ Error parsing trajectory: {e}")

    # 3. 添加坐标轴 (RGB = XYZ, 长度 0.5m)
    # 这在学术展示中必须有，用以说明尺度
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(axis)

    if not geometries:
        print("Nothing to show!")
        return

    # 4. [核心] 高级渲染器配置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PiSLAM Result (Academic)", width=1280, height=720)
    
    for g in geometries:
        vis.add_geometry(g)

    # 获取渲染选项并修改
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # ⚪️ 纯白背景
    opt.point_size = 3.0                         # 增大点云，使其更明显
    opt.line_width = 50.0                        # 尝试加粗线条 (Open3D对线条宽度的支持取决于OpenGL版本)
    opt.show_coordinate_frame = False            # 我们已经手动加了一个更好看的坐标轴

    print("\n---------------------------------------")
    print("🖱️  Controls:")
    print("   [Left Click + Drag]: Rotate")
    print("   [Ctrl + Left Drag]:  Pan")
    print("   [Wheel]:             Zoom")
    print("   [P]:                 Take Screenshot")
    print("---------------------------------------")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()