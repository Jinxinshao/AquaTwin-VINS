import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
TRAJ_FILE = os.path.expanduser("~/UW_pislam_output/trajectory.txt")
OUTPUT_IMG = os.path.expanduser("~/UW_pislam_output/trajectory_3views.png")
# ===========================================

def plot_3views_academic():
    print("🎨 正在生成学术级三视图轨迹 (Generating 3-View Plot)...")
    
    if not os.path.exists(TRAJ_FILE):
        print(f"❌ 找不到文件: {TRAJ_FILE}")
        return

    try:
        # 1. 加载数据 (TUM 格式: timestamp, tx, ty, tz, qx, qy, qz, qw)
        data = np.loadtxt(TRAJ_FILE)
        
        # 提取坐标
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]
        
        # 2. 创建画布 (1行3列)
        # figsize 控制图片比例，(18, 6) 适合插入论文的宽图模式
        plt.style.use('seaborn-v0_8-paper') # 使用适合论文的样式（如果没有安装 seaborn，matplotlib 会自动回退）
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # 颜色随时间渐变
        time_color = range(len(x))
        cmap = 'plasma' # 这种配色在黑白打印时也比较容易分辨深浅
        
        # ==========================================
        # 子图 1: X-Y 平面 (通常是俯视图/Top View)
        # ==========================================
        ax1 = axs[0]
        sc1 = ax1.scatter(x, y, c=time_color, cmap=cmap, s=2, alpha=0.5)
        ax1.plot(x[0], y[0], 'g^', markersize=10, markeredgecolor='black', label='Start') # 起点
        ax1.plot(x[-1], y[-1], 'r*', markersize=12, markeredgecolor='black', label='End') # 终点
        ax1.set_title('Top View (X-Y)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.axis('equal') # 关键！保证比例尺一致
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # ==========================================
        # 子图 2: X-Z 平面 (通常是侧视图/Side View)
        # ==========================================
        ax2 = axs[1]
        ax2.scatter(x, z, c=time_color, cmap=cmap, s=2, alpha=0.5)
        ax2.plot(x[0], z[0], 'g^', markersize=10, markeredgecolor='black')
        ax2.plot(x[-1], z[-1], 'r*', markersize=12, markeredgecolor='black')
        ax2.set_title('Side View (X-Z)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.axis('equal')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # ==========================================
        # 子图 3: Y-Z 平面 (通常是前视图/Front View)
        # ==========================================
        ax3 = axs[2]
        # 这里加个 colorbar 关联到最后一个图，显示时间流逝
        sc3 = ax3.scatter(y, z, c=time_color, cmap=cmap, s=2, alpha=0.5)
        ax3.plot(y[0], z[0], 'g^', markersize=10, markeredgecolor='black')
        ax3.plot(y[-1], z[-1], 'r*', markersize=12, markeredgecolor='black')
        ax3.set_title('Front View (Y-Z)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.axis('equal')
        ax3.grid(True, linestyle=':', alpha=0.6)

        # 添加公共的 Colorbar
        cbar = fig.colorbar(sc3, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('Time Steps (Trajectory Order)', fontsize=10)

        # 3. 调整布局并保存
        plt.suptitle('Estimated Trajectory Multi-View Projection', fontsize=16, y=0.98)
        # plt.tight_layout() # 自适应布局
        
        # 保存为高分辨率 PNG
        plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
        print(f"✅ 三视图已保存至: {OUTPUT_IMG}")
        print("   (包含 X-Y, X-Z, Y-Z 三个平面的投影，并带有时间热力图)")

    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_3views_academic()