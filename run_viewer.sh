#!/bin/bash

# 1. 加载 Conda 环境
if [ -f "/home/shaojx8/miniforge3/etc/profile.d/conda.sh" ]; then
    source /home/shaojx8/miniforge3/etc/profile.d/conda.sh
else
    echo "Error: Conda not found"
    exit 1
fi

# 2. 激活环境
conda activate torch24 
cd /home/shaojx8/UW_SLAM_ins/

# =======================================================
# 🔧 [关键修复] 强制使用软件渲染，防止 GLX 报错
# 这行命令告诉系统：“别找显卡了，直接用 CPU 算画面吧”
# =======================================================
export LIBGL_ALWAYS_SOFTWARE=1

# 3. 运行学术查看器
echo "🎨 Launching Academic Viewer..."
python3 plot_paper_3views.py
python3 view_academic_3d.py


# 4. 保持窗口
read -p "Press Enter to close..."