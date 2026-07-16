#!/bin/bash
# ==============================================================================
# PiSLAM: Edge AI Visual SLAM Launcher (Academic Demo)
# Author: Jinxin Shao
# ==============================================================================

# 定义颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   PiSLAM System Launch Protocol                            ${NC}"
echo -e "${BLUE}   Platform: Raspberry Pi 5 + Hailo-8L NPU                  ${NC}"
echo -e "${BLUE}============================================================${NC}"

# 1. 检查工作目录
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
# ============================================================
# 🟢 [新增] 强制指定 Qt/OpenCV 显示后端为 X11 (修复 Wayland 卡死问题)
# ============================================================
export QT_QPA_PLATFORM=xcb
export OPENCV_VIDEOIO_PRIORITY_MSMF=0

# 3. 检查 NPU 设备
if lspci | grep -q "Hailo"; then
    echo -e "${GREEN}[INFO] Hailo NPU detected.${NC}"
    MODE="hailo"
else
    echo -e "${RED}[WARN] Hailo NPU NOT detected! Falling back to dataset mode.${NC}"
    MODE="dataset"
fi

# 4. 配置参数 (学术展示模式)
CONFIG_FILE="slam_config_enhanced.yaml"
HEF_MODEL="scdepthv3.hef"
OUTPUT_DIR="$HOME/UW_pislam_output"

# ================= 启动辅助模块 =================

# 定义清理函数：退出时杀死所有后台进程
cleanup() {
    echo -e "\n${RED}[STOP] System shutdown sequence initiated...${NC}"
    
    if [ -n "$IMU_PID" ]; then
        echo -e "   - Killing IMU Driver (PID $IMU_PID)..."
        kill $IMU_PID 2>/dev/null
    fi
    
    if [ -n "$DASH_PID" ]; then
        echo -e "   - Killing Dashboard (PID $DASH_PID)..."
        kill $DASH_PID 2>/dev/null
    fi
    
    exit
}
trap cleanup INT TERM

echo -e "${CYAN}[INFO] Launching Peripheral Drivers...${NC}"

# 启动 IMU 驱动 (后台运行)
# 注意：如果您的USB端口不是 /dev/imu_usb，请先修改 wheeltec_full_driver.py 或在此处设置软链接
if [ -f "wheeltec_full_driver.py" ]; then
    python3 wheeltec_full_driver.py > /dev/null 2>&1 & 
    IMU_PID=$!
    echo -e "${GREEN}[OK] IMU Driver started (PID $IMU_PID)${NC}"
    sleep 1 # 等待驱动初始化
else
    echo -e "${RED}[WARN] IMU driver not found! Skipping.${NC}"
fi

# 启动 IMU 可视化仪表盘 (后台运行)
if [ -f "imu_dashboard.py" ]; then
    echo -e "${CYAN}[INFO] Starting IMU Real-time Dashboard...${NC}"
    python3 imu_dashboard.py &
    DASH_PID=$!
else
    echo -e "${RED}[WARN] Dashboard script not found. Skipping visualization.${NC}"
fi

# ===============================================

# 6. 启动 SLAM 核心系统
echo -e "${GREEN}[INFO] Starting PiSLAM core...${NC}"
echo -e "${BLUE}   > Mode: $MODE${NC}"
echo -e "${BLUE}   > Config: $CONFIG_FILE${NC}"
echo -e "${BLUE}   > Viz: Enabled (Dashboard & Map)${NC}"

# 运行 Python 主程序
python3 run_slam.py \
    --mode "$MODE" \
    --config "$CONFIG_FILE" \
    --hef "$HEF_MODEL" \
    --camera 0 

# 7. 运行结束提示
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Session completed. Data saved to $OUTPUT_DIR${NC}"
    echo -e "${BLUE}[HINT] Run './view_map.sh' to visualize the 3D map.${NC}"
else
    echo -e "${RED}[ERROR] System crashed or exited with error.${NC}"
fi

# 手动调用清理（如果是正常退出）
cleanup

# 4. 保持窗口
# read -p "Press Enter to close..."