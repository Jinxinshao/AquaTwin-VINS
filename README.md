1、 基本说明

该示例程序提供了使用python解析YESENSE产品的标准私有协议输出

2、安装与测试

2.1、安装要求

python >= 3.8.0
pyserial >= 3.4 #### 如需要安装可以使用命令'pip install pyserial'

2.2、测试环境说明

##################################
windows 10 21H2
python: 3.12.0
pyserial: 3.5 

##################################
ubuntu 22.04
python: 3.10.12
pyserial: 3.5 

3、使用说明

3.1、程序说明

main.py: 主程序入口
yis_std_dec.py：YESENSE标准二进制协议解析模块
port_manager.py：端口管理模块

3.2、查看帮助

打开命令行或终端进入main.py脚本目录，使用python main.py --help或python main.py -h查看如何使用脚本

3.3、运行脚本

Linux系统：
`$ python3 main.py --port /dev/ttyUSB0 --bps 460800`

`$ python3 main.py --port /dev/ttyUSB0 --bps 460800 --dbg `

在linux系统通常会遇到串口权限不够，可以通过以下方法来解决
a：临时修改权限，sudo chmod 777 /dev/ttyUSBxx，其中/dev/ttyUSBxx为对应的串口设备名，还有可能是/dev/ttyACMxx和/dev/ttySCxx
b：永久修改权限，sudo usermode -aG dialout 用户名，用户名为系统当前用户名，可以用whoami来查看

Windows：
`$ python3 main.py --port COM6 --bps 460800 `

`$ python3 main.py --port COM6 --bps 460800 --dbg `

其中选项port为必选，bps和dbg选项为可选，bps默认值为460800，dbg默认值为False

3.4、输出数据说明

该例程支持解析内容为：帧序号、加速度、角速度、传感器温度、欧拉角、四元数、磁场强度原始值和归一化值、位置、UTC、采样时间戳和dataready时间戳、速度(NED)、组合状态

解析的结果存在以下内容中

yis_out['tid']:			帧序号
yis_out['roll']:		横滚角，单位°
yis_out['pitch']:		俯仰角，单位°
yis_out['yaw']:			航向角，单位°
yis_out['q0']:			四元数q0，单位无
yis_out['q1']:			四元数q1，单位无
yis_out['q2']:			四元数q2，单位无
yis_out['q3']:			四元数q3，单位无
yis_out['sensor_temp']:	传感器温度，单位℃
yis_out['acc_x']:		加速度X轴数据，单位m/s2
yis_out['acc_y']:		加速度Y轴数据，单位m/s2
yis_out['acc_z']:		加速度Z轴数据，单位m/s2
yis_out['gyro_x']:		角速度X轴数据，单位dps
yis_out['gyro_y']:		角速度Y轴数据，单位dps
yis_out['gyro_z']:		角速度Z轴数据，单位dps
yis_out['norm_mag_x']:	归一化磁场X轴数据，单位无
yis_out['norm_mag_y']:	归一化磁场Y轴数据，单位无
yis_out['norm_mag_z']:	归一化磁场Z轴数据，单位无
yis_out['raw_mag_x']:	磁场原始X轴数据，单位mGauss
yis_out['raw_mag_y']:	磁场原始Y轴数据，单位mGauss
yis_out['raw_mag_z']:	磁场原始Z轴数据，单位mGauss
yis_out['lat']:			纬度，单位°
yis_out['longt']:		经度，单位°
yis_out['alt']:			高度，单位m
yis_out['vel_e']:		东向速度，单位m/s
yis_out['vel_n']:		北向速度，单位m/s
yis_out['vel_u']:		天向速度，单位m/s
yis_out['ms']:			毫秒
yis_out['year']:		年
yis_out['month']:		月
yis_out['day']:			日
yis_out['hour']:		时（0时区的小时，如果是北京时间需要加8）
yis_out['minute']:		分
yis_out['second']:		秒
yis_out['smp_timestamp']:	采样时间戳
yis_out['ready_timestamp']:	数据完成时间戳
yis_out['status']:0			组合导航状态