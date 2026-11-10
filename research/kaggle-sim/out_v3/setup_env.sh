#!/bin/bash
# Kaggle (Ubuntu 24.04, T4) reproduction of hackathon/docker/Dockerfile without Docker:
#  - NVIDIA user-space graphics libs matching the host kernel driver (Vulkan ICD)
#  - Isaac Sim 5.0 + Isaac Lab 2.2 (pip, python 3.11, micromamba env "isaac")
#  - ROS 2 Humble from RoboStack (micromamba env "ros") + bodyctrl_msgs rebuilt from the organiser .deb's own .msg/.srv
set -x
W=/kaggle/working; T=/tmp/w; mkdir -p $T $W/logs; cd $T
export MAMBA_ROOT_PREFIX=$T/mm
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee $W/logs/gpu.txt
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
# 1) driver user-space (graphics) libs of exactly the host version
curl -sfL -o drv.run https://download.nvidia.com/XFree86/Linux-x86_64/$DRV/NVIDIA-Linux-x86_64-$DRV.run && sh drv.run --extract-only --target $T/drv >/dev/null
mkdir -p $T/nvlib; find $T/drv -maxdepth 1 -name 'lib*.so*' -exec cp {} $T/nvlib/ \;
cd $T/nvlib; for f in *.so.$DRV; do b=${f%.$DRV}; ln -sf $f $b.1 2>/dev/null; ln -sf $f $b.0 2>/dev/null; ln -sf $f $b; done; cd $T
cat > $T/nvidia_icd.json <<J
{"file_format_version":"1.0.1","ICD":{"library_path":"$T/nvlib/libGLX_nvidia.so.0","api_version":"1.4.312"}}
J
mkdir -p $T/egl; cat > $T/egl/10_nvidia.json <<J
{"file_format_version":"1.0.0","ICD":{"library_path":"$T/nvlib/libEGL_nvidia.so.0"}}
J
apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libglu1-mesa libxt6t64 libxrandr2 libxinerama1 libxcursor1 libxi6 libvulkan1 libsm6 libice6 libxkbcommon0 libegl1 libgl1 > $W/logs/apt.log 2>&1
# 2) micromamba
curl -sfL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $T bin/micromamba
MM=$T/bin/micromamba
( $MM create -y -q -n ros -c robostack-staging -c conda-forge python=3.11 ros-humble-ros-base ros-humble-robot-state-publisher \
   ros-humble-cv-bridge ros-humble-tf2-ros ros-humble-image-transport ros-humble-rosidl-default-generators \
   colcon-common-extensions cmake make compilers ninja pkg-config yaml-cpp pyzmq pyyaml "numpy<2" scipy > $W/logs/ros_env.log 2>&1
  echo ros_env_rc=$? >> $W/logs/ros_env.log ) &
RP=$!
( $MM create -y -q -n isaac -c conda-forge python=3.11 pip > $W/logs/isaac_env.log 2>&1
  P=$MAMBA_ROOT_PREFIX/envs/isaac/bin/python
  $P -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 >> $W/logs/isaac_env.log 2>&1
  $P -m pip install -q -U pip "setuptools<70" wheel >> $W/logs/isaac_env.log 2>&1
  $P -m pip install -q --no-build-isolation flatdict==4.0.1 >> $W/logs/isaac_env.log 2>&1
  $P -m pip install -q "isaaclab[isaacsim,all]==2.2.0" --extra-index-url https://pypi.nvidia.com >> $W/logs/isaac_env.log 2>&1
  $P -m pip install -q pyzmq "numpy<2" "packaging==24.2" pyyaml opencv-python-headless >> $W/logs/isaac_env.log 2>&1
  echo isaac_env_rc=$? >> $W/logs/isaac_env.log ) &
IP=$!
# 3) repo copy (dataset is read-only)
SRC=$(find /kaggle/input -maxdepth 5 -type d -name polygon | head -1); echo SRC=$SRC; ls /kaggle/input; mkdir -p /tmp/nssr_sim && cp -r $SRC/. /tmp/nssr_sim/ ; ln -sfn /tmp/nssr_sim /nssr_sim
wait $RP; wait $IP
# 4) bodyctrl_msgs from the deb's own interface files
mkdir -p $T/bc $T/ws/src/bodyctrl_msgs && dpkg-deb -x /tmp/nssr_sim/teleoperation/msgs/ros-humble-bodyctrl-msgs_0.0.1-1_amd64.deb $T/bc
S=$T/bc/opt/ros/humble/share/bodyctrl_msgs; D=$T/ws/src/bodyctrl_msgs; mkdir -p $D/msg $D/srv
cp $S/msg/*.msg $D/msg/; cp $S/srv/*.srv $D/srv/; cp $S/package.xml $D/
python3 - "$D" <<'PY'
import sys,os,glob;d=sys.argv[1]
msgs=sorted('msg/'+os.path.basename(f) for f in glob.glob(d+'/msg/*.msg'));srvs=sorted('srv/'+os.path.basename(f) for f in glob.glob(d+'/srv/*.srv'))
# drop Request/Response msgs generated from srv
msgs=[m for m in msgs if not (m.endswith('_Request.msg') or m.endswith('_Response.msg'))]
open(d+'/CMakeLists.txt','w').write("""cmake_minimum_required(VERSION 3.8)
project(bodyctrl_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
%s
DEPENDENCIES std_msgs geometry_msgs builtin_interfaces sensor_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
"""%'\n'.join('"%s"'%x for x in msgs+srvs))
PY
eval "$($MM shell hook -s bash)"; micromamba activate ros
cd $T/ws && colcon build --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=$(which python) > $W/logs/bodyctrl_build.log 2>&1; echo bodyctrl_rc=$? | tee -a $W/logs/bodyctrl_build.log
source $T/ws/install/setup.bash && python -c "from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg; from bodyctrl_msgs.srv import SetSpeed; print('bodyctrl import OK')" | tee -a $W/logs/bodyctrl_build.log
micromamba deactivate
# 5) simulator package (as in docker/run.sh init)
P=$MAMBA_ROOT_PREFIX/envs/isaac/bin/python
cd /tmp/nssr_sim/source && $P -m pip install -q --no-deps -e . > $W/logs/nssr_pkg.log 2>&1; $P -c 'import nssr_sim; print("nssr_sim OK")' >> $W/logs/nssr_pkg.log 2>&1
df -h /tmp | tail -1 >> $W/logs/gpu.txt
