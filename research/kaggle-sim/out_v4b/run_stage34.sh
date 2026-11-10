#!/bin/bash
# Stage 3-4: start sim headless (as start_sim.sh: bridge in ROS python, sim_runner in Isaac python), run check_interfaces.py, dump TF/camera_info.
set -x
W=/kaggle/working; T=/tmp/w; export MAMBA_ROOT_PREFIX=$T/mm; MM=$T/bin/micromamba; cd /tmp/nssr_sim
export ROS_DOMAIN_ID=146 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES
export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/nssr_sim/hackathon/docker/fastdds_hack.xml
export HACK_TASK=${HACK_TASK:-sort3} HACK_POOL=${HACK_POOL:-3} HACK_SEED=${HACK_SEED:-1} HACK_HEADLESS=1 HACK_RES=${HACK_RES:-1280x720} HACK_RENDER_INTERVAL=3
export VK_ICD_FILENAMES=$T/nvidia_icd.json VK_DRIVER_FILES=$T/nvidia_icd.json __EGL_VENDOR_LIBRARY_DIRS=$T/egl
ISAAC_LD=$T/nvlib
rosenv(){ ( eval "$($MM shell hook -s bash)"; micromamba activate ros; source $T/ws/install/setup.bash; "$@" ); }
# bridge (ROS side)
rosenv python teleoperation/bridges/ros2_zmq_bridge.py > $W/logs/bridge.log 2>&1 &
BR=$!; sleep 3
KIT="--/rtx/raytracing/fractionalCutoutOpacity=true --/rtx/pathtracing/fractionalCutoutOpacity=true --/rtx/translucency/maxRefractionBounces=4 --/app/runLoops/main/rateLimitEnabled=false --/app/renderer/resolution/width=${HACK_RES%x*} --/app/renderer/resolution/height=${HACK_RES#*x}"
export HACK_KIT_ARGS="$KIT"
LD_LIBRARY_PATH=$ISAAC_LD:$LD_LIBRARY_PATH $MAMBA_ROOT_PREFIX/envs/isaac/bin/python scripts/sim_runner.py --task NSSRSim-TiangongPro-Parlor-v0 --enable_cameras --num_envs 1 \
  --perf_stats --step_hz 100 --headless --device cpu > $W/logs/sim_${HACK_TASK}_p${HACK_POOL}_s${HACK_SEED}.log 2>&1 &
SIM=$!
# wait for /sim/task
D=$((SECONDS+900)); for i in $(seq 1 120); do [ $SECONDS -gt $D ] && break
  if rosenv timeout 5 ros2 topic echo --once --qos-durability transient_local /sim/task std_msgs/msg/String > $W/logs/sim_task.txt 2>/dev/null; then break; fi
  kill -0 $SIM 2>/dev/null || break; sleep 5
done
echo "waited $((i*10))s"; cat $W/logs/sim_task.txt
rosenv timeout 600 python hackathon/tools/check_interfaces.py > $W/logs/check_interfaces.log 2>&1; echo check_rc=$? >> $W/logs/check_interfaces.log
rosenv timeout 20 ros2 topic echo --once /cam_head/color/camera_info > $W/logs/caminfo_head_color.txt 2>&1
rosenv timeout 20 ros2 topic echo --once /cam_head/depth/camera_info > $W/logs/caminfo_head_depth.txt 2>&1
rosenv timeout 20 ros2 topic echo --once /cam_torso/color/camera_info > $W/logs/caminfo_torso_color.txt 2>&1
rosenv timeout 30 ros2 run tf2_ros tf2_echo pelvis camera_head_link > $W/logs/tf_pelvis_camhead.txt 2>&1
rosenv timeout 20 ros2 topic echo --once /tf_static > $W/logs/tf_static.txt 2>&1
rosenv timeout 20 ros2 topic list -t > $W/logs/topics.txt 2>&1
rosenv timeout 20 ros2 topic echo --once /joint_states > $W/logs/joint_states.txt 2>&1
rosenv timeout 60 python /kaggle/working/grab_frames.py > $W/logs/grab.log 2>&1
kill -INT $SIM; sleep 10; kill $BR; pkill -f sim_node.py; pkill -f camera_bridge.py; pkill -f robot_state_publisher; pkill -9 -f sim_runner.py
