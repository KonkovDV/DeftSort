#!/bin/bash
# Pool-3 sorting runs: start sim once per task variant, then per layout: randomize(seed) -> run.sh (fresh process) -> /sim/task/check.
pkill -9 -f ros2_zmq_bridge; pkill -9 -f camera_bridge.py; pkill -9 -f sim_node.py; pkill -9 -f sim_runner.py; sleep 3
W=/kaggle/working; exec 2> >(tee -a /kaggle/working/logs/run_sort_trace.log >&2); set -x; T=/tmp/w; export MAMBA_ROOT_PREFIX=$T/mm; MM=$T/bin/micromamba; cd /tmp/nssr_sim
export ROS_DOMAIN_ID=146 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES
export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/nssr_sim/hackathon/docker/fastdds_hack.xml
export VK_ICD_FILENAMES=$T/nvidia_icd.json VK_DRIVER_FILES=$T/nvidia_icd.json __EGL_VENDOR_LIBRARY_DIRS=$T/egl
rosenv(){ ( eval "$($MM shell hook -s bash)"; micromamba activate ros; source $T/ws/install/setup.bash; [ -f /tmp/sol/install/setup.bash ] && source /tmp/sol/install/setup.bash; "$@" ); }
# build solution as the organisers do (colcon build in /solution copy)
rm -rf /tmp/sol; cp -r /kaggle/working/solution /tmp/sol; (cd /tmp/sol && rosenv colcon build > $W/logs/sol_build.log 2>&1; echo sol_build_rc=$? >> $W/logs/sol_build.log)
for TASK in ${TASKS:-sort2 sort3}; do
 export HACK_TASK=$TASK HACK_POOL=${HACK_POOL:-3} HACK_SEED=${SEED0:-100} HACK_HEADLESS=1 HACK_RES=1280x720 HACK_RENDER_INTERVAL=3
 export HACK_KIT_ARGS="--/rtx/raytracing/fractionalCutoutOpacity=true --/rtx/pathtracing/fractionalCutoutOpacity=true --/rtx/translucency/maxRefractionBounces=4 --/app/runLoops/main/rateLimitEnabled=false --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
 rosenv python teleoperation/bridges/ros2_zmq_bridge.py > $W/logs/bridge_$TASK.log 2>&1 & BR=$!; sleep 3
 LD_LIBRARY_PATH=$T/nvlib:$LD_LIBRARY_PATH $MAMBA_ROOT_PREFIX/envs/isaac/bin/python scripts/sim_runner.py --task NSSRSim-TiangongPro-Parlor-v0 --enable_cameras --num_envs 1 --perf_stats --step_hz 100 --headless --device cpu > $W/logs/sim_$TASK.log 2>&1 & SIM=$!
 D=$((SECONDS+900)); for i in $(seq 1 120); do [ $SECONDS -gt $D ] && break; rosenv timeout -k 2 8 ros2 topic echo --once --qos-durability transient_local /sim/task std_msgs/msg/String >/dev/null 2>&1 && break; sleep 5; done
 for k in $(seq 1 ${NLAYOUTS:-5}); do
   S=$((SEED0+k)); [ $k -gt 1 ] && { rosenv ros2 topic pub --once /sim/objects/randomize std_msgs/msg/Int32 "{data: $S}" >/dev/null 2>&1; sleep 15; }
   rosenv timeout -k 5 60 python /kaggle/working/score.py start > $W/logs/t0_${TASK}_$S.txt 2>&1
   ( cd /tmp/sol && rosenv timeout --kill-after=20 ${RUNTO:-900} bash run.sh ) > $W/logs/run_${TASK}_$S.log 2>&1
   rosenv timeout -k 5 120 python /kaggle/working/score.py check $TASK $S >> $W/results.jsonl 2>> $W/logs/score_err.log
   tail -1 $W/results.jsonl
 done
 kill -INT $SIM; sleep 10; kill $BR; pkill -f sim_node.py; pkill -f camera_bridge.py; pkill -f robot_state_publisher; pkill -9 -f sim_runner.py; sleep 5
done
