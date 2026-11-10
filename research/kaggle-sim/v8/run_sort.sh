#!/bin/bash
# v7: pool-3 sorting with the NEW solution (colcon build), per task: start sim once, per layout randomize(seed) -> run.sh -> score.
# Global hard deadline DEADLINE_EPOCH (set by the kernel driver); per-run wall caps RUNTO_<task>. Debug topics only in this harness.
pkill -9 -f ros2_zmq_bridge; pkill -9 -f camera_bridge.py; pkill -9 -f sim_node.py; pkill -9 -f sim_runner.py; sleep 3
W=/kaggle/working; mkdir -p $W/logs $W/trace $W/frames_end; exec 2> >(tee -a $W/logs/run_sort_trace.log >&2); set -x; T=/tmp/w; export MAMBA_ROOT_PREFIX=$T/mm; MM=$T/bin/micromamba; cd /tmp/nssr_sim
export ROS_DOMAIN_ID=146 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES
export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/nssr_sim/hackathon/docker/fastdds_hack.xml
export VK_ICD_FILENAMES=$T/nvidia_icd.json VK_DRIVER_FILES=$T/nvidia_icd.json __EGL_VENDOR_LIBRARY_DIRS=$T/egl
DL=${DEADLINE_EPOCH:-$(( $(date +%s)+4800 ))}; left(){ echo $(( DL-$(date +%s) )); }
rosenv(){ ( eval "$($MM shell hook -s bash)"; micromamba activate ros; source $T/ws/install/setup.bash; [ -f /tmp/sol/install/setup.bash ] && source /tmp/sol/install/setup.bash; "$@" ); }
# build solution as the organisers do (colcon build in a copy of the repo)
rm -rf /tmp/sol; cp -r $W/solution /tmp/sol; (cd /tmp/sol && rosenv colcon build > $W/logs/sol_build.log 2>&1; echo sol_build_rc=$? >> $W/logs/sol_build.log)
grep -q "sol_build_rc=0" $W/logs/sol_build.log || { echo "SOLUTION BUILD FAILED"; exit 3; }
for TASK in ${TASKS:-sort3 sort2}; do
 [ $(left) -lt 600 ] && { echo "deadline: skip $TASK"; break; }
 NL=$(eval echo \${NLAYOUTS_$TASK:-${NLAYOUTS:-3}}); RT0=$(eval echo \${RUNTO_$TASK:-${RUNTO:-900}})
 export HACK_TASK=$TASK HACK_POOL=${HACK_POOL:-3} HACK_SEED=${SEED0:-100} HACK_HEADLESS=1 HACK_RES=1280x720 HACK_RENDER_INTERVAL=3
 export HACK_KIT_ARGS="--/rtx/raytracing/fractionalCutoutOpacity=true --/rtx/pathtracing/fractionalCutoutOpacity=true --/rtx/translucency/maxRefractionBounces=4 --/app/runLoops/main/rateLimitEnabled=false --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
 rosenv python teleoperation/bridges/ros2_zmq_bridge.py > $W/logs/bridge_$TASK.log 2>&1 & BR=$!; sleep 3
 LD_LIBRARY_PATH=$T/nvlib:$LD_LIBRARY_PATH $MAMBA_ROOT_PREFIX/envs/isaac/bin/python scripts/sim_runner.py --task NSSRSim-TiangongPro-Parlor-v0 --enable_cameras --num_envs 1 --perf_stats --step_hz 100 --headless --device cpu > $W/logs/sim_$TASK.log 2>&1 & SIM=$!
 D=$((SECONDS+900)); for i in $(seq 1 120); do [ $SECONDS -gt $D ] && break; [ $(left) -lt 300 ] && break; rosenv timeout -k 2 8 ros2 topic echo --once --qos-durability transient_local /sim/task std_msgs/msg/String > $W/logs/sim_task_$TASK.txt 2>&1 && break; sleep 5; done
 SL=$(eval echo \${SEEDS_$TASK:-}); [ -z "$SL" ] && SL=$(for k in $(seq 1 $NL); do echo $((SEED0+k)); done)
 for S in $SL; do
   L=$(left); [ $L -lt 300 ] && { echo "deadline: stop before $TASK $S"; break; }
   RT=$(( RT0 < L-180 ? RT0 : L-180 ))
   rosenv ros2 topic pub --once /sim/objects/randomize std_msgs/msg/Int32 "{data: $S}" >/dev/null 2>&1; sleep 15
   for w in $(seq 1 24); do rosenv timeout -k 5 30 python $W/score.py start 2>/dev/null | grep -q "\"sim_t\": [0-9]" && break; sleep 5; done   # v7 s101: run.sh started before /clock was up
   rosenv timeout -k 5 60 python $W/score.py start > $W/logs/t0_${TASK}_$S.txt 2>&1
   GRAB_OUT=$W/frames_end/${TASK}_${S}_start rosenv timeout -k 5 60 python $W/grab_frames.py > /dev/null 2>&1
   export TEAM048_TRACE_DIR=$W/trace/${TASK}_$S
   ( cd /tmp/sol && rosenv timeout --kill-after=20 $RT bash run.sh ) > $W/logs/run_${TASK}_$S.log 2>&1; echo "run rc=$? cap=$RT" >> $W/logs/run_${TASK}_$S.log
   rosenv timeout -k 5 120 python $W/score.py check $TASK $S >> $W/results.jsonl 2>> $W/logs/score_err.log
   GRAB_OUT=$W/frames_end/${TASK}_${S}_end rosenv timeout -k 5 60 python $W/grab_frames.py > /dev/null 2>&1
   tail -1 $W/results.jsonl
 done
 kill -INT $SIM; sleep 10; kill $BR; pkill -f sim_node.py; pkill -f camera_bridge.py; pkill -f robot_state_publisher; pkill -9 -f sim_runner.py; sleep 5
done
