#!/bin/bash
# v9 (fix 1 of fixes/V8_ANALYSIS_RU.md): ONE task per Kaggle session. v8 lost 230 s of sim time because run.sh was
# killed by a 840 s wall cap (RTF 0.21 -> 175 s sim) and sort2 never started (50 min session, Isaac boot 22 min).
# Required env: TASK (sort3|sort2), SEEDS ("103"), RUNTO (wall cap per layout, s), DEADLINE_EPOCH (session cap).
# Optional: HACK_SELF_COLLISION=1 (sim: robot self-collisions ON), SIMBOOT_MAX (s, default 1800).
# Debug topics (/sim/objects/randomize, /sim/task/check) are used ONLY by this harness, never by the solution.
pkill -9 -f ros2_zmq_bridge; pkill -9 -f camera_bridge.py; pkill -9 -f sim_node.py; pkill -9 -f sim_runner.py; sleep 3
W=/kaggle/working; mkdir -p $W/logs $W/trace $W/frames_end; exec 2> >(tee -a $W/logs/run_sort_trace.log >&2); set -x; T=/tmp/w; export MAMBA_ROOT_PREFIX=$T/mm; MM=$T/bin/micromamba; cd /tmp/nssr_sim
export ROS_DOMAIN_ID=146 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES
export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/nssr_sim/hackathon/docker/fastdds_hack.xml
export VK_ICD_FILENAMES=$T/nvidia_icd.json VK_DRIVER_FILES=$T/nvidia_icd.json __EGL_VENDOR_LIBRARY_DIRS=$T/egl
: ${TASK:?TASK required}; : ${SEEDS:?SEEDS required}; : ${RUNTO:?RUNTO required}
DL=${DEADLINE_EPOCH:-$(( $(date +%s)+5700 ))}; left(){ echo $(( DL-$(date +%s) )); }
rosenv(){ ( eval "$($MM shell hook -s bash)"; micromamba activate ros; source $T/ws/install/setup.bash; [ -f /tmp/sol/install/setup.bash ] && source /tmp/sol/install/setup.bash; "$@" ); }
RPY=$MAMBA_ROOT_PREFIX/envs/ros/bin/python
# solution: organiser-style setup.sh (Rust wheel; never fails) with the ros python, then colcon build
rm -rf /tmp/sol; cp -r $W/solution /tmp/sol
$RPY -m pip --version >/dev/null 2>&1 || $RPY -m ensurepip >/dev/null 2>&1 || echo "no pip in the ros env" >> $W/logs/accel_setup.log   # conda python may lack pip
(cd /tmp/sol && [ -f setup.sh ] && TEAM048_PY=$RPY TEAM048_NO_BUILD=${TEAM048_NO_BUILD:-1} bash setup.sh) > $W/logs/accel_setup.log 2>&1
$RPY -c "import team048_accel as m;print('accel import OK',m.__version__)" >> $W/logs/accel_setup.log 2>&1 || echo "accel import FAILED (NumPy reference)" >> $W/logs/accel_setup.log
(cd /tmp/sol && rosenv colcon build > $W/logs/sol_build.log 2>&1; echo sol_build_rc=$? >> $W/logs/sol_build.log)
grep -q "sol_build_rc=0" $W/logs/sol_build.log || { echo "SOLUTION BUILD FAILED"; exit 3; }
export HACK_TASK=$TASK HACK_POOL=${HACK_POOL:-3} HACK_SEED=${SEED0:-100} HACK_HEADLESS=1 HACK_RES=1280x720 HACK_RENDER_INTERVAL=3
export HACK_SELF_COLLISION=${HACK_SELF_COLLISION:-0}
export HACK_KIT_ARGS="--/rtx/raytracing/fractionalCutoutOpacity=true --/rtx/pathtracing/fractionalCutoutOpacity=true --/rtx/translucency/maxRefractionBounces=4 --/app/runLoops/main/rateLimitEnabled=false --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
rosenv python teleoperation/bridges/ros2_zmq_bridge.py > $W/logs/bridge_$TASK.log 2>&1 & BR=$!; sleep 3
LD_LIBRARY_PATH=$T/nvlib:$LD_LIBRARY_PATH $MAMBA_ROOT_PREFIX/envs/isaac/bin/python scripts/sim_runner.py --task NSSRSim-TiangongPro-Parlor-v0 --enable_cameras --num_envs 1 --perf_stats --step_hz 100 --headless --device cpu > $W/logs/sim_$TASK.log 2>&1 & SIM=$!
# v8: Isaac needed 22 min to the first camera frame (cold shader cache) -> wait up to SIMBOOT_MAX for /sim/task AND /clock
D=$((SECONDS+${SIMBOOT_MAX:-1800})); up=0
while [ $SECONDS -lt $D ] && [ $(left) -gt $((RUNTO/2+300)) ]; do
  rosenv timeout -k 2 8 ros2 topic echo --once --qos-durability transient_local /sim/task std_msgs/msg/String > $W/logs/sim_task_$TASK.txt 2>&1 && { up=1; break; }; sleep 5
done
echo "sim boot: up=$up after ${SECONDS}s"; [ $up = 1 ] || { echo "SIM NOT UP"; kill -INT $SIM; exit 4; }
grep -q "self-collisions ON" $W/logs/sim_$TASK.log && echo "self-collisions ON (sim log)" || echo "self-collisions: not reported by the sim (HACK_SELF_COLLISION=$HACK_SELF_COLLISION)"
for S in $SEEDS; do
  L=$(left); [ $L -lt 400 ] && { echo "deadline: stop before $TASK $S"; break; }
  RT=$(( RUNTO < L-240 ? RUNTO : L-240 )); [ $RT -lt $RUNTO ] && echo "WARNING: wall cap reduced to $RT s by the session deadline (needed $RUNTO)"
  rosenv ros2 topic pub --once /sim/objects/randomize std_msgs/msg/Int32 "{data: $S}" >/dev/null 2>&1; sleep 15
  for w in $(seq 1 36); do rosenv timeout -k 5 30 python $W/score.py start 2>/dev/null | grep -q "\"sim_t\": [0-9]" && break; sleep 5; done
  rosenv timeout -k 5 60 python $W/score.py start > $W/logs/t0_${TASK}_$S.txt 2>&1
  GRAB_OUT=$W/frames_end/${TASK}_${S}_start rosenv timeout -k 5 60 python $W/grab_frames.py > /dev/null 2>&1
  export TEAM048_TRACE_DIR=$W/trace/${TASK}_$S
  ( cd /tmp/sol && rosenv timeout --kill-after=20 $RT bash run.sh ) > $W/logs/run_${TASK}_$S.log 2>&1; echo "run rc=$? cap=$RT" >> $W/logs/run_${TASK}_$S.log
  rosenv timeout -k 5 120 python $W/score.py check $TASK $S >> $W/results.jsonl 2>> $W/logs/score_err.log
  GRAB_OUT=$W/frames_end/${TASK}_${S}_end rosenv timeout -k 5 60 python $W/grab_frames.py > /dev/null 2>&1
  tail -1 $W/results.jsonl
done
kill -INT $SIM; sleep 10; kill $BR; pkill -f sim_node.py; pkill -f camera_bridge.py; pkill -f robot_state_publisher; pkill -9 -f sim_runner.py; sleep 5
