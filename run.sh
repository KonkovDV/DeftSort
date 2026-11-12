#!/bin/bash
# One run per layout: waits for /sim/task, sorts, publishes /sim/task/done, exits.
cd "$(dirname "$0")" || exit 1
# The workspace overlay (colcon build in this directory) is sourced here as well, so the launch does not depend on the
# caller's shell having done it; sourcing it twice is harmless. No `exec` below: this shell must stay the parent of the
# node, its start time is the "run.sh started" instant the time limit counts from (budget.launch_age_s).
if [ -n "${BASH_VERSION:-}" ] && [ -f install/setup.bash ]; then
  # shellcheck disable=SC1091
  source install/setup.bash
fi
ros2 run team048_solution solution --ros-args -p use_sim_time:=true
