#!/bin/bash
# One run per layout: waits for /sim/task, sorts, publishes /sim/task/done, exits.
cd "$(dirname "$0")"
ros2 run team048_solution solution --ros-args -p use_sim_time:=true
