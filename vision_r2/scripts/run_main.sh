#!/bin/bash
# Wrapper to run main.py with warnings filtered
export ROS_LOG_LEVEL=ERROR
export RCUTILS_LOGGING_USE_STDOUT=false
cd "$(dirname "$(dirname "$(dirname "$0")")")"
source install/setup.bash
ros2 run vision_r2 main.py 2>&1 | grep -v "sequence size\|QFont\|XDG_SESSION\|qt.qpa"
