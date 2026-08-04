cmake_minimum_required(VERSION 3.0.2)
project(franka_control_bridge)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  franka_gripper
  geometry_msgs
  rospy
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/control_bridge.py
  scripts/gripper_grasp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

