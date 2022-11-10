cmake_minimum_required(VERSION 3.0.2)
project(pos_pub)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
 LIBRARIES pos_pub
 CATKIN_DEPENDS rospy std_msgs
 DEPENDS system_lib
)

# Build

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)

# Install

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)