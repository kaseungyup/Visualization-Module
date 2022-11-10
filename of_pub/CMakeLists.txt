cmake_minimum_required(VERSION 3.0.2)
project(of_pub)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  of_msg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
 LIBRARIES of_pub
 CATKIN_DEPENDS rospy std_msgs message_runtime
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
     