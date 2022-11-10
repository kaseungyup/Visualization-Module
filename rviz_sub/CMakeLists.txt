cmake_minimum_required(VERSION 3.0.2)
project(rviz_sub)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  std_msgs
  tf
  visualization_msgs
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rviz_sub
  CATKIN_DEPENDS geometry_msgs rospy std_msgs tf visualization_msgs
  DEPENDS system_lib
)

# Build

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)

# add_executable(rviz_sub src/rviz_sub.py)
# add_dependencies(rviz_sub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(rviz_sub ${catkin_LIBRARIES})

# Install

# install(TARGETS rviz_sub
#         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY rviz
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)