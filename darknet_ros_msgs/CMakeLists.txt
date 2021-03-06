cmake_minimum_required(VERSION 2.8.12)

project(darknet_ros_msgs)

set(CMAKE_CXX_FLAGS "-lpthread -std=c++11 -fPIC -lX11 -lz -lxcb -lrt ${CMAKE_CXX_FLAGS}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(catkin REQUIRED
  COMPONENTS
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    message_generation
)

add_message_files(
  FILES
    BoundingBox.msg
    BoundingBoxes.msg
)

add_action_files(
  FILES
    CheckForObjects.action
)

generate_messages(
  DEPENDENCIES
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    message_runtime
    std_msgs
)
