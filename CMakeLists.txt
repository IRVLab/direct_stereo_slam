cmake_minimum_required(VERSION 2.8.3)
project(direct_stereo_slam)

set( CMAKE_BUILD_TYPE Release )

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.0 REQUIRED)
find_package(g2o REQUIRED )

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  pcl_ros
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-march=native -Wall")

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if (NOT DSO_PATH)
    set(DSO_PATH ~/Workspace/dso)
endif()
set(DSO_INCLUDE_DIRS "${DSO_PATH}/src" "${DSO_PATH}/thirdparty/Sophus")
set(DSO_LIBRARY_PATH "${DSO_PATH}/build/lib")
find_library(DSO_LIBRARY dso ${DSO_LIBRARY_PATH})
message("-- Found DSO library at \"${DSO_LIBRARY}\"")

find_package(Boost COMPONENTS system thread)
find_package(Pangolin 0.2 REQUIRED)

set(DSO_HELPERS_SOURCE_FILES
  src/dso_helpers/FrontEndDebugStuff.cpp
  src/dso_helpers/FrontEndMarginalize.cpp
  src/dso_helpers/FrontEndOptimize.cpp
  src/dso_helpers/FrontEndOptPoint.cpp
)

include_directories(${DSO_INCLUDE_DIRS} src)

add_executable(
  slam_node 
  src/main.cpp 
  src/FrontEnd.cpp
  ${DSO_HELPERS_SOURCE_FILES} 
  src/scale_optimization/ScaleOptimizer.cpp 
  src/loop_closure/LoopHandler.cpp 
  src/loop_closure/pangolin_viewer/PangolinLoopViewer.cpp 
  src/loop_closure/pangolin_viewer/KeyFrameDisplay.cpp 
  src/loop_closure/scan_context/ScanContext.cpp 
  src/loop_closure/pose_estimation/PoseEstimator.cpp)
  
target_link_libraries(slam_node
    ${DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_eigen g2o_stuff)
