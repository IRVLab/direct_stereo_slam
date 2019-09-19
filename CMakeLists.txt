cmake_minimum_required(VERSION 2.4.6)
project(direct_stereo_slam)

set(BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.0 REQUIRED)
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

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

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

set(SODSOSystem_SOURCE_FILES
  src/SODSOSystem.cpp
  src/dso_helpers/SODSOSystemDebugStuff.cpp
  src/dso_helpers/SODSOSystemMarginalize.cpp
  src/dso_helpers/SODSOSystemOptimize.cpp
  src/dso_helpers/SODSOSystemOptPoint.cpp
)

include_directories(${DSO_INCLUDE_DIRS} src)
  add_executable(
  so_dso_node src/so_dso_node.cpp 
  ${SODSOSystem_SOURCE_FILES} 
  src/output_wrappers/OutputWrapperLoop.cpp 
  src/scale_optimization/ScaleOptimizer.cpp 
  src/place_recognition/scan_context/ScanContext.cpp)
  
target_link_libraries(so_dso_node
    ${DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES})
