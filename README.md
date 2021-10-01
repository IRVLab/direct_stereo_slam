# Fast Direct Stereo Visual SLAM
## Related Publications
- **Direct Sparse Odometry**, J. Engel, V. Koltun, D. Cremers, In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2018
- **Extending Monocular Visual Odometry to Stereo Camera System by Scale Optimization**, J. Mo and J. Sattar, In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019
- **A Fast and Robust Place Recognition Approach for Stereo Visual Odometry Using LiDAR Descriptors**, J. Mo and J. Sattar, In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020

## Dependencies
[ROS](https://www.ros.org/), [PCL](https://pointclouds.org/), [g2o](https://github.com/RainerKuemmerle/g2o), and [DSO](https://github.com/JakobEngel/dso) ([OpenCV](https://opencv.org/), [Pangolin](https://github.com/stevenlovegrove/Pangolin))

**For g2o, DSO, and Pangolin, we strongly recommend the (old) versions in the provided [dependencies.zip](https://github.com/IRVLab/direct_stereo_slam/blob/master/dependencies.zip) for smooth install and reasonable results.**
1. Install Pangolin
```
cd Pangolin
mkdir build && cd build
cmake ..
make -j4
sudo make install
```
2. Install DSO
```
sudo apt install libsuitesparse-dev libeigen3-dev libboost-all-dev libopencv-dev
cd DSO
mkdir build && cd build
cmake ..
make -j4
```
3. Install g2o
```
cd g2o
mkdir build && cd build
cmake -DBUILD_WITH_MARCH_NATIVE=ON .. # use the flag to avoid possible double-free error
make -j4
sudo make install
```

## Install
1. Link to the external DSO library:
```
export DSO_PATH=[PATH_TO_DSO] (e.g., export DSO_PATH=~/Workspace/dso)
```
or set the **DSO_PATH** in CMakeLists.txt.

2. Install direct_stereo_slam using ros/catkin 
```
cd catkin_ws/src
git clone https://github.com/IRVLab/direct_stereo_slam.git
cd ..
catkin_make
```

## Usage
- Calibrate stereo cameras with format of [cams](https://github.com/IRVLab/direct_stereo_slam/blob/master/cams). T_stereo is the pose of camera0 in camera1 coordinate, rememeber to put a small number in T_stereo[2,2] for numerical stability if images are stereo pre-calibrated. Refer to [DSO](https://github.com/JakobEngel/dso) for more details of intrisic parameters.

- Create a launch file with the format of [sample.launch](https://github.com/IRVLab/direct_stereo_slam/blob/master/launch/sample.launch).

```
roslaunch direct_stereo_slam [YOUR_LAUNCH_FILE]
```

- Ctrl-C to terminate the program, the final trajectory (dslam.txt) will be written to .ros folder.

## Output file
- dslam.txt: final trajectory [incoming_id, x, y, z];
- sodso.txt: the trajectory without loop closure, output for comparision.
