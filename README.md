# Fast Direct Stereo Visual SLAM
Copyright (C) <2020> <Jiawei Mo, Junaed Sattar>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Related Publications
- **Direct Sparse Odometry**, J. Engel, V. Koltun, D. Cremers, In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2018
- **Extending Monocular Visual Odometry to Stereo Camera System by Scale Optimization**, J. Mo and J. Sattar, In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019
- **A Fast and Robust Place Recognition Approach for Stereo Visual Odometry Using LiDAR Descriptors**, J. Mo and J. Sattar, In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020

## Dependencies
[g2o](https://github.com/RainerKuemmerle/g2o)

When comple, remember to enable DBUILD_WITH_MARCH_NATIVE option:
```
cmake -DBUILD_WITH_MARCH_NATIVE=ON ..
```
otherwise, you will get **double free or corruption** error.

[DSO](https://github.com/JakobEngel/dso)
```
export DSO_PATH=[PATH_TO_DSO]/dso
```
or set the **DSO_PATH** in CMakeLists.txt.

## Install
```
cd catkin_ws/src
git clone https://github.com/jiawei-mo/direct_stereo_slam.git
cd ..
catkin_make
```

## Usage
- Calibrate stereo cameras, format of [cams](https://github.com/jiawei-mo/scale_optimization/blob/master/cams). Refer to [DSO](https://github.com/JakobEngel/dso) for more details of intrisic parameters.

- Create a launch file with the format of [sample.launch](https://github.com/jiawei-mo/scale_optimization/blob/master/launch/sample.launch).

```
roslaunch direct_stereo_slam [YOUR_LAUNCH_FILE]
```

- Ctrl-C to terminate and the results will be written to .ros folder.

## Output files
- id_pose.txt: final trajectory [incoming_id, x, y, z].