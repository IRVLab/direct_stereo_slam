// Copyright (C) <2020> <Jiawei Mo, Junaed Sattar>

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <Eigen/Core>

#define RES_X 1.0
#define RES_Y 0.5
#define RES_Z 1.0

inline void generate_spherical_points(
    std::vector<std::pair<int, Eigen::Vector3d>> &pts_nearby,
    std::unordered_map<int, Eigen::Matrix<double, 6, 1>> &id_pose_wc,
    const dso::SE3 &cur_cw, double lidar_range,
    std::vector<Eigen::Vector3d> &pts_spherical) {
  // if the oriention difference between a historical keyframe and current
  // keyframe is large, trim all associated points
  for (auto &ip : id_pose_wc) {
    auto pose_diff_se3 = (cur_cw * dso::SE3::exp(ip.second)).log();
    auto rotation_norm = pose_diff_se3.tail(3).norm();
    if (rotation_norm > 0.5) {
      id_pose_wc.erase(ip.first);
    }
  }

  // get/filter spherical points
  std::vector<double> steps{1.0 / RES_X, 1.0 / RES_Y, 1.0 / RES_Z};
  std::vector<int> voxel_size{
      static_cast<int>(floor(2 * lidar_range * steps[0]) + 1),
      static_cast<int>(floor(2 * lidar_range * steps[1]) + 1),
      static_cast<int>(floor(2 * lidar_range * steps[2]) + 1)};
  std::vector<int> loc_step{1, voxel_size[0], voxel_size[0] * voxel_size[1]};
  std::unordered_map<int, std::pair<int, Eigen::Vector3d>> loc2idx_pt;
  for (size_t i = 0; i < pts_nearby.size(); i++) {
    if (id_pose_wc.find(pts_nearby[i].first) == id_pose_wc.end()) {
      continue; // orientation changed too much
    }

    Eigen::Vector4d p_g(pts_nearby[i].second(0), pts_nearby[i].second(1),
                        pts_nearby[i].second(2), 1.0);
    Eigen::Vector3d p_l = cur_cw.matrix3x4() * p_g;

    if (p_l.norm() >= lidar_range) {
      continue; // out of range
    }

    // voxel indices
    int xi = static_cast<int>(floor((p_l(0) + lidar_range) * steps[0]));
    int yi = static_cast<int>(floor((p_l(1) + lidar_range) * steps[1]));
    int zi = static_cast<int>(floor((p_l(2) + lidar_range) * steps[2]));
    int loc = xi * loc_step[0] + yi * loc_step[1] + zi * loc_step[2];

    // store the highest points
    if (loc2idx_pt.find(loc) == loc2idx_pt.end() ||
        -loc2idx_pt[loc].second(1) < -p_l(1)) {
      loc2idx_pt[loc] = {i, p_l};
    }
  }

  // output useful points
  std::vector<std::pair<int, Eigen::Vector3d>> new_pts_nearby;
  for (auto &l_ip : loc2idx_pt) {
    pts_spherical.push_back(l_ip.second.second);
    new_pts_nearby.push_back(pts_nearby[l_ip.second.first]);
  }

  // update nearby pts
  pts_nearby = new_pts_nearby;
}
