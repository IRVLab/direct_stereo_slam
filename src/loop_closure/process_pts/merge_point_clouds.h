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

inline void merge_points(const std::vector<Eigen::Vector3d> &pts_target,
                         const std::vector<Eigen::Vector3d> &pts_source,
                         const Eigen::Matrix4d &tfm_target_source) {
  Eigen::Matrix<double, 4, 1> pt_source, pt_target;
  pt_source(3) = 1.0;
  for (size_t i = 0; i < pts_source.size(); i++) {
    pt_source.head(3) = pts_source[i];
    pt_target = tfm_target_source * pt_source;
    pts_target.push_back({pt_target(0), pt_target(2), -pt_target(1)});
  }
}