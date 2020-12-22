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
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

#include <flann/flann.hpp>

typedef std::vector<std::pair<int, double>> SigType;

class ScanContext {
public:
  ScanContext();
  ScanContext(int s, int r);

  unsigned int getHeight();
  unsigned int getWidth();

  void generate(const std::vector<Eigen::Vector3d> &pts_spherical,
                flann::Matrix<float> &ringkey, SigType &signature,
                double lidar_range, Eigen::Matrix4d &tfm_pca_rig);

private:
  int num_s_;
  int num_r_;
};
