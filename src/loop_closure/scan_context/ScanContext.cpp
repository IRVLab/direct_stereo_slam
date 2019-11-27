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

#include "ScanContext.h"
#include <cmath>

inline void align_points_PCA(const std::vector<Eigen::Vector3d> &pts_clr_in,
                             std::vector<Eigen::Vector3d> &pts_clr_out,
                             Eigen::Matrix4d &tfm_pca_rig) {
  double mx(0), my(0), mz(0);
  for (auto &pc : pts_clr_in) {
    mx += pc(0);
    my += pc(1);
    mz += pc(2);
  }
  mx /= pts_clr_in.size();
  my /= pts_clr_in.size();
  mz /= pts_clr_in.size();

  // normalize pts and color
  Eigen::MatrixXd pts_mat(pts_clr_in.size(), 3);
  std::vector<float> color;
  for (size_t i = 0; i < pts_clr_in.size(); i++) {
    pts_mat(i, 0) = pts_clr_in[i](0) - mx;
    pts_mat(i, 1) = pts_clr_in[i](1) - my;
    pts_mat(i, 2) = pts_clr_in[i](2) - mz;
  }

  // PCA
  auto cov = pts_mat.transpose() * pts_mat;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
  auto v0 = es.eigenvectors().col(0);
  auto v1 = es.eigenvectors().col(1);
  auto v2 = es.eigenvectors().col(2);

  // rotate pts
  auto nx = pts_mat * v0;
  auto ny = pts_mat * v1;
  auto nz = pts_mat * v2;

  pts_clr_out.clear();
  for (size_t i = 0; i < pts_clr_in.size(); i++) {
    pts_clr_out.push_back({nx(i), ny(i), nz(i)});
  }

  // Transformation
  tfm_pca_rig.setIdentity();
  tfm_pca_rig.block<1, 3>(0, 0) = v0.transpose();
  tfm_pca_rig.block<1, 3>(1, 0) = v1.transpose();
  tfm_pca_rig.block<1, 3>(2, 0) = v2.transpose();
  Eigen::Vector3d t;
  t << mx, my, mz;
  t = -tfm_pca_rig.topLeftCorner<3, 3>() * t;
  tfm_pca_rig.block<3, 1>(0, 3) = t;
}

ScanContext::ScanContext() {
  num_s_ = 60;
  num_r_ = 20;
}

ScanContext::ScanContext(int s, int r) : num_s_(s), num_r_(r) {}

unsigned int ScanContext::getHeight() { return num_r_; }
unsigned int ScanContext::getWidth() { return num_s_; }

void ScanContext::generate(const std::vector<Eigen::Vector3d> &pts_spherical,
                           flann::Matrix<float> &ringkey, SigType &signature,
                           double lidar_range, Eigen::Matrix4d &tfm_pca_rig) {

  // align spherical points by PCA
  std::vector<Eigen::Vector3d> pts_aligned;
  align_points_PCA(pts_spherical, pts_aligned, tfm_pca_rig);

  // ringkey
  ringkey = flann::Matrix<float>(new float[num_r_], 1, num_r_);
  for (int i = 0; i < num_r_; i++) {
    ringkey[0][i] = 0.0;
  }

  // signature matrix A
  Eigen::VectorXd max_height =
      (-lidar_range - 1.0) * Eigen::VectorXd::Ones(num_s_ * num_r_);

  for (size_t i = 0; i < pts_aligned.size(); i++) // loop on pts
  {
    // projection to polar coordinate
    // After PCA, x: up; y: left; z:back
    double yp = pts_aligned[i](1);
    double zp = pts_aligned[i](2);

    double rho = std::sqrt(yp * yp + zp * zp);
    double theta = std::atan2(zp, yp);
    while (theta < 0)
      theta += 2.0 * M_PI;
    while (theta >= 2.0 * M_PI)
      theta -= 2.0 * M_PI;

    // get projection bin w.r.t. theta and rho
    int si = theta / (2.0 * M_PI) * num_s_;
    assert(si < num_s_);
    int ri = rho / lidar_range * num_r_;
    if (ri >= num_r_) // happens because PCA translated the points
      continue;

    max_height(si * num_r_ + ri) =
        std::max(max_height(si * num_r_ + ri), pts_aligned[i](0));
  }

  // calculate ringkey and signature
  Eigen::VectorXd sig_norm_si = Eigen::VectorXd::Zero(num_s_);
  for (int i = 0; i < num_s_ * num_r_; i++) {
    if (max_height(i) >= (-lidar_range)) {
      ringkey[0][i % num_r_]++;
      signature.push_back({i, max_height(i)});
      sig_norm_si(i / num_r_) += max_height(i) * max_height(i);
    }
  }

  // normalize ringkey
  for (int i = 0; i < num_r_; i++) {
    ringkey[0][i] /= num_s_;
  }

  // normalize signature
  sig_norm_si = sig_norm_si.cwiseSqrt();
  for (size_t i = 0; i < signature.size(); i++) {
    assert(sig_norm_si(signature[i].first / num_r_) > 0);
    signature[i].second /= sig_norm_si(signature[i].first / num_r_);
  }
}