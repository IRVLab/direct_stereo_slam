#pragma once
#include <Eigen/Core>
#include <vector>

inline void align_points_PCA(
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts_clr_in,
    std::vector<std::pair<Eigen::Vector3d, float>> &pts_clr_out) {
  double mx(0), my(0), mz(0);
  for (auto &pc : pts_clr_in) {
    mx += pc.first(0);
    my += pc.first(1);
    mz += pc.first(2);
  }
  mx /= pts_clr_in.size();
  my /= pts_clr_in.size();
  mz /= pts_clr_in.size();

  // normalize pts and color
  Eigen::MatrixXd pts_mat(pts_clr_in.size(), 3);
  std::vector<float> color;
  for (int i = 0; i < pts_clr_in.size(); i++) {
    pts_mat(i, 0) = pts_clr_in[i].first(0) - mx;
    pts_mat(i, 1) = pts_clr_in[i].first(1) - my;
    pts_mat(i, 2) = pts_clr_in[i].first(2) - mz;
  }

  // PCA
  Eigen::MatrixXd cov = pts_mat.transpose() * pts_mat;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
  Eigen::VectorXd v0 = es.eigenvectors().col(0);
  Eigen::VectorXd v1 = es.eigenvectors().col(1);
  Eigen::VectorXd v2 = es.eigenvectors().col(2);

  // rotate pts
  Eigen::VectorXd nx = pts_mat * v0;
  Eigen::VectorXd ny = pts_mat * v1;
  Eigen::VectorXd nz = pts_mat * v2;

  pts_clr_out.clear();
  for (int i = 0; i < pts_clr_in.size(); i++) {
    pts_clr_out.push_back(
        {Eigen::Vector3d(nx(i), ny(i), nz(i)), pts_clr_in[i].second});
  }
}