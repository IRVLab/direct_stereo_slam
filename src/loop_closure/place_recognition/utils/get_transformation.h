#pragma once
#include <Eigen/Core>

inline Eigen::Matrix<double, 4, 4>
get_transformation(const Eigen::Matrix<double, 4, 4> &T_pca_rig,
                   const Eigen::MatrixXd &Ts_pca_rig, int idx, int yaw_reverse,
                   int sc_width) {
  // T_pca_matched
  Eigen::Matrix<double, 4, 4> T_pca_matched =
      Ts_pca_rig.block<4, 4>(4 * idx, 0);

  // T_reverse_pca
  Eigen::Matrix<double, 4, 4> T_reverse_pca =
      Eigen::Matrix<double, 4, 4>::Identity();
  if (yaw_reverse - (yaw_reverse / 2) * 2) { // reversed
    // After PCA, x: up; y: left; z:back
    T_reverse_pca(0, 0) = -1;
    T_reverse_pca(2, 2) = -1;
  }

  // T_yaw_reverse
  double yaw = double(yaw_reverse / 2) / sc_width * (2 * M_PI);
  Eigen::Matrix3d R_yaw_reverse;
  R_yaw_reverse = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitX());
  Eigen::Matrix<double, 4, 4> T_yaw_reverse =
      Eigen::Matrix<double, 4, 4>::Identity();
  T_yaw_reverse.block<3, 3>(0, 0) = R_yaw_reverse;

  // T_query_matched
  Eigen::Matrix<double, 4, 4> T_query_matched =
      (T_yaw_reverse * T_reverse_pca * T_pca_rig).inverse() * T_pca_matched;

  return T_query_matched;
}