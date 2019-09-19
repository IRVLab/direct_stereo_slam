#pragma once
#include <Eigen/Core>

Eigen::VectorXd
getDifferenceByYaw(const Eigen::Block<Eigen::VectorXd> &signature,
                   const Eigen::Block<Eigen::MatrixXd> &signature_history,
                   int yaw, bool reverse) {
  return signature; // TODO
}

int find_closest_place(const Eigen::VectorXd &signature,
                       const Eigen::MatrixXd &id_signature_history,
                       int loop_margin, int sc_width) {
  Eigen::Block<Eigen::VectorXd> ids = id_signature_history.col(0);

  // Separate signatures
  int signature_size = (id_signature_history.cols() - 1) / 2;

  Eigen::Block<Eigen::VectorXd> signature_structure =
      signature.head(signature_size);
  Eigen::Block<Eigen::MatrixXd> signature_history_structure =
      id_signature_history.block(0, 1, id_signature_history.rows(),
                                 signature_size);

  Eigen::Block<Eigen::VectorXd> signature_intensity =
      signature.tail(signature_size);
  Eigen::Block<Eigen::MatrixXd> signature_history_intensity =
      id_signature_history.block(0, 1 + signature_size,
                                 id_signature_history.rows(), signature_size);

  Eigen::MatrixXd differece_structure_matrix(2 * sc_width, ids.rows());
  for (int yaw = 0; yaw < sc_width; yaw++) {
    for (int reverse = 0; reverse < 2; reverse++) {
      // Get individual differece vectors
      Eigen::VectorXd differece_structure = getDifferenceByYaw(
          signature_structure, signature_history_structure, yaw, reverse);
      Eigen::VectorXd differece_intensity = getDifferenceByYaw(
          signature_intensity, signature_history_intensity, yaw, reverse);

      // Fush two difference vectors
      Eigen::VectorXd differece_fused =
          differece_structure - differece_structure.mean() +
          differece_intensity - differece_intensity.mean();
      differece_fused.tail(loop_margin) = 9999; // Block too close places

      differece_structure_matrix.row(2 * yaw + reverse) = differece_fused;
    }
  }

  // Find the closest place
}