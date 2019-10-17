#pragma once
#include <Eigen/Core>

inline Eigen::VectorXd getDifferenceByYaw(Eigen::VectorXd &signature,
                                          const Eigen::MatrixXd &signatures,
                                          int sc_height, int sc_width, int yaw,
                                          bool reverse) {
  // Convert back to Scan Contect 2D image
  Eigen::Map<Eigen::MatrixXd> signature_img(signature.data(), sc_height,
                                            sc_width);

  // Permute columns by yaw and reverse
  Eigen::MatrixXd signature_img_yaw(sc_height, sc_width);
  signature_img_yaw.block(0, 0, sc_height, sc_width - yaw) =
      signature_img.block(0, yaw, sc_height, sc_width - yaw);
  signature_img_yaw.block(0, sc_width - yaw, sc_height, yaw) =
      signature_img.block(0, 0, sc_height, yaw);
  if (reverse) {
    signature_img_yaw = signature_img_yaw.colwise().reverse();
  }

  // Compute differece by Euclidean distance
  Eigen::Map<Eigen::VectorXd> signature_yaw(signature_img_yaw.data(),
                                            sc_height * sc_width);
  return Eigen::VectorXd::Ones(signatures.rows()) - signatures * signature_yaw;
}

inline void find_closest_place(Eigen::VectorXd &signature_structure,
                               Eigen::VectorXd &signature_intensity,
                               const Eigen::MatrixXd &signatures_structure,
                               const Eigen::MatrixXd &signatures_intensity,
                               int loop_margin, int sc_height, int sc_width,
                               int &idx, double &yaw, bool &reverse,
                               double &difference) {
  Eigen::MatrixXd differece_matrix_yaw(2 * sc_width,
                                       signatures_structure.rows());
  for (int yaw = 0; yaw < sc_width; yaw++) {
    for (int rev = 0; rev < 2; rev++) {
      // Get individual differece vectors
      Eigen::VectorXd differece_structure =
          getDifferenceByYaw(signature_structure, signatures_structure,
                             sc_height, sc_width, yaw, rev);
      Eigen::VectorXd differece_intensity =
          getDifferenceByYaw(signature_intensity, signatures_intensity,
                             sc_height, sc_width, yaw, rev);

      // Fush two difference vectors
      Eigen::VectorXd differece_ones =
          Eigen::VectorXd::Ones(signatures_structure.rows());
      Eigen::VectorXd differece_structure_zero_mean =
          differece_structure - differece_structure.mean() * differece_ones;
      Eigen::VectorXd differece_intensity_zero_mean =
          differece_intensity - differece_intensity.mean() * differece_ones;
      double std_structure =
          std::sqrt(differece_structure_zero_mean.squaredNorm() /
                    (differece_structure.size() - 1));
      double std_intensity =
          std::sqrt(differece_intensity_zero_mean.squaredNorm() /
                    (differece_intensity.size() - 1));
      Eigen::VectorXd differece_fused =
          (differece_structure_zero_mean + differece_intensity_zero_mean) /
          (std_structure + std_intensity);
      differece_fused.tail(loop_margin) =
          9999.9 * Eigen::VectorXd::Ones(loop_margin); // Block nearby places

      differece_matrix_yaw.row(2 * yaw + rev) = differece_fused.transpose();
    }
  }

  // Find the closest place
  int yaw_reverse;
  difference = differece_matrix_yaw.minCoeff(&yaw_reverse, &idx);
  reverse = yaw_reverse - (yaw_reverse / 2) * 2;
  yaw = double(yaw_reverse / 2) / sc_width * (2 * M_PI);
}