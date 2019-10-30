#pragma once
#include <Eigen/Core>

#include <vector>

struct pairComparator {
  bool operator()(const std::pair<double, int> &a,
                  const std::pair<double, int> &b) {
    return a.first > b.first;
  }
};

inline void find_closest_place_ring_key(Eigen::VectorXd &ring_key,
                                        const Eigen::MatrixXd &ring_keys,
                                        int loop_margin, double diff_thres,
                                        std::vector<int> &indexes) {
  Eigen::VectorXd score = ring_keys * ring_key;
  score.tail(loop_margin) =
      -Eigen::VectorXd::Ones(loop_margin); // Block nearby places

  indexes.clear();
  double min_score = 1.0 - 2.0 * diff_thres;
  for (int i = 0; i < score.rows(); i++) {
    if (score(i) > min_score) {
      indexes.emplace_back(i);
    }
  }
}

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
  return (Eigen::VectorXd::Ones(signatures.rows()) -
          signatures * signature_yaw) /
         2.0;
}

inline void find_closest_place_sc(Eigen::VectorXd &signature_structure,
                                  Eigen::VectorXd &signature_intensity,
                                  const Eigen::MatrixXd &signatures_structure,
                                  const Eigen::MatrixXd &signatures_intensity,
                                  int loop_margin, int sc_height, int sc_width,
                                  const std::vector<int> &indexes, int &idx,
                                  double &yaw, bool &reverse,
                                  double &difference) {
  Eigen::MatrixXd signatures_structure_top(indexes.size(),
                                           signatures_structure.cols());
  Eigen::MatrixXd signatures_intensity_top(indexes.size(),
                                           signatures_intensity.cols());
  for (int i = 0; i < indexes.size(); i++) {
    signatures_structure_top.row(i) = signatures_structure.row(indexes[i]);
    signatures_intensity_top.row(i) = signatures_intensity.row(indexes[i]);
  }

  Eigen::MatrixXd differece_matrix_yaw(2 * sc_width,
                                       signatures_structure_top.rows());
  for (int yaw = 0; yaw < sc_width; yaw++) {
    for (int rev = 0; rev < 2; rev++) {
      // Get individual differece vectors
      Eigen::VectorXd differece_structure =
          getDifferenceByYaw(signature_structure, signatures_structure_top,
                             sc_height, sc_width, yaw, rev);
      Eigen::VectorXd differece_intensity =
          getDifferenceByYaw(signature_intensity, signatures_intensity_top,
                             sc_height, sc_width, yaw, rev);

      // Fush two difference vectors
      //   Eigen::VectorXd differece_ones =
      //       Eigen::VectorXd::Ones(signatures_structure_top.rows());
      //   Eigen::VectorXd differece_structure_zero_mean =
      //       differece_structure - differece_structure.mean() *
      //       differece_ones;
      //   Eigen::VectorXd differece_intensity_zero_mean =
      //       differece_intensity - differece_intensity.mean() *
      //       differece_ones;
      //   double std_structure =
      //       std::sqrt(differece_structure_zero_mean.squaredNorm() /
      //                 (differece_structure.size() - 1));
      //   double std_intensity =
      //       std::sqrt(differece_intensity_zero_mean.squaredNorm() /
      //                 (differece_intensity.size() - 1));
      //   Eigen::VectorXd differece_fused =
      //       (differece_structure_zero_mean + differece_intensity_zero_mean) /
      //       (std_structure + std_intensity);

      //   differece_matrix_yaw.row(2 * yaw + rev) =
      //   differece_fused.transpose();

      differece_matrix_yaw.row(2 * yaw + rev) =
          (differece_structure.transpose() + differece_intensity.transpose()) /
          2.0;
    }
  }

  // Find the closest place
  int yaw_reverse;
  difference = differece_matrix_yaw.minCoeff(&yaw_reverse, &idx);
  idx = indexes[idx];
  reverse = yaw_reverse - (yaw_reverse / 2) * 2;
  yaw = double(yaw_reverse / 2) / sc_width * (2 * M_PI);
}