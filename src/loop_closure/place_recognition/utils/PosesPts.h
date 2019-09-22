#pragma once
#include <Eigen/Core>

#include <fstream>

struct IDPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix<double, 3, 4> w2c;
  int incoming_id;
  IDPose(int iid, const Eigen::Matrix<double, 3, 4> &w)
      : incoming_id(iid), w2c(w) {}

  friend std::ostream &operator<<(std::ostream &os, const IDPose &fip) {
    os << fip.incoming_id << " ";
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        os << fip.w2c(i, j) << " ";
      }
    }
    os << std::endl;
    return os;
  }
};

struct IDPtIntensity {
  int incoming_id;
  Eigen::Vector3d pt;
  float intensity;

  IDPtIntensity(int iid, const Eigen::Vector3d &p, float it)
      : incoming_id(iid), pt(p), intensity(it) {}

  friend std::ostream &operator<<(std::ostream &os, const IDPtIntensity &pci) {
    os << pci.incoming_id << " " << pci.pt(0) << " " << pci.pt(1) << " "
       << pci.pt(2) << " " << pci.intensity << std::endl;
    return os;
  }
};
