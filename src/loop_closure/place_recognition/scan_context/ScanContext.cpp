#include "ScanContext.h"
#include <cmath>

ScanContext::ScanContext() {
  numS = 60;
  numR = 20;
}

ScanContext::ScanContext(int s, int r) : numS(s), numR(r) {}

unsigned int ScanContext::getHeight() { return numR; }
unsigned int ScanContext::getWidth() { return numS; }
unsigned int ScanContext::getSignatureSize() { return numS * numR; }

void ScanContext::getSignature(
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts_clr,
    Eigen::VectorXd &structure_output, Eigen::VectorXd &intensity_output,
    double max_rho) {
  if (max_rho < 0) {
    for (auto &pc : pts_clr) {
      double curRho = pc.first.norm();
      if (max_rho < curRho)
        max_rho = curRho;
    }
  }

  // signature matrix A
  Eigen::VectorXd pts_count = Eigen::VectorXd::Zero(numS * numR); // pts count
  Eigen::VectorXd height_lowest =
      Eigen::VectorXd::Zero(numS * numR); // height lowest
  Eigen::VectorXd height_highest =
      Eigen::VectorXd::Zero(numS * numR); // height highest
  Eigen::VectorXd pts_intensity =
      Eigen::VectorXd::Zero(numS * numR); // intensity

  for (int i = 0; i < pts_clr.size(); i++) // loop on pts
  {
    // projection to polar coordinate
    // After PCA, x: up; y: left; z:back
    double yp = pts_clr[i].first(1);
    double zp = pts_clr[i].first(2);

    double rho = std::sqrt(yp * yp + zp * zp);
    double theta = std::atan2(zp, yp);
    while (theta < 0)
      theta += 2.0 * M_PI;
    while (theta >= 2.0 * M_PI)
      theta -= 2.0 * M_PI;

    // get projection bin w.r.t. theta and rho
    int si = theta / (2.0 * M_PI) * numS;
    if (si == numS)
      si = 0;
    int ri = rho / max_rho * numR;
    if (ri >= numR)
      continue;

    if (pts_count(si * numR + ri) == 0) {
      pts_intensity(si * numR + ri) = pts_clr[i].second;
      height_lowest(si * numR + ri) = pts_clr[i].first(0);
      height_highest(si * numR + ri) = pts_clr[i].first(0);
    } else {
      pts_intensity(si * numR + ri) += double(pts_clr[i].second);
      height_lowest(si * numR + ri) =
          std::min(height_lowest(si * numR + ri), pts_clr[i].first(0));
      height_highest(si * numR + ri) =
          std::max(height_highest(si * numR + ri), pts_clr[i].first(0));
    }

    pts_count(si * numR + ri)++;
  }

  // average intensity
  float ave_intensity = 0;
  for (int i = 0; i < pts_clr.size(); i++) {
    ave_intensity += pts_clr[i].second;
  }
  ave_intensity = ave_intensity / pts_clr.size();

  // binarize average intensity for each bin
  for (int i = 0; i < numS * numR; i++) {
    if (pts_count(i)) {
      pts_intensity(i) = pts_intensity(i) / pts_count(i);
      pts_intensity(i) = pts_intensity(i) > ave_intensity ? 1 : 0;
    }
  }

  structure_output = height_highest - height_lowest; // height difference
  intensity_output = pts_intensity;
}