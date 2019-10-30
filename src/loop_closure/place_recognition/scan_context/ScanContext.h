#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

class ScanContext {
public:
  ScanContext();
  ScanContext(int s, int r);

  unsigned int getHeight();
  unsigned int getWidth();
  unsigned int getSignatureSize();

  void
  getSignature(const std::vector<std::pair<Eigen::Vector3d, float>> &pts_clr,
               Eigen::VectorXd &ringkey, Eigen::VectorXd &structure_output,
               Eigen::VectorXd &intensity_output, double max_rho = -1.0);

private:
  int numS;
  int numR;
};
