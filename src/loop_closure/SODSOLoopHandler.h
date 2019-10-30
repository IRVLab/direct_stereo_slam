#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include "loop_closure/place_recognition/scan_context/ScanContext.h"
#include "loop_closure/place_recognition/utils/PosesPts.h"
#include "loop_closure/pose_estimation/PoseEstimator.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <pcl/visualization/cloud_viewer.h>

#include <string>

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

class SODSOLoopHandler {
private:
  std::vector<IDPose *> poses_history;
  std::vector<IDPtIntensity *> pts_history;
  std::vector<IDPtIntensity *> pts_nearby;
  int previous_incoming_id;
  int pts_idx;

  float lidarRange;
  float voxelAngle;

  ScanContext *sc_ptr;
  Eigen::VectorXi ids;
  Eigen::MatrixXd ring_keys;
  Eigen::MatrixXd signatures_structure;
  Eigen::MatrixXd signatures_intensity;
  int signature_count;

  Eigen::MatrixXd Ts_pca_rig;

  std::vector<std::vector<std::pair<Eigen::Vector3d, float>>>
      pts_spherical_history;

  PoseEstimator *pose_estimator;
  std::vector<std::pair<AffLight, float>> affLightExposures;

  pcl::visualization::CloudViewer *pcl_viewer;

  g2o::SparseOptimizer optimizer;

public:
  SODSOLoopHandler();
  SODSOLoopHandler(float lr, float va);
  ~SODSOLoopHandler();

  void addKeyFrame(FrameHessian *fh, CalibHessian *HCalib, Mat66 poseHessian);
};

} // namespace dso