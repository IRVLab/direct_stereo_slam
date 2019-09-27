#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include "loop_closure/place_recognition/scan_context/ScanContext.h"
#include "loop_closure/place_recognition/utils/PosesPts.h"
#include "loop_closure/pose_estimation/PoseEstimator.h"

#define COMPARE_PCL true

#if COMPARE_PCL
#include "loop_closure/place_recognition/utils/merge_point_clouds.h"
#include <pcl/visualization/cloud_viewer.h>
#endif

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

  double lidarRange;
  double voxelAngle;

  ScanContext *sc_ptr;
  Eigen::VectorXi ids;
  Eigen::MatrixXd signatures_structure;
  Eigen::MatrixXd signatures_intensity;
  int signature_count;

  Eigen::MatrixXd Ts_pca_rig;

  std::vector<std::vector<std::pair<Eigen::Vector3d, float>>>
      pts_spherical_history;

  PoseEstimator *pose_estimator;
  std::vector<std::pair<AffLight, float>> affLightExposures;

#if COMPARE_PCL
  pcl::visualization::CloudViewer *pcl_viewer;
#endif

public:
  SODSOLoopHandler();
  SODSOLoopHandler(double lr, double va);
  ~SODSOLoopHandler();

  void addKeyFrame(FrameHessian *fh, CalibHessian *HCalib, Mat66 poseHessian);
};

} // namespace dso
