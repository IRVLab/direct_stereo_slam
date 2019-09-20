#pragma once
#include "IOWrapper/Output3DWrapper.h"
#include "boost/thread.hpp"
#include "util/MinimalImage.h"

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include "place_recognition/scan_context/ScanContext.h"
#include "place_recognition/utils/PosesPts.h"

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap {

class OutputWrapperLoop : public Output3DWrapper {
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

public:
  OutputWrapperLoop();
  OutputWrapperLoop(double lr, double va);
  ~OutputWrapperLoop();

  void publishKeyframes(std::vector<FrameHessian *> &frames, bool final,
                        CalibHessian *HCalib) override;
};

} // namespace IOWrap

} // namespace dso
