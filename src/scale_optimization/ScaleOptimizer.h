#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "ScaleAccumulator.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "util/NumType.h"
#include "util/settings.h"
#include "vector"
#include <math.h>

namespace dso {
struct FrameHessian;
struct PointFrameResidual;

class ScaleOptimizer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  ScaleOptimizer(int w_, int h_, const Eigen::Matrix3f &K, const SE3 &T_stereo_,
                 float accept_th_);
  ~ScaleOptimizer();

  bool optimize(FrameHessian *newFrameHessian, float &scale, int coarsestLvl,
                Vec5 minResForAbort, IOWrap::Output3DWrapper *wrap = 0);

  void makeK(CalibHessian *HCalib);
  void setCoarseTrackingRef(std::vector<FrameHessian *> frameHessians);

  bool debugPrint, debugPlot;
  FrameHessian *lastRef;
  FrameHessian *newFrame;
  int refFrameID;

  // act as pure ouptut
  Vec5 lastResiduals;
  Vec3 lastFlowIndicators;
  double firstCoarseRMSE;

private:
  // th for accepting final scale
  float accept_th;

  // pc buffers
  float *pc_u[PYR_LEVELS];
  float *pc_v[PYR_LEVELS];
  float *pc_idepth[PYR_LEVELS];
  float *pc_color[PYR_LEVELS];
  int pc_n[PYR_LEVELS];

  // cam params buffers
  int w[PYR_LEVELS]; // shared by both image
  int h[PYR_LEVELS];
  Mat33f Ki_orig[PYR_LEVELS];
  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];
  SE3 T_stereo;

  // idepth buffers
  float *idepth[PYR_LEVELS];
  float *weightSums[PYR_LEVELS];
  float *weightSums_bak[PYR_LEVELS];

  // warped buffers
  float *buf_warped_rx1;
  float *buf_warped_rx2;
  float *buf_warped_rx3;
  float *buf_warped_dx;
  float *buf_warped_dy;
  float *buf_warped_residual;
  float *buf_warped_weight;
  float *buf_warped_refColor;
  int buf_warped_n;

  // helpers
  std::vector<float *> ptrToDelete;
  ScaleAccumulator acc;

  void makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians);
  void calcGSSSE(int lvl, float &H_out, float &b_out, const SE3 &refToNew,
                 float scale);
  Vec6 calcRes(int lvl, const SE3 &T_stereo, float scale, float cutoffTH);
};

} // namespace dso
