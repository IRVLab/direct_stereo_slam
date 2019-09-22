#pragma once

#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"
#include "util/settings.h"
#include "vector"
#include <math.h>

namespace dso {
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class PoseEstimator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  PoseEstimator(int w, int h);
  ~PoseEstimator();

  bool estimate(const std::vector<std::pair<Eigen::Vector3d, float>> &pts,
                const std::pair<AffLight, float> &affLightExposure,
                FrameHessian *newFrameHessian, CalibHessian *HCalib,
                Eigen::Matrix<double, 4, 4> &lastToNew_out, int coarsestLvl = 3,
                Vec5 minResForAbort = Vec5::Constant(NAN));

private:
  void makeK(CalibHessian *HCalib);

  void setPointsRef(const std::vector<std::pair<Eigen::Vector3d, float>> &pts);

  Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
                    AffLight aff_g2l, float cutoffTH);
  Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
  void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
                 AffLight aff_g2l);
  void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
              AffLight aff_g2l);

  bool debugPrint, debugPlot;

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];
  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float fxi[PYR_LEVELS];
  float fyi[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];
  float cxi[PYR_LEVELS];
  float cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

  AffLight lastRef_aff_g2l;
  float lastRef_ab_exposure;
  FrameHessian *newFrame;

  // act as pure ouptut
  Vec5 lastResiduals;
  Vec3 lastFlowIndicators;

  // pc buffers
  Eigen::MatrixXf pointxyzi;

  // warped buffers
  float *buf_warped_idepth;
  float *buf_warped_u;
  float *buf_warped_v;
  float *buf_warped_dx;
  float *buf_warped_dy;
  float *buf_warped_residual;
  float *buf_warped_weight;
  float *buf_warped_refColor;
  int buf_warped_n;

  std::vector<float *> ptrToDelete;

  Accumulator9 acc;
};

} // namespace dso