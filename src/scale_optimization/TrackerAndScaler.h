// Copyright (C) <2020> <Jiawei Mo, Junaed Sattar>

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This file is modified from <https://github.com/JakobEngel/dso>

#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "ScaleAccumulator.h"
#include "util/NumType.h"
#include "util/settings.h"

#include <math.h>
#include <vector>

namespace dso {
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class TrackerAndScaler {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  TrackerAndScaler(int w, int h, const std::vector<double> &tfm_vec,
                   const Mat33f &K1);
  ~TrackerAndScaler();

  void setCoarseTrackingRef(std::vector<FrameHessian *> frameHessians);

  void scaleCoarseDepthL0(float scale);

  void makeK(CalibHessian *HCalib);

  void debugPlotIDepthMap(float *minID, float *maxID,
                          std::vector<IOWrap::Output3DWrapper *> &wraps);
  void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper *> &wraps);

  bool trackNewestCoarse(FrameHessian *newFrameHessian, SE3 &lastToNew_out,
                         AffLight &aff_g2l_out, int coarsestLvl,
                         Vec5 minResForAbort, Vec5 &lastResiduals,
                         IOWrap::Output3DWrapper *wrap = 0);

  float optimizeScale(FrameHessian *fh1, float &scale, int coarsestLvl);

  // act as pure ouptut
  int refFrameID;
  FrameHessian *lastRef;
  AffLight lastRef_aff_g2l;
  Vec3 lastFlowIndicators;
  double firstCoarseRMSE;

private:
  std::vector<float *> ptr_to_delete_;

  void makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians);

  /**************************Coarse Tracker***************************/
  Vec6 calcResPose(int lvl, const SE3 &refToNew, AffLight aff_g2l,
                   float cutoffTH, bool plot_img = false);
  void calcGSSSEPose(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
                     AffLight aff_g2l);

  float *idepth_[PYR_LEVELS];
  float *weight_sums_[PYR_LEVELS];
  float *weight_sums_bak_[PYR_LEVELS];

  Mat33f Ki_[PYR_LEVELS];
  float fx_[PYR_LEVELS];
  float fy_[PYR_LEVELS];
  float cx_[PYR_LEVELS];
  float cy_[PYR_LEVELS];
  int w_[PYR_LEVELS];
  int h_[PYR_LEVELS];

  // pc buffers
  float *pc_u_[PYR_LEVELS];
  float *pc_v_[PYR_LEVELS];
  float *pc_idepth_[PYR_LEVELS];
  float *pc_color_[PYR_LEVELS];
  int pc_n_[PYR_LEVELS];

  // warped buffers
  float *pose_buf_warped_idepth_;
  float *pose_buf_warped_u_;
  float *pose_buf_warped_v_;
  float *pose_buf_warped_dx_;
  float *pose_buf_warped_dy_;
  float *pose_buf_warped_residual_;
  float *pose_buf_warped_weight_;
  float *pose_buf_warped_refColor_;
  int pose_buf_warped_n_;

  FrameHessian *new_frame_;
  Accumulator9 pose_acc_;

  /**************************Scale Optimization***************************/
  void calcGSSSEScale(int lvl, float &H_out, float &b_out, float scale);
  Vec6 calcResScale(int lvl, float scale, float cutoffTH,
                    bool plot_img = false);

  // Transformation from frame0 to frame1
  SE3 tfm_f1_f0_;

  // cam1 params
  float fx1_[PYR_LEVELS];
  float fy1_[PYR_LEVELS];
  float cx1_[PYR_LEVELS];
  float cy1_[PYR_LEVELS];

  // warped buffers
  float *scale_buf_warped_rx1_;
  float *scale_buf_warped_rx2_;
  float *scale_buf_warped_rx3_;
  float *scale_buf_warped_dx_;
  float *scale_buf_warped_dy_;
  float *scale_buf_warped_residual_;
  float *scale_buf_warped_weight_;
  float *scale_buf_warped_ref_color_;
  int scale_buf_warped_n_;

  FrameHessian *fh1_;
  ScaleAccumulator scale_acc_;
};

class CoarseDistanceMap {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];

  CoarseDistanceMap(int w, int h);
  ~CoarseDistanceMap();

  void makeDistanceMap(std::vector<FrameHessian *> frameHessians,
                       FrameHessian *frame);

  void makeInlierVotes(std::vector<FrameHessian *> frameHessians);

  void makeK(CalibHessian *HCalib);

  float *fwdWarpedIDDistFinal;

  void addIntoDistFinal(int u, int v);

private:
  int w_[PYR_LEVELS];
  int h_[PYR_LEVELS];

  PointFrameResidual **coarse_projection_grid_;
  int *coarse_projection_grid_num_;
  Eigen::Vector2i *bfs_list1_;
  Eigen::Vector2i *bfs_list2_;

  void growDistBFS(int bfsNum);
};

} // namespace dso
