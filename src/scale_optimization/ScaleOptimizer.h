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

#include "ScaleAccumulator.h"
#include "util/Undistort.h"
#include "util/settings.h"

#include <opencv2/core/core.hpp>
#include <vector>

#define DEBUG_PRINT false
#define DEBUG_PLOT false

namespace dso {
struct FrameHessian;
struct CalibHessian;
struct PointFrameResidual;

class ScaleOptimizer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  ScaleOptimizer(Undistort *undistorter1_, const std::vector<double> &tfm_vec);
  ~ScaleOptimizer();

  float optimize(std::vector<FrameHessian *> frameHessians0,
                 const cv::Mat &img1, CalibHessian *HCalib0, float &scale,
                 int coarsestLvl);

private:
  // image dimensions, shared by both image
  int w_[PYR_LEVELS];
  int h_[PYR_LEVELS];

  // cam0 params buffers
  Mat33f K0_inv_[PYR_LEVELS]; // K0 inverse

  // cam1 params buffers
  float fx1_[PYR_LEVELS];
  float fy1_[PYR_LEVELS];
  float cx1_[PYR_LEVELS];
  float cy1_[PYR_LEVELS];

  // Transformation from frame0 to frame1
  SE3 tfm_f1_f0_;

  // pc buffers
  float *pc_u_[PYR_LEVELS];
  float *pc_v_[PYR_LEVELS];
  float *pc_idepth_[PYR_LEVELS];
  float *pc_color_[PYR_LEVELS];
  int pc_n_[PYR_LEVELS];

  // idepth_ buffers
  float *idepth_[PYR_LEVELS];
  float *weight_sums_[PYR_LEVELS];
  float *weight_sums_bak_[PYR_LEVELS];

  // warped buffers
  float *buf_warped_rx1_;
  float *buf_warped_rx2_;
  float *buf_warped_rx3_;
  float *buf_warped_dx_;
  float *buf_warped_dy_;
  float *buf_warped_residual_;
  float *buf_warped_weight_;
  float *buf_warped_ref_color_;
  int buf_warped_n_;

  std::vector<float *> ptr_to_delete_;

  void makeK0(CalibHessian *HCalib0);
  void setFrame0(std::vector<FrameHessian *> frameHessians0);
  void makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians0);
  void calcGSSSE(int lvl, float &H_out, float &b_out, float scale);
  Vec6 calcRes(int lvl, float scale, float cutoffTH);

  int frame0_id_;
  FrameHessian *frame0_;
  FrameHessian *frame1_;
  Undistort *undistorter1_; // undistort image
  ScaleAccumulator scale_acc_;
};

} // namespace dso
