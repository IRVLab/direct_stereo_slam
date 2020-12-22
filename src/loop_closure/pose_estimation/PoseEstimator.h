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

#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"
#include "util/settings.h"
#include "vector"
#include <math.h>

#define RES_THRES 10.0
#define INNER_PERCENT 90
// #define TRANS_THRES 3.0
// #define ROT_THRES 0.2

namespace dso {
struct FrameHessian;
struct PointFrameResidual;

class PoseEstimator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  PoseEstimator(int w, int h);
  ~PoseEstimator();

  bool estimate(const std::vector<std::pair<Eigen::Vector3d, float *>> &pts,
                float ref_ab_exposure, FrameHessian *new_fh,
                const std::vector<float> &new_cam, int coarsest_lvl,
                Eigen::Matrix4d &ref_to_new, float &pose_error);

private:
  void makeK(const std::vector<float> &cam);
  Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH,
               bool plot_img = false);
  void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
                 AffLight aff_g2l);

  float fx_[PYR_LEVELS];
  float fy_[PYR_LEVELS];
  float cx_[PYR_LEVELS];
  float cy_[PYR_LEVELS];
  int w_[PYR_LEVELS];
  int h_[PYR_LEVELS];

  // pc buffers
  std::vector<std::pair<Eigen::Vector3d, float *>> pts_;

  // warped buffers
  float *buf_warped_idepth_;
  float *buf_warped_u_;
  float *buf_warped_v_;
  float *buf_warped_dx_;
  float *buf_warped_dy_;
  float *buf_warped_residual_;
  float *buf_warped_weight_;
  float *buf_warped_ref_color_;
  int buf_warped_n_;

  Accumulator9 acc_;

  std::vector<float *> ptr_to_delete_;

  // photometric terms
  AffLight ref_aff_g2l_;
  float ref_ab_exposure_;
  FrameHessian *new_frame_;
};

} // namespace dso