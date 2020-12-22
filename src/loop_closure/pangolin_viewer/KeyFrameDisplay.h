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

#undef Success
#include "util/NumType.h"
#include <Eigen/Core>
#include <pangolin/pangolin.h>

#include <fstream>
#include <sstream>

namespace dso {
class CalibHessian;
class FrameHessian;
class FrameShell;

namespace IOWrap {

template <int ppp> struct InputPointSparse {
  float u;
  float v;
  float idpeth;
  float idepth_hessian;
  float relObsBaseline;
  int numGoodRes;
  unsigned char color[ppp];
  unsigned char status;
};

struct MyVertex {
  float point[3];
  unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  KeyFrameDisplay();
  ~KeyFrameDisplay();

  // copies points from KF over to internal buffer,
  // keeping some additional information so we can render it differently.
  void setFromKF(FrameHessian *fh, CalibHessian *HCalib);

  // copies points from KF over to internal buffer,
  // keeping some additional information so we can render it differently.
  void setFromF(FrameShell *fs, CalibHessian *HCalib);

  // copies & filters internal data to GL buffer for rendering. if nothing to
  // do: does nothing.
  void refreshPC();

  // renders cam & pointcloud.
  void drawPC(float pointSize);

  int id_;
  bool active_;
  SE3 tfm_w_c_;
  bool need_refresh_;

  inline bool operator<(const KeyFrameDisplay &other) const {
    return (id_ < other.id_);
  }

private:
  float fx_, fy_, cx_, cy_;
  float fxi_, fyi_, cxi_, cyi_;
  int width_, height_;

  float my_scaled_th_, my_abs_th_, my_scale_;
  int my_sparsify_factor_;
  int my_displayMode_;
  float my_min_rel_bs_;

  int num_sparse_points_;
  int num_sparse_buffer_size_;
  InputPointSparse<MAX_RES_PER_POINT> *original_input_sparse_;

  bool buffer_valid_;
  int num_gl_buffer_points_;
  int num_gl_buffer_good_points_;
  pangolin::GlBuffer vertex_buffer_;
  pangolin::GlBuffer color_buffer_;
};

} // namespace IOWrap
} // namespace dso
