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
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include <deque>
#include <map>
#include <pangolin/pangolin.h>

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap {

class KeyFrameDisplay;

class PangolinLoopViewer : public Output3DWrapper {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PangolinLoopViewer(int w, int h, bool startRunThread = true);
  virtual ~PangolinLoopViewer();

  void run();
  void close();

  void addImageToDisplay(std::string name, MinimalImageB3 *image);
  void clearAllImagesToDisplay();

  // ==================== Output3DWrapper Functionality ======================
  virtual void publishKeyframes(std::vector<FrameHessian *> &frames, bool final,
                                CalibHessian *HCalib) override;

  virtual void modifyKeyframePoseByKFID(int id, const SE3 &poseCamToWorld);

  void refreshLidarData(const std::vector<Eigen::Vector3d> &pts, size_t cur_sz);

  virtual void pushDepthImage(MinimalImageB3 *image) override;

  virtual void join() override;

private:
  void drawConstraints();

  void drawLidar();

  boost::thread run_thread_;
  bool running_;
  int w_, h_;

  // 3D model rendering
  boost::mutex model_3d_mutex_;
  std::vector<KeyFrameDisplay *> keyframes_;
  std::map<int, KeyFrameDisplay *> keyframes_by_id_;

  // lidar rendering
  boost::mutex model_lidar_mutex_;
  std::vector<Eigen::Vector3d> lidar_pts_;
  size_t lidar_cur_sz_;

  // images rendering
  boost::mutex open_images_mutex_;
  MinimalImageB3 *internal_kf_img_;
  bool kf_img_changed_;
};

} // namespace IOWrap

} // namespace dso
