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

#include "PangolinLoopViewer.h"
#include "KeyFrameDisplay.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/globalCalib.h"
#include "util/settings.h"

namespace dso {
namespace IOWrap {

PangolinLoopViewer::PangolinLoopViewer(int w, int h, bool startRunThread) {
  w_ = w;
  h_ = h;
  running_ = true;

  boost::unique_lock<boost::mutex> lk(open_images_mutex_);
  internal_kf_img_ = new MinimalImageB3(w_, h_);
  internal_kf_img_->setBlack();

  setting_render_renderWindowFrames = false;
  setting_render_plotTrackingFull = false;
  setting_render_displayCoarseTrackingFull = false;

  if (startRunThread)
    run_thread_ = boost::thread(&PangolinLoopViewer::run, this);

  lidar_cur_sz_ = 0;
}

PangolinLoopViewer::~PangolinLoopViewer() {
  close();
  run_thread_.join();
}

void PangolinLoopViewer::run() {
  pangolin::CreateWindowAndBind("DirectSLAM", 960, 1080);
  const float ratio = w_ / float(h_);

  auto proj_mat =
      pangolin::ProjectionMatrix(w_, h_, 200, 200, w_ / 2, h_ / 2, 0.1, 1000);
  auto model_view =
      pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY);

  glEnable(GL_DEPTH_TEST);

  // 3D visualization
  pangolin::OpenGlRenderState Visualization3D_camera(proj_mat, model_view);

  pangolin::View &Visualization3D_display =
      pangolin::CreateDisplay()
          .SetBounds(0.3, 1.0, 0.0, 1.0, -ratio)
          .SetHandler(new pangolin::Handler3D(Visualization3D_camera));

  // keyframe depth visualization
  pangolin::GlTexture texKFDepth(w_, h_, GL_RGB, false, 0, GL_RGB,
                                 GL_UNSIGNED_BYTE);
  pangolin::View &d_kfDepth = pangolin::Display("imgKFDepth").SetAspect(ratio);

  // lidar visualization
  pangolin::OpenGlRenderState Visualization_lidar_camera(proj_mat, model_view);
  pangolin::View &Visualization_lidar_display =
      pangolin::Display("lidarDisplay")
          .SetAspect(ratio)
          .SetHandler(new pangolin::Handler3D(Visualization_lidar_camera));

  pangolin::CreateDisplay()
      .SetBounds(0.0, 0.3, 0.0, 1.0)
      .SetLayout(pangolin::LayoutEqualHorizontal)
      .AddDisplay(d_kfDepth)
      .AddDisplay(Visualization_lidar_display);

  // Default hooks for exiting (Esc) and fullscreen (tab).
  while (!pangolin::ShouldQuit() && running_) {
    // Clear entire screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // Activate efficiently by object
    Visualization3D_display.Activate(Visualization3D_camera);
    boost::unique_lock<boost::mutex> lk3d(model_3d_mutex_);
    // pangolin::glDrawColouredCube();
    for (KeyFrameDisplay *fh : keyframes_) {
      fh->refreshPC();
      fh->drawPC(1);
    }
    drawConstraints();
    lk3d.unlock();

    open_images_mutex_.lock();
    if (kf_img_changed_)
      texKFDepth.Upload(internal_kf_img_->data, GL_BGR, GL_UNSIGNED_BYTE);
    kf_img_changed_ = false;
    open_images_mutex_.unlock();

    d_kfDepth.Activate();
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    texKFDepth.RenderToViewportFlipY();

    Visualization_lidar_display.Activate(Visualization_lidar_camera);
    boost::unique_lock<boost::mutex> lklidar(model_lidar_mutex_);
    drawLidar();
    lklidar.unlock();

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  exit(1);
}

void PangolinLoopViewer::close() { running_ = false; }

void PangolinLoopViewer::join() {
  run_thread_.join();
  printf("JOINED Pangolin thread!\n");
}

void PangolinLoopViewer::drawConstraints() {
  float colorRed[3] = {1, 0, 0};
  glColor3f(colorRed[0], colorRed[1], colorRed[2]);
  glLineWidth(3);

  glBegin(GL_LINE_STRIP);
  for (unsigned int i = 0; i < keyframes_.size(); i++) {
    glVertex3f((float)keyframes_[i]->tfm_w_c_.translation()[0],
               (float)keyframes_[i]->tfm_w_c_.translation()[1],
               (float)keyframes_[i]->tfm_w_c_.translation()[2]);
  }
  glEnd();
}

void PangolinLoopViewer::publishKeyframes(std::vector<FrameHessian *> &frames,
                                          bool final, CalibHessian *HCalib) {
  // only work on marginalized frame
  if (!final)
    return;

  assert(frames.size() == 1); // contains only one marginalized frame
  FrameHessian *fh = frames[0];

  static int prv_id = -1;

  // keep incoming id increasing
  if (prv_id >= fh->frameID) {
    return;
  }
  prv_id = fh->frameID;

  boost::unique_lock<boost::mutex> lk(model_3d_mutex_);
  if (keyframes_by_id_.find(fh->frameID) == keyframes_by_id_.end()) {
    KeyFrameDisplay *kfd = new KeyFrameDisplay();
    keyframes_by_id_[fh->frameID] = kfd;
    keyframes_.push_back(kfd);
  }
  keyframes_by_id_[fh->frameID]->setFromKF(fh, HCalib);
}

void PangolinLoopViewer::modifyKeyframePoseByKFID(int id,
                                                  const SE3 &poseCamToWorld) {
  boost::unique_lock<boost::mutex> lk3d(model_3d_mutex_);
  keyframes_by_id_[id]->tfm_w_c_ = poseCamToWorld;
  keyframes_by_id_[id]->need_refresh_ = true;
}

void PangolinLoopViewer::refreshLidarData(
    const std::vector<Eigen::Vector3d> &pts, size_t cur_sz) {
  boost::unique_lock<boost::mutex> lk(model_lidar_mutex_);
  assert(cur_sz <= pts.size());
  lidar_pts_ = pts;
  lidar_cur_sz_ = cur_sz;
}

void PangolinLoopViewer::drawLidar() {
  glPointSize(3.0);

  glBegin(GL_POINTS);
  for (size_t i = 0; i < lidar_pts_.size(); i++) {
    if (i < lidar_cur_sz_) {
      glColor3ub(0, 255, 0);
    } else {
      glColor3ub(255, 0, 0);
    }
    glVertex3f(lidar_pts_[i](0), lidar_pts_[i](1), lidar_pts_[i](2));
  }
  glEnd();
}

void PangolinLoopViewer::pushDepthImage(MinimalImageB3 *image) {

  if (!setting_render_displayDepth)
    return;
  if (disableAllDisplay)
    return;

  boost::unique_lock<boost::mutex> lk(open_images_mutex_);
  memcpy(internal_kf_img_->data, image->data, w_ * h_ * 3);
  kf_img_changed_ = true;
}

} // namespace IOWrap
} // namespace dso
