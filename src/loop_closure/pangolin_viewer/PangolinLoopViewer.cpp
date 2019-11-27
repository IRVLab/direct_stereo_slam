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
}

PangolinLoopViewer::~PangolinLoopViewer() {
  close();
  run_thread_.join();
}

void PangolinLoopViewer::run() {
  pangolin::CreateWindowAndBind("Main", 2 * w_, 2 * h_);
  const int UI_WIDTH = 180;

  glEnable(GL_DEPTH_TEST);

  // 3D visualization
  pangolin::OpenGlRenderState Visualization3D_camera(
      pangolin::ProjectionMatrix(w_, h_, 400, 400, w_ / 2, h_ / 2, 0.1, 1000),
      pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY));

  pangolin::View &Visualization3D_display =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0,
                     -w_ / (float)h_)
          .SetHandler(new pangolin::Handler3D(Visualization3D_camera));

  pangolin::View &d_kfDepth =
      pangolin::Display("imgKFDepth").SetAspect(w_ / (float)h_);
  pangolin::GlTexture texKFDepth(w_, h_, GL_RGB, false, 0, GL_RGB,
                                 GL_UNSIGNED_BYTE);
  pangolin::CreateDisplay()
      .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(d_kfDepth);

  // parameter reconfigure gui
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                        pangolin::Attach::Pix(UI_WIDTH));

  pangolin::Var<int> settings_point_cloud_mode_("ui.PC_mode", 1, 1, 4, false);

  pangolin::Var<int> settings_sparsity_("ui.sparsity", 1, 1, 20, false);
  pangolin::Var<double> settings_scaled_var_th_("ui.relVarTH", 0.001, 1e-10,
                                                1e10, true);
  pangolin::Var<double> settings_abs_var_th_("ui.absVarTH", 0.001, 1e-10, 1e10,
                                             true);
  pangolin::Var<double> settings_min_rel_bs_("ui.minRelativeBS", 0.1, 0, 1,
                                             false);

  pangolin::Var<int> settings_nPts(
      "ui.activePoints", setting_desiredPointDensity, 50, 5000, false);
  pangolin::Var<int> settings_nCandidates(
      "ui.pointCandidates", setting_desiredImmatureDensity, 50, 5000, false);
  pangolin::Var<int> settings_nMaxFrames("ui.maxFrames", setting_maxFrames, 4,
                                         10, false);
  pangolin::Var<double> settings_kfFrequency(
      "ui.kfFrequency", setting_kfGlobalWeight, 0.1, 3, false);
  pangolin::Var<double> settings_gradHistAdd(
      "ui.minGradAdd", setting_minGradHistAdd, 0, 15, false);

  // Default hooks for exiting (Esc) and fullscreen (tab).
  while (!pangolin::ShouldQuit() && running_) {
    // Clear entire screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // Activate efficiently by object
    Visualization3D_display.Activate(Visualization3D_camera);
    boost::unique_lock<boost::mutex> lk3d(model_3d_mutex_);
    // pangolin::glDrawColouredCube();
    int refreshed = 0;
    for (KeyFrameDisplay *fh : keyframes_) {
      refreshed += (int)(fh->refreshPC(
          refreshed < 10, this->settings_scaled_var_th_,
          this->settings_abs_var_th_, this->settings_point_cloud_mode_,
          this->settings_min_rel_bs_, this->settings_sparsity_));
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

    // update parameters
    this->settings_point_cloud_mode_ = settings_point_cloud_mode_.Get();

    this->settings_abs_var_th_ = settings_abs_var_th_.Get();
    this->settings_scaled_var_th_ = settings_scaled_var_th_.Get();
    this->settings_min_rel_bs_ = settings_min_rel_bs_.Get();
    this->settings_sparsity_ = settings_sparsity_.Get();

    setting_desiredPointDensity = settings_nPts.Get();
    setting_desiredImmatureDensity = settings_nCandidates.Get();
    setting_maxFrames = settings_nMaxFrames.Get();
    setting_kfGlobalWeight = settings_kfFrequency.Get();
    setting_minGradHistAdd = settings_gradHistAdd.Get();

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  // exit(1);
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
    glVertex3f((float)keyframes_[i]->tfm_c_w_.translation()[0],
               (float)keyframes_[i]->tfm_c_w_.translation()[1],
               (float)keyframes_[i]->tfm_c_w_.translation()[2]);
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
  keyframes_by_id_[id]->tfm_c_w_ = poseCamToWorld;
  keyframes_by_id_[id]->need_refresh_ = true;
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
