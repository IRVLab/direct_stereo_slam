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

#include "FrontEnd.h"

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/ImageRW.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/ImmaturePoint.h"

namespace dso {

void FrontEnd::debugPlotTracking() {
  if (disableAllDisplay)
    return;
  if (!setting_render_plotTrackingFull)
    return;
  int wh = hG[0] * wG[0];

  int idx = 0;
  for (FrameHessian *f : frame_hessians_) {
    std::vector<MinimalImageB3 *> images;

    // make images for all frames. will be deleted by the FrameHessian's
    // destructor.
    for (FrameHessian *f2 : frame_hessians_)
      if (f2->debugImage == 0)
        f2->debugImage = new MinimalImageB3(wG[0], hG[0]);

    for (FrameHessian *f2 : frame_hessians_) {
      MinimalImageB3 *debugImage = f2->debugImage;
      images.push_back(debugImage);

      Eigen::Vector3f *fd = f2->dI;

      Vec2 affL = AffLight::fromToVecExposure(f2->ab_exposure, f->ab_exposure,
                                              f2->aff_g2l(), f->aff_g2l());

      for (int i = 0; i < wh; i++) {
        // BRIGHTNESS TRANSFER
        float colL = affL[0] * fd[i][0] + affL[1];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        debugImage->at(i) = Vec3b(colL, colL, colL);
      }
    }

    for (PointHessian *ph : f->pointHessians) {
      assert(ph->status == PointHessian::ACTIVE);
      if (ph->status == PointHessian::ACTIVE ||
          ph->status == PointHessian::MARGINALIZED) {
        for (PointFrameResidual *r : ph->residuals)
          r->debugPlot();
        f->debugImage->setPixel9(ph->u + 0.5, ph->v + 0.5,
                                 makeRainbow3B(ph->idepth_scaled));
      }
    }

    char buf[100];
    snprintf(buf, 100, "IMG %d", idx);
    IOWrap::displayImageStitch(buf, images);
    idx++;
  }

  IOWrap::waitKey(0);
}

void FrontEnd::debugPlot(std::string name) {
  if (disableAllDisplay)
    return;
  if (!setting_render_renderWindowFrames)
    return;
  std::vector<MinimalImageB3 *> images;

  float minID = 0, maxID = 0;
  if ((int)(freeDebugParam5 + 0.5f) == 7 || (debugSaveImages && false)) {
    std::vector<float> allID;
    for (unsigned int f = 0; f < frame_hessians_.size(); f++) {
      for (PointHessian *ph : frame_hessians_[f]->pointHessians)
        if (ph != 0)
          allID.push_back(ph->idepth_scaled);

      for (PointHessian *ph : frame_hessians_[f]->pointHessiansMarginalized)
        if (ph != 0)
          allID.push_back(ph->idepth_scaled);

      for (PointHessian *ph : frame_hessians_[f]->pointHessiansOut)
        if (ph != 0)
          allID.push_back(ph->idepth_scaled);
    }
    std::sort(allID.begin(), allID.end());
    int n = allID.size() - 1;
    minID = allID[(int)(n * 0.05)];
    maxID = allID[(int)(n * 0.95)];

    // slowly adapt: change by maximum 10% of old span.
    float maxChange = 0.1 * (max_id_jet_vis_debug_ - min_id_jet_vis_debug_);
    if (max_id_jet_vis_debug_ < 0 || min_id_jet_vis_debug_ < 0)
      maxChange = 1e5;

    if (minID < min_id_jet_vis_debug_ - maxChange)
      minID = min_id_jet_vis_debug_ - maxChange;
    if (minID > min_id_jet_vis_debug_ + maxChange)
      minID = min_id_jet_vis_debug_ + maxChange;

    if (maxID < max_id_jet_vis_debug_ - maxChange)
      maxID = max_id_jet_vis_debug_ - maxChange;
    if (maxID > max_id_jet_vis_debug_ + maxChange)
      maxID = max_id_jet_vis_debug_ + maxChange;

    max_id_jet_vis_debug_ = maxID;
    min_id_jet_vis_debug_ = minID;
  }

  int wh = hG[0] * wG[0];
  for (unsigned int f = 0; f < frame_hessians_.size(); f++) {
    MinimalImageB3 *img = new MinimalImageB3(wG[0], hG[0]);
    images.push_back(img);
    // float* fd = frame_hessians_[f]->I;
    Eigen::Vector3f *fd = frame_hessians_[f]->dI;

    for (int i = 0; i < wh; i++) {
      int c = fd[i][0] * 0.9f;
      if (c > 255)
        c = 255;
      img->at(i) = Vec3b(c, c, c);
    }

    if ((int)(freeDebugParam5 + 0.5f) == 0) {
      for (PointHessian *ph : frame_hessians_[f]->pointHessians) {
        if (ph == 0)
          continue;

        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f,
                          makeRainbow3B(ph->idepth_scaled));
      }
      for (PointHessian *ph : frame_hessians_[f]->pointHessiansMarginalized) {
        if (ph == 0)
          continue;
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f,
                          makeRainbow3B(ph->idepth_scaled));
      }
      for (PointHessian *ph : frame_hessians_[f]->pointHessiansOut)
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 255));
    } else if ((int)(freeDebugParam5 + 0.5f) == 1) {
      for (PointHessian *ph : frame_hessians_[f]->pointHessians) {
        if (ph == 0)
          continue;
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f,
                          makeRainbow3B(ph->idepth_scaled));
      }

      for (PointHessian *ph : frame_hessians_[f]->pointHessiansMarginalized)
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));

      for (PointHessian *ph : frame_hessians_[f]->pointHessiansOut)
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 255));
    } else if ((int)(freeDebugParam5 + 0.5f) == 2) {

    } else if ((int)(freeDebugParam5 + 0.5f) == 3) {
      for (ImmaturePoint *ph : frame_hessians_[f]->immaturePoints) {
        if (ph == 0)
          continue;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD ||
            ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ||
            ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) {
          if (!std::isfinite(ph->idepth_max))
            img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
          else {
            img->setPixelCirc(
                ph->u + 0.5f, ph->v + 0.5f,
                makeRainbow3B((ph->idepth_min + ph->idepth_max) * 0.5f));
          }
        }
      }
    } else if ((int)(freeDebugParam5 + 0.5f) == 4) {
      for (ImmaturePoint *ph : frame_hessians_[f]->immaturePoints) {
        if (ph == 0)
          continue;

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 255, 0));
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 0));
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 255));
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 0));
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 255));
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
      }
    } else if ((int)(freeDebugParam5 + 0.5f) == 5) {
      for (ImmaturePoint *ph : frame_hessians_[f]->immaturePoints) {
        if (ph == 0)
          continue;

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
          continue;
        float d = freeDebugParam1 * (sqrtf(ph->quality) - 1);
        if (d < 0)
          d = 0;
        if (d > 1)
          d = 1;
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f,
                          Vec3b(0, d * 255, (1 - d) * 255));
      }

    } else if ((int)(freeDebugParam5 + 0.5f) == 6) {
      for (PointHessian *ph : frame_hessians_[f]->pointHessians) {
        if (ph == 0)
          continue;
        if (ph->my_type == 0)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 255));
        if (ph->my_type == 1)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 0));
        if (ph->my_type == 2)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 255));
        if (ph->my_type == 3)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 255, 255));
      }
      for (PointHessian *ph : frame_hessians_[f]->pointHessiansMarginalized) {
        if (ph == 0)
          continue;
        if (ph->my_type == 0)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 255));
        if (ph->my_type == 1)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 0));
        if (ph->my_type == 2)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 255));
        if (ph->my_type == 3)
          img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 255, 255));
      }
    }
    if ((int)(freeDebugParam5 + 0.5f) == 7) {
      for (PointHessian *ph : frame_hessians_[f]->pointHessians) {
        img->setPixelCirc(
            ph->u + 0.5f, ph->v + 0.5f,
            makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID))));
      }
      for (PointHessian *ph : frame_hessians_[f]->pointHessiansMarginalized) {
        if (ph == 0)
          continue;
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
      }
    }
  }
  IOWrap::displayImageStitch(name.c_str(), images);
  IOWrap::waitKey(5);

  for (unsigned int i = 0; i < images.size(); i++)
    delete images[i];

  if ((debugSaveImages && false)) {
    for (unsigned int f = 0; f < frame_hessians_.size(); f++) {
      MinimalImageB3 *img = new MinimalImageB3(wG[0], hG[0]);
      Eigen::Vector3f *fd = frame_hessians_[f]->dI;

      for (int i = 0; i < wh; i++) {
        int c = fd[i][0] * 0.9f;
        if (c > 255)
          c = 255;
        img->at(i) = Vec3b(c, c, c);
      }

      for (PointHessian *ph : frame_hessians_[f]->pointHessians) {
        img->setPixelCirc(
            ph->u + 0.5f, ph->v + 0.5f,
            makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID))));
      }
      for (PointHessian *ph : frame_hessians_[f]->pointHessiansMarginalized) {
        if (ph == 0)
          continue;
        img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
      }

      char buf[1000];
      snprintf(buf, 1000, "images_out/kf_%05d_%05d_%02d.png",
               frame_hessians_.back()->shell->id,
               frame_hessians_.back()->frameID, f);
      IOWrap::writeImage(buf, img);

      delete img;
    }
  }
}

} // namespace dso
