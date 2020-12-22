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

#include "TrackerAndScaler.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "IOWrapper/ImageRW.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

#define DEBUG_PRINT false
#define DEBUG_PLOT false

namespace dso {

template <int b, typename T>
T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
  const int padT = 1 + ((1 << b) / sizeof(T));
  T *ptr = new T[size + padT];
  rawPtrVec.push_back(ptr);
  T *alignedPtr = (T *)((((uintptr_t)(ptr + padT)) >> b) << b);
  return alignedPtr;
}

TrackerAndScaler::TrackerAndScaler(int ww, int hh,
                                   const std::vector<double> &tfm_vec,
                                   const Mat33f &K1)
    : lastRef_aff_g2l(0, 0) {
  // make coarse tracking templates.
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = ww >> lvl;
    int hl = hh >> lvl;

    idepth_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    weight_sums_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    weight_sums_bak_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);

    pc_u_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    pc_v_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    pc_idepth_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    pc_color_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
  }

  // pose warped buffers
  pose_buf_warped_idepth_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_u_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_v_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_dx_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_dy_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_residual_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_weight_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  pose_buf_warped_refColor_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);

  new_frame_ = 0;
  lastRef = 0;
  w_[0] = h_[0] = 0;
  refFrameID = -1;

  // tranformation form frame0 to frame1
  Eigen::Matrix4d tfm_eigen;
  cv::Mat tfm_stereo_cv = cv::Mat(tfm_vec);
  tfm_stereo_cv = tfm_stereo_cv.reshape(0, 4);
  cv::cv2eigen(tfm_stereo_cv, tfm_eigen);
  tfm_f1_f0_ = SE3(tfm_eigen);

  // make camera1 parameters
  fx1_[0] = K1(0, 0);
  fy1_[0] = K1(1, 1);
  cx1_[0] = K1(0, 2);
  cy1_[0] = K1(1, 2);
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    fx1_[level] = fx1_[level - 1] * 0.5;
    fy1_[level] = fy1_[level - 1] * 0.5;
    cx1_[level] = (cx1_[0] + 0.5) / ((int)1 << level) - 0.5;
    cy1_[level] = (cy1_[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  // scale warped buffers
  scale_buf_warped_rx1_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_rx2_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_rx3_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_dx_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_dy_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_residual_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_weight_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  scale_buf_warped_ref_color_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
}

TrackerAndScaler::~TrackerAndScaler() {
  for (float *ptr : ptr_to_delete_)
    delete[] ptr;
  ptr_to_delete_.clear();
}

void TrackerAndScaler::makeK(CalibHessian *HCalib) {
  w_[0] = wG[0];
  h_[0] = hG[0];

  fx_[0] = HCalib->fxl();
  fy_[0] = HCalib->fyl();
  cx_[0] = HCalib->cxl();
  cy_[0] = HCalib->cyl();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w_[level] = w_[0] >> level;
    h_[level] = h_[0] >> level;
    fx_[level] = fx_[level - 1] * 0.5;
    fy_[level] = fy_[level - 1] * 0.5;
    cx_[level] = (cx_[0] + 0.5) / ((int)1 << level) - 0.5;
    cy_[level] = (cy_[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < pyrLevelsUsed; ++level) {
    Mat33f K;
    K << fx_[level], 0.0, cx_[level], 0.0, fy_[level], cy_[level], 0.0, 0.0,
        1.0;
    Ki_[level] = K.inverse();
  }
}

void TrackerAndScaler::makeCoarseDepthL0(
    std::vector<FrameHessian *> frameHessians) {
  // make coarse tracking templates for latstRef.
  memset(idepth_[0], 0, sizeof(float) * w_[0] * h_[0]);
  memset(weight_sums_[0], 0, sizeof(float) * w_[0] * h_[0]);

  for (FrameHessian *fh : frameHessians) {
    for (PointHessian *ph : fh->pointHessians) {
      if (ph->lastResiduals[0].first != 0 &&
          ph->lastResiduals[0].second == ResState::IN) {
        PointFrameResidual *r = ph->lastResiduals[0].first;
        assert(r->efResidual->isActive() && r->target == lastRef);
        int u = r->centerProjectedTo[0] + 0.5f;
        int v = r->centerProjectedTo[1] + 0.5f;
        float new_idepth = r->centerProjectedTo[2];
        float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

        idepth_[0][u + w_[0] * v] += new_idepth * weight;
        weight_sums_[0][u + w_[0] * v] += weight;
      }
    }
  }

  for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
    int lvlm1 = lvl - 1;
    int wl = w_[lvl], hl = h_[lvl], wlm1 = w_[lvlm1];

    float *idepth_l = idepth_[lvl];
    float *weight_sums_l = weight_sums_[lvl];

    float *idepth_lm = idepth_[lvlm1];
    float *weight_sums_lm = weight_sums_[lvlm1];

    for (int y = 0; y < hl; y++)
      for (int x = 0; x < wl; x++) {
        int bidx = 2 * x + 2 * y * wlm1;
        idepth_l[x + y * wl] = idepth_lm[bidx] + idepth_lm[bidx + 1] +
                               idepth_lm[bidx + wlm1] +
                               idepth_lm[bidx + wlm1 + 1];

        weight_sums_l[x + y * wl] =
            weight_sums_lm[bidx] + weight_sums_lm[bidx + 1] +
            weight_sums_lm[bidx + wlm1] + weight_sums_lm[bidx + wlm1 + 1];
      }
  }

  // dilate idepth_ by 1.
  for (int lvl = 0; lvl < 2; lvl++) {
    int numIts = 1;

    for (int it = 0; it < numIts; it++) {
      int wh = w_[lvl] * h_[lvl] - w_[lvl];
      int wl = w_[lvl];
      float *weightSumsl = weight_sums_[lvl];
      float *weightSumsl_bak = weight_sums_bak_[lvl];
      memcpy(weightSumsl_bak, weightSumsl, w_[lvl] * h_[lvl] * sizeof(float));
      float *idepthl =
          idepth_[lvl]; // dotnt need to make a temp copy of depth, since I only
                        // read values with weightSumsl>0, and write ones with
                        // weightSumsl<=0.
      for (int i = w_[lvl]; i < wh; i++) {
        if (weightSumsl_bak[i] <= 0) {
          float sum = 0, num = 0, numn = 0;
          if (weightSumsl_bak[i + 1 + wl] > 0) {
            sum += idepthl[i + 1 + wl];
            num += weightSumsl_bak[i + 1 + wl];
            numn++;
          }
          if (weightSumsl_bak[i - 1 - wl] > 0) {
            sum += idepthl[i - 1 - wl];
            num += weightSumsl_bak[i - 1 - wl];
            numn++;
          }
          if (weightSumsl_bak[i + wl - 1] > 0) {
            sum += idepthl[i + wl - 1];
            num += weightSumsl_bak[i + wl - 1];
            numn++;
          }
          if (weightSumsl_bak[i - wl + 1] > 0) {
            sum += idepthl[i - wl + 1];
            num += weightSumsl_bak[i - wl + 1];
            numn++;
          }
          if (numn > 0) {
            idepthl[i] = sum / numn;
            weightSumsl[i] = num / numn;
          }
        }
      }
    }
  }

  // dilate idepth_ by 1 (2 on lower levels).
  for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
    int wh = w_[lvl] * h_[lvl] - w_[lvl];
    int wl = w_[lvl];
    float *weightSumsl = weight_sums_[lvl];
    float *weightSumsl_bak = weight_sums_bak_[lvl];
    memcpy(weightSumsl_bak, weightSumsl, w_[lvl] * h_[lvl] * sizeof(float));
    float *idepthl =
        idepth_[lvl]; // dotnt need to make a temp copy of depth, since I only
                      // read values with weightSumsl>0, and write ones with
                      // weightSumsl<=0.
    for (int i = w_[lvl]; i < wh; i++) {
      if (weightSumsl_bak[i] <= 0) {
        float sum = 0, num = 0, numn = 0;
        if (weightSumsl_bak[i + 1] > 0) {
          sum += idepthl[i + 1];
          num += weightSumsl_bak[i + 1];
          numn++;
        }
        if (weightSumsl_bak[i - 1] > 0) {
          sum += idepthl[i - 1];
          num += weightSumsl_bak[i - 1];
          numn++;
        }
        if (weightSumsl_bak[i + wl] > 0) {
          sum += idepthl[i + wl];
          num += weightSumsl_bak[i + wl];
          numn++;
        }
        if (weightSumsl_bak[i - wl] > 0) {
          sum += idepthl[i - wl];
          num += weightSumsl_bak[i - wl];
          numn++;
        }
        if (numn > 0) {
          idepthl[i] = sum / numn;
          weightSumsl[i] = num / numn;
        }
      }
    }
  }

  // normalize idepths and weights.
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    float *weightSumsl = weight_sums_[lvl];
    float *idepthl = idepth_[lvl];
    Eigen::Vector3f *dIRefl = lastRef->dIp[lvl];

    int wl = w_[lvl], hl = h_[lvl];

    int lpc_n = 0;
    float *lpc_u = pc_u_[lvl];
    float *lpc_v = pc_v_[lvl];
    float *lpc_idepth = pc_idepth_[lvl];
    float *lpc_color = pc_color_[lvl];

    for (int y = 2; y < hl - 2; y++)
      for (int x = 2; x < wl - 2; x++) {
        int i = x + y * wl;

        if (weightSumsl[i] > 0) {
          idepthl[i] /= weightSumsl[i];
          lpc_u[lpc_n] = x;
          lpc_v[lpc_n] = y;
          lpc_idepth[lpc_n] = idepthl[i];
          lpc_color[lpc_n] = dIRefl[i][0];

          if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0)) {
            idepthl[i] = -1;
            continue; // just skip if something is wrong.
          }
          lpc_n++;
        } else
          idepthl[i] = -1;

        weightSumsl[i] = 1;
      }

    pc_n_[lvl] = lpc_n;
  }
}

void TrackerAndScaler::setCoarseTrackingRef(
    std::vector<FrameHessian *> frameHessians) {
  assert(frameHessians.size() > 0);
  lastRef = frameHessians.back();
  makeCoarseDepthL0(frameHessians);

  refFrameID = lastRef->shell->id;
  lastRef_aff_g2l = lastRef->aff_g2l();

  firstCoarseRMSE = -1;
}

void TrackerAndScaler::scaleCoarseDepthL0(float scale) {
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    float *lpc_idepth = pc_idepth_[lvl];
    for (int p = 0; p < pc_n_[lvl]; p++) {
      lpc_idepth[p] /= scale;
    }
  }
}

void TrackerAndScaler::debugPlotIDepthMap(
    float *minID_pt, float *maxID_pt,
    std::vector<IOWrap::Output3DWrapper *> &wraps) {
  if (w_[1] == 0)
    return;

  int lvl = 0;

  {
    std::vector<float> allID;
    for (int i = 0; i < h_[lvl] * w_[lvl]; i++) {
      if (idepth_[lvl][i] > 0)
        allID.push_back(idepth_[lvl][i]);
    }
    std::sort(allID.begin(), allID.end());
    int n = allID.size() - 1;

    float minID_new = allID[(int)(n * 0.05)];
    float maxID_new = allID[(int)(n * 0.95)];

    float minID, maxID;
    minID = minID_new;
    maxID = maxID_new;
    if (minID_pt != 0 && maxID_pt != 0) {
      if (*minID_pt < 0 || *maxID_pt < 0) {
        *maxID_pt = maxID;
        *minID_pt = minID;
      } else {

        // slowly adapt: change by maximum 10% of old span.
        float maxChange = 0.3 * (*maxID_pt - *minID_pt);

        if (minID < *minID_pt - maxChange)
          minID = *minID_pt - maxChange;
        if (minID > *minID_pt + maxChange)
          minID = *minID_pt + maxChange;

        if (maxID < *maxID_pt - maxChange)
          maxID = *maxID_pt - maxChange;
        if (maxID > *maxID_pt + maxChange)
          maxID = *maxID_pt + maxChange;

        *maxID_pt = maxID;
        *minID_pt = minID;
      }
    }

    MinimalImageB3 mf(w_[lvl], h_[lvl]);
    mf.setBlack();
    for (int i = 0; i < h_[lvl] * w_[lvl]; i++) {
      int c = lastRef->dIp[lvl][i][0] * 0.9f;
      if (c > 255)
        c = 255;
      mf.at(i) = Vec3b(c, c, c);
    }
    int wl = w_[lvl];
    for (int y = 3; y < h_[lvl] - 3; y++)
      for (int x = 3; x < wl - 3; x++) {
        int idx = x + y * wl;
        float sid = 0, nid = 0;
        float *bp = idepth_[lvl] + idx;

        if (bp[0] > 0) {
          sid += bp[0];
          nid++;
        }
        if (bp[1] > 0) {
          sid += bp[1];
          nid++;
        }
        if (bp[-1] > 0) {
          sid += bp[-1];
          nid++;
        }
        if (bp[wl] > 0) {
          sid += bp[wl];
          nid++;
        }
        if (bp[-wl] > 0) {
          sid += bp[-wl];
          nid++;
        }

        if (bp[0] > 0 || nid >= 3) {
          float id = ((sid / nid) - minID) / ((maxID - minID));
          mf.setPixelCirc(x, y, makeJet3B(id));
          // mf.at(idx) = makeJet3B(id);
        }
      }
    // IOWrap::displayImage("coarseDepth LVL0", &mf, false);

    for (IOWrap::Output3DWrapper *ow : wraps)
      ow->pushDepthImage(&mf);

    if (debugSaveImages) {
      char buf[1000];
      snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png",
               lastRef->shell->id, refFrameID);
      IOWrap::writeImage(buf, &mf);
    }
  }
}

void TrackerAndScaler::debugPlotIDepthMapFloat(
    std::vector<IOWrap::Output3DWrapper *> &wraps) {
  if (w_[1] == 0)
    return;
  int lvl = 0;
  MinimalImageF mim(w_[lvl], h_[lvl], idepth_[lvl]);
  for (IOWrap::Output3DWrapper *ow : wraps)
    ow->pushDepthImageFloat(&mim, lastRef);
}

bool TrackerAndScaler::trackNewestCoarse(FrameHessian *newFrameHessian,
                                         SE3 &lastToNew_out,
                                         AffLight &aff_g2l_out, int coarsestLvl,
                                         Vec5 minResForAbort,
                                         Vec5 &lastResiduals,
                                         IOWrap::Output3DWrapper *wrap) {
  assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

  lastResiduals.setConstant(NAN);
  lastFlowIndicators.setConstant(1000);

  new_frame_ = newFrameHessian;
  int maxIterations[] = {10, 20, 50, 50, 50};
  float lambdaExtrapolationLimit = 0.001;

  SE3 refToNew_current = lastToNew_out;
  AffLight aff_g2l_current = aff_g2l_out;

  bool haveRepeated = false;

  for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
    Mat88 H;
    Vec8 b;
    float levelCutoffRepeat = 1;
    Vec6 resOld = calcResPose(lvl, refToNew_current, aff_g2l_current,
                              setting_coarseCutoffTH * levelCutoffRepeat);
    while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
      levelCutoffRepeat *= 2;
      resOld = calcResPose(lvl, refToNew_current, aff_g2l_current,
                           setting_coarseCutoffTH * levelCutoffRepeat);

      if (!setting_debugout_runquiet)
        printf("INCREASING cutoff to %f (ratio is %f)!\n",
               setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
    }

    calcGSSSEPose(lvl, H, b, refToNew_current, aff_g2l_current);

    float lambda = 0.01;

    if (DEBUG_PRINT) {
      Vec2f relAff = AffLight::fromToVecExposure(
                         lastRef->ab_exposure, new_frame_->ab_exposure,
                         lastRef_aff_g2l, aff_g2l_current)
                         .cast<float>();
      printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = "
             "%f)! \t",
             lvl, -1, lambda, 1.0f, "INITIA", 0.0f, resOld[0] / resOld[1], 0,
             (int)resOld[1], 0.0f);
      std::cout << refToNew_current.log().transpose() << " AFF "
                << aff_g2l_current.vec().transpose() << " (rel "
                << relAff.transpose() << ")\n";
    }

    for (int iteration = 0; iteration < maxIterations[lvl]; iteration++) {
      Mat88 Hl = H;
      for (int i = 0; i < 8; i++)
        Hl(i, i) *= (1 + lambda);
      Vec8 inc = Hl.ldlt().solve(-b);

      if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0) // fix a, b
      {
        inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
        inc.tail<2>().setZero();
      }
      if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0) // fix b
      {
        inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
        inc.tail<1>().setZero();
      }
      if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0)) // fix a
      {
        Mat88 HlStitch = Hl;
        Vec8 bStitch = b;
        HlStitch.col(6) = HlStitch.col(7);
        HlStitch.row(6) = HlStitch.row(7);
        bStitch[6] = bStitch[7];
        Vec7 incStitch =
            HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
        inc.setZero();
        inc.head<6>() = incStitch.head<6>();
        inc[6] = 0;
        inc[7] = incStitch[6];
      }

      float extrapFac = 1;
      if (lambda < lambdaExtrapolationLimit)
        extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
      inc *= extrapFac;

      Vec8 incScaled = inc;
      incScaled.segment<3>(0) *= SCALE_XI_ROT;
      incScaled.segment<3>(3) *= SCALE_XI_TRANS;
      incScaled.segment<1>(6) *= SCALE_A;
      incScaled.segment<1>(7) *= SCALE_B;

      if (!std::isfinite(incScaled.sum()))
        incScaled.setZero();

      SE3 refToNew_new =
          SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
      AffLight aff_g2l_new = aff_g2l_current;
      aff_g2l_new.a += incScaled[6];
      aff_g2l_new.b += incScaled[7];

      Vec6 resNew = calcResPose(lvl, refToNew_new, aff_g2l_new,
                                setting_coarseCutoffTH * levelCutoffRepeat);

      bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

      if (DEBUG_PRINT) {
        Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure,
                                                   new_frame_->ab_exposure,
                                                   lastRef_aff_g2l, aff_g2l_new)
                           .cast<float>();
        printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = "
               "%f)! \t",
               lvl, iteration, lambda, extrapFac,
               (accept ? "ACCEPT" : "REJECT"), resOld[0] / resOld[1],
               resNew[0] / resNew[1], (int)resOld[1], (int)resNew[1],
               inc.norm());
        std::cout << refToNew_new.log().transpose() << " AFF "
                  << aff_g2l_new.vec().transpose() << " (rel "
                  << relAff.transpose() << ")\n";
      }
      if (accept) {
        calcGSSSEPose(lvl, H, b, refToNew_new, aff_g2l_new);
        resOld = resNew;
        aff_g2l_current = aff_g2l_new;
        refToNew_current = refToNew_new;
        lambda *= 0.5;
      } else {
        lambda *= 4;
        if (lambda < lambdaExtrapolationLimit)
          lambda = lambdaExtrapolationLimit;
      }

      if (!(inc.norm() > 1e-3)) {
        if (DEBUG_PRINT)
          printf("inc too small, break!\n");
        break;
      }
    }

    // set last residual for that level, as well as flow indicators.
    lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
    lastFlowIndicators = resOld.segment<3>(2);
    if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
      return false;

    if (levelCutoffRepeat > 1 && !haveRepeated) {
      lvl++;
      haveRepeated = true;
    }
  }

  // if (haveRepeated) {
  //   printf("REPEAT LEVEL!\n");
  // }

  // set!
  lastToNew_out = refToNew_current;
  aff_g2l_out = aff_g2l_current;

  if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2)) ||
      (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
    return false;

  Vec2f relAff =
      AffLight::fromToVecExposure(lastRef->ab_exposure, new_frame_->ab_exposure,
                                  lastRef_aff_g2l, aff_g2l_out)
          .cast<float>();

  if ((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5)) ||
      (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
    return false;

  if (setting_affineOptModeA < 0)
    aff_g2l_out.a = 0;
  if (setting_affineOptModeB < 0)
    aff_g2l_out.b = 0;

  if (DEBUG_PLOT) {
    calcResPose(0, lastToNew_out, aff_g2l_out, setting_coarseCutoffTH, true);
  }

  return true;
}

void TrackerAndScaler::calcGSSSEPose(int lvl, Mat88 &H_out, Vec8 &b_out,
                                     const SE3 &refToNew, AffLight aff_g2l) {
  pose_acc_.initialize();

  __m128 fxl = _mm_set1_ps(fx_[lvl]);
  __m128 fyl = _mm_set1_ps(fy_[lvl]);
  __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
  __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(
      lastRef->ab_exposure, new_frame_->ab_exposure, lastRef_aff_g2l,
      aff_g2l)[0]));

  __m128 one = _mm_set1_ps(1);
  __m128 minusOne = _mm_set1_ps(-1);
  __m128 zero = _mm_set1_ps(0);

  int n = pose_buf_warped_n_;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 dx = _mm_mul_ps(_mm_load_ps(pose_buf_warped_dx_ + i), fxl);
    __m128 dy = _mm_mul_ps(_mm_load_ps(pose_buf_warped_dy_ + i), fyl);
    __m128 u = _mm_load_ps(pose_buf_warped_u_ + i);
    __m128 v = _mm_load_ps(pose_buf_warped_v_ + i);
    __m128 id = _mm_load_ps(pose_buf_warped_idepth_ + i);

    pose_acc_.updateSSE_eighted(
        _mm_mul_ps(id, dx), _mm_mul_ps(id, dy),
        _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx),
                                                   _mm_mul_ps(v, dy)))),
        _mm_sub_ps(
            zero,
            _mm_add_ps(_mm_mul_ps(_mm_mul_ps(u, v), dx),
                       _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),
        _mm_add_ps(_mm_mul_ps(_mm_mul_ps(u, v), dy),
                   _mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),
        _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),
        _mm_mul_ps(a,
                   _mm_sub_ps(b0, _mm_load_ps(pose_buf_warped_refColor_ + i))),
        minusOne, _mm_load_ps(pose_buf_warped_residual_ + i),
        _mm_load_ps(pose_buf_warped_weight_ + i));
  }

  pose_acc_.finish();
  H_out = pose_acc_.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
  b_out = pose_acc_.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

  H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
  H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
  H_out.block<8, 1>(0, 6) *= SCALE_A;
  H_out.block<8, 1>(0, 7) *= SCALE_B;
  H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
  H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
  H_out.block<1, 8>(6, 0) *= SCALE_A;
  H_out.block<1, 8>(7, 0) *= SCALE_B;
  b_out.segment<3>(0) *= SCALE_XI_ROT;
  b_out.segment<3>(3) *= SCALE_XI_TRANS;
  b_out.segment<1>(6) *= SCALE_A;
  b_out.segment<1>(7) *= SCALE_B;
}

Vec6 TrackerAndScaler::calcResPose(int lvl, const SE3 &refToNew,
                                   AffLight aff_g2l, float cutoffTH,
                                   bool plot_img) {
  float E = 0;
  int numTermsInE = 0;
  int numTermsInWarped = 0;
  int numSaturated = 0;

  int wl = w_[lvl];
  int hl = h_[lvl];
  Eigen::Vector3f *dINewl = new_frame_->dIp[lvl];
  float fxl = fx_[lvl];
  float fyl = fy_[lvl];
  float cxl = cx_[lvl];
  float cyl = cy_[lvl];

  Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki_[lvl]);
  Vec3f t = (refToNew.translation()).cast<float>();
  Vec2f affLL =
      AffLight::fromToVecExposure(lastRef->ab_exposure, new_frame_->ab_exposure,
                                  lastRef_aff_g2l, aff_g2l)
          .cast<float>();

  float sumSquaredShiftT = 0;
  float sumSquaredShiftRT = 0;
  float sumSquaredShiftNum = 0;

  float maxEnergy =
      2 * setting_huberTH * cutoffTH -
      setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

  MinimalImageB3 *resImage = 0;
  if (plot_img) {
    resImage = new MinimalImageB3(wl, hl);
    resImage->setConst(Vec3b(255, 255, 255));
  }

  int nl = pc_n_[lvl];
  float *lpc_u = pc_u_[lvl];
  float *lpc_v = pc_v_[lvl];
  float *lpc_idepth = pc_idepth_[lvl];
  float *lpc_color = pc_color_[lvl];

  for (int i = 0; i < nl; i++) {
    float id = lpc_idepth[i];
    float x = lpc_u[i];
    float y = lpc_v[i];

    Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
    float u = pt[0] / pt[2];
    float v = pt[1] / pt[2];
    float Ku = fxl * u + cxl;
    float Kv = fyl * v + cyl;
    float new_idepth = id / pt[2];

    if (lvl == 0 && i % 32 == 0) {
      // translation only (positive)
      Vec3f ptT = Ki_[lvl] * Vec3f(x, y, 1) + t * id;
      float uT = ptT[0] / ptT[2];
      float vT = ptT[1] / ptT[2];
      float KuT = fxl * uT + cxl;
      float KvT = fyl * vT + cyl;

      // translation only (negative)
      Vec3f ptT2 = Ki_[lvl] * Vec3f(x, y, 1) - t * id;
      float uT2 = ptT2[0] / ptT2[2];
      float vT2 = ptT2[1] / ptT2[2];
      float KuT2 = fxl * uT2 + cxl;
      float KvT2 = fyl * vT2 + cyl;

      // translation and rotation (negative)
      Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
      float u3 = pt3[0] / pt3[2];
      float v3 = pt3[1] / pt3[2];
      float Ku3 = fxl * u3 + cxl;
      float Kv3 = fyl * v3 + cyl;

      // translation and rotation (positive)
      // already have it.

      sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
      sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
      sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
      sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
      sumSquaredShiftNum += 2;
    }

    if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
      continue;

    float refColor = lpc_color[i];
    Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
    if (!std::isfinite((float)hitColor[0]))
      continue;
    float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
    float hw =
        fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    if (fabs(residual) > cutoffTH) {
      if (plot_img)
        resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
      E += maxEnergy;
      numTermsInE++;
      numSaturated++;
    } else {
      if (plot_img)
        resImage->setPixel4(
            lpc_u[i], lpc_v[i],
            Vec3b(residual + 128, residual + 128, residual + 128));

      E += hw * residual * residual * (2 - hw);
      numTermsInE++;

      pose_buf_warped_idepth_[numTermsInWarped] = new_idepth;
      pose_buf_warped_u_[numTermsInWarped] = u;
      pose_buf_warped_v_[numTermsInWarped] = v;
      pose_buf_warped_dx_[numTermsInWarped] = hitColor[1];
      pose_buf_warped_dy_[numTermsInWarped] = hitColor[2];
      pose_buf_warped_residual_[numTermsInWarped] = residual;
      pose_buf_warped_weight_[numTermsInWarped] = hw;
      pose_buf_warped_refColor_[numTermsInWarped] = lpc_color[i];
      numTermsInWarped++;
    }
  }

  while (numTermsInWarped % 4 != 0) {
    pose_buf_warped_idepth_[numTermsInWarped] = 0;
    pose_buf_warped_u_[numTermsInWarped] = 0;
    pose_buf_warped_v_[numTermsInWarped] = 0;
    pose_buf_warped_dx_[numTermsInWarped] = 0;
    pose_buf_warped_dy_[numTermsInWarped] = 0;
    pose_buf_warped_residual_[numTermsInWarped] = 0;
    pose_buf_warped_weight_[numTermsInWarped] = 0;
    pose_buf_warped_refColor_[numTermsInWarped] = 0;
    numTermsInWarped++;
  }
  pose_buf_warped_n_ = numTermsInWarped;

  if (plot_img) {
    IOWrap::displayImage("Tracking Residual", resImage, false);
    IOWrap::waitKey(0);
    delete resImage;
  }

  Vec6 rs;
  rs[0] = E;
  rs[1] = numTermsInE;
  rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
  rs[3] = 0;
  rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
  rs[5] = numSaturated / (float)numTermsInE;

  return rs;
}

float TrackerAndScaler::optimizeScale(FrameHessian *fh1, float &scale,
                                      int coarsestLvl) {
  assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);
  fh1_ = fh1;

  Vec5 last_residuals;
  last_residuals.setConstant(NAN);

  int maxIterations[] = {10, 20, 50, 50, 50};
  float lambdaExtrapolationLimit = 0.001;

  float scale_current = scale;

  bool haveRepeated = false;

  for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
    float H;
    float b;
    float levelCutoffRepeat = 1;
    Vec6 resOld = calcResScale(lvl, scale_current,
                               setting_coarseCutoffTH * levelCutoffRepeat);
    while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
      levelCutoffRepeat *= 2;
      resOld = calcResScale(lvl, scale_current,
                            setting_coarseCutoffTH * levelCutoffRepeat);

      if (!setting_debugout_runquiet)
        printf("INCREASING cutoff to %f (ratio is %f)!\n",
               setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
    }

    calcGSSSEScale(lvl, H, b, scale_current);

    float lambda = 0.01;

    if (DEBUG_PRINT) {
      printf(
          "lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (inc = %f)! \n",
          lvl, -1, lambda, 1.0f, "INITIA", 0.0f, resOld[0] / resOld[1], 0,
          (int)resOld[1], 0.0f);
      std::cout << " Current scale " << scale_current << std::endl;
    }

    for (int iteration = 0; iteration < maxIterations[lvl]; iteration++) {
      float Hl = H;
      Hl *= (1 + lambda);
      float inc = -b / Hl;

      float extrapFac = 1;
      if (lambda < lambdaExtrapolationLimit)
        extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
      inc *= extrapFac;

      if (!std::isfinite(inc) || fabs(inc) > scale_current)
        inc = 0.0;

      float scale_new = scale_current + inc;

      Vec6 resNew = calcResScale(lvl, scale_new,
                                 setting_coarseCutoffTH * levelCutoffRepeat);

      bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

      if (DEBUG_PRINT) {
        printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (inc = "
               "%f)! \t",
               lvl, iteration, lambda, extrapFac,
               (accept ? "ACCEPT" : "REJECT"), resOld[0] / resOld[1],
               resNew[0] / resNew[1], (int)resOld[1], (int)resNew[1], inc);
        std::cout << " New scale " << scale_new << std::endl;
      }

      if (accept) {
        calcGSSSEScale(lvl, H, b, scale_new);
        resOld = resNew;
        scale_current = scale_new;
        lambda *= 0.5;
      } else {
        lambda *= 4;
        if (lambda < lambdaExtrapolationLimit)
          lambda = lambdaExtrapolationLimit;
      }

      if (!(inc > 1e-3)) {
        if (DEBUG_PRINT)
          printf("inc too small, break!\n");
        break;
      }
    }

    // set last residual for that level, as well as flow indicators.
    last_residuals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));

    if (levelCutoffRepeat > 1 && !haveRepeated) {
      lvl++;
      haveRepeated = true;
    }
  }

  // set!
  scale = scale_current;

  if (DEBUG_PLOT) {
    printf("scale = %.2f\n", scale);
    // calcResScale(1, scale / 10, setting_coarseCutoffTH, true);
    calcResScale(1, scale, setting_coarseCutoffTH, true);
    // calcResScale(1, scale * 10, setting_coarseCutoffTH, true);
  }

  return last_residuals[0];
}

void TrackerAndScaler::calcGSSSEScale(int lvl, float &H_out, float &b_out,
                                      float scale) {
  scale_acc_.initialize();

  __m128 fx1l = _mm_set1_ps(fx1_[lvl]);
  __m128 fy1l = _mm_set1_ps(fy1_[lvl]);

  __m128 s = _mm_set1_ps(scale);
  __m128 tx = _mm_set1_ps(tfm_f1_f0_.translation()[0]);
  __m128 ty = _mm_set1_ps(tfm_f1_f0_.translation()[1]);
  __m128 tz = _mm_set1_ps(tfm_f1_f0_.translation()[2]);

  __m128 one = _mm_set1_ps(1);

  int n = scale_buf_warped_n_;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 dxfx = _mm_mul_ps(_mm_load_ps(scale_buf_warped_dx_ + i), fx1l);
    __m128 dyfy = _mm_mul_ps(_mm_load_ps(scale_buf_warped_dy_ + i), fy1l);
    __m128 rx1 = _mm_load_ps(scale_buf_warped_rx1_ + i);
    __m128 rx2 = _mm_load_ps(scale_buf_warped_rx2_ + i);
    __m128 rx3 = _mm_load_ps(scale_buf_warped_rx3_ + i);

    __m128 deno_sqrt = _mm_add_ps(_mm_mul_ps(s, rx3), tz);
    __m128 deno = _mm_div_ps(one, _mm_mul_ps(deno_sqrt, deno_sqrt));

    __m128 xno = _mm_sub_ps(_mm_mul_ps(rx1, tz), _mm_mul_ps(rx3, tx));
    __m128 yno = _mm_sub_ps(_mm_mul_ps(rx2, tz), _mm_mul_ps(rx3, ty));

    scale_acc_.updateSSE_oneed(
        _mm_add_ps(_mm_mul_ps(dxfx, _mm_mul_ps(deno, xno)),
                   _mm_mul_ps(dyfy, _mm_mul_ps(deno, yno))),
        _mm_load_ps(scale_buf_warped_residual_ + i),
        _mm_load_ps(scale_buf_warped_weight_ + i));
  }

  scale_acc_.finish();
  H_out = scale_acc_.hessian_(0, 0) * (1.0f / n);
  b_out = scale_acc_.hessian_(0, 1) * (1.0f / n);
}

Vec6 TrackerAndScaler::calcResScale(int lvl, float scale, float cutoffTH,
                                    bool plot_img) {
  float E = 0;
  int numTermsInE = 0;
  int numTermsInWarped = 0;
  int numSaturated = 0;

  int wl = w_[lvl];
  int hl = h_[lvl];
  Eigen::Vector3f *dINewl = fh1_->dIp[lvl];
  float fx1l = fx1_[lvl];
  float fy1l = fy1_[lvl];
  float cx1l = cx1_[lvl];
  float cy1l = cy1_[lvl];

  Mat33f rot_f1_f0_K0_i =
      (tfm_f1_f0_.rotationMatrix().cast<float>() * Ki_[lvl]);
  Vec3f tsl_f1_f0 = (tfm_f1_f0_.translation()).cast<float>();

  float sumSquaredShiftT = 0;
  float sumSquaredShiftRT = 0;
  float sumSquaredShiftNum = 0;

  float maxEnergy =
      2 * setting_huberTH * cutoffTH -
      setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

  MinimalImageB3 *resImage = 0;
  MinimalImageB3 *projImage = 0;
  if (plot_img) {
    resImage = new MinimalImageB3(wl, hl);
    resImage->setConst(Vec3b(255, 255, 255));

    projImage = new MinimalImageB3(wl, hl);
    projImage->setBlack();
    for (int i = 0; i < h_[lvl] * w_[lvl]; i++) {
      int c = fh1_->dIp[lvl][i][0] * 0.9f;
      if (c > 255)
        c = 255;
      projImage->at(i) = Vec3b(c, c, c);
    }
  }

  int nl = pc_n_[lvl];
  float *lpc_u = pc_u_[lvl];
  float *lpc_v = pc_v_[lvl];
  float *lpc_idepth = pc_idepth_[lvl];
  float *lpc_color = pc_color_[lvl];

  for (int i = 0; i < nl; i++) {
    float id = lpc_idepth[i];
    float x = lpc_u[i];
    float y = lpc_v[i];

    Vec3f pt = scale * rot_f1_f0_K0_i * Vec3f(x, y, 1) + tsl_f1_f0 * id;
    float u = pt[0] / pt[2];
    float v = pt[1] / pt[2];
    float Ku = fx1l * u + cx1l;
    float Kv = fy1l * v + cy1l;
    float new_idepth = id / pt[2];

    Vec3f rx = rot_f1_f0_K0_i * Vec3f(x, y, 1) / id;

    if (lvl == 0 && i % 32 == 0) {
      // translation only (positive)
      Vec3f ptT = scale * Ki_[lvl] * Vec3f(x, y, 1) + tsl_f1_f0 * id;
      float uT = ptT[0] / ptT[2];
      float vT = ptT[1] / ptT[2];
      float KuT = fx1l * uT + cx1l;
      float KvT = fy1l * vT + cy1l;

      // translation only (negative)
      Vec3f ptT2 = scale * Ki_[lvl] * Vec3f(x, y, 1) - tsl_f1_f0 * id;
      float uT2 = ptT2[0] / ptT2[2];
      float vT2 = ptT2[1] / ptT2[2];
      float KuT2 = fx1l * uT2 + cx1l;
      float KvT2 = fy1l * vT2 + cy1l;

      // translation and rotation (negative)
      Vec3f pt3 = scale * rot_f1_f0_K0_i * Vec3f(x, y, 1) - tsl_f1_f0 * id;
      float u3 = pt3[0] / pt3[2];
      float v3 = pt3[1] / pt3[2];
      float Ku3 = fx1l * u3 + cx1l;
      float Kv3 = fy1l * v3 + cy1l;

      // translation and rotation (positive)
      // already have it.

      sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
      sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
      sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
      sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
      sumSquaredShiftNum += 2;
    }

    if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
      continue;

    float refColor = lpc_color[i];
    Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
    if (!std::isfinite((float)hitColor[0]))
      continue;
    float residual = hitColor[0] - refColor;
    float hw =
        fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    if (plot_img)
      projImage->setPixel4(Ku, Kv, Vec3b(refColor, refColor, refColor));

    if (fabs(residual) > cutoffTH) {
      if (plot_img)
        resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
      E += maxEnergy;
      numTermsInE++;
      numSaturated++;
    } else {
      if (plot_img)
        resImage->setPixel4(
            lpc_u[i], lpc_v[i],
            Vec3b(residual + 128, residual + 128, residual + 128));
      E += hw * residual * residual * (2 - hw);
      numTermsInE++;

      scale_buf_warped_rx1_[numTermsInWarped] = rx[0];
      scale_buf_warped_rx2_[numTermsInWarped] = rx[1];
      scale_buf_warped_rx3_[numTermsInWarped] = rx[2];
      scale_buf_warped_dx_[numTermsInWarped] = hitColor[1];
      scale_buf_warped_dy_[numTermsInWarped] = hitColor[2];
      scale_buf_warped_residual_[numTermsInWarped] = residual;
      scale_buf_warped_weight_[numTermsInWarped] = hw;
      scale_buf_warped_ref_color_[numTermsInWarped] = lpc_color[i];
      numTermsInWarped++;
    }
  }

  while (numTermsInWarped % 4 != 0) {
    scale_buf_warped_rx1_[numTermsInWarped] = 0;
    scale_buf_warped_rx2_[numTermsInWarped] = 0;
    scale_buf_warped_rx3_[numTermsInWarped] = 0;
    scale_buf_warped_dx_[numTermsInWarped] = 0;
    scale_buf_warped_dy_[numTermsInWarped] = 0;
    scale_buf_warped_residual_[numTermsInWarped] = 0;
    scale_buf_warped_weight_[numTermsInWarped] = 0;
    scale_buf_warped_ref_color_[numTermsInWarped] = 0;
    numTermsInWarped++;
  }
  scale_buf_warped_n_ = numTermsInWarped;

  if (plot_img) {
    IOWrap::displayImage("Scale Residual", resImage, false);
    IOWrap::displayImage("Projection", projImage, false);
    IOWrap::waitKey(0);
    delete resImage;
    delete projImage;
  }

  Vec6 rs;
  rs[0] = E;
  rs[1] = numTermsInE;
  rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
  rs[3] = 0;
  rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
  rs[5] = numSaturated / (float)numTermsInE;

  return rs;
}

CoarseDistanceMap::CoarseDistanceMap(int ww, int hh) {
  fwdWarpedIDDistFinal = new float[ww * hh / 4];

  bfs_list1_ = new Eigen::Vector2i[ww * hh / 4];
  bfs_list2_ = new Eigen::Vector2i[ww * hh / 4];

  int fac = 1 << (pyrLevelsUsed - 1);

  coarse_projection_grid_ =
      new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
  coarse_projection_grid_num_ = new int[ww * hh / (fac * fac)];

  w_[0] = h_[0] = 0;
}

CoarseDistanceMap::~CoarseDistanceMap() {
  delete[] fwdWarpedIDDistFinal;
  delete[] bfs_list1_;
  delete[] bfs_list2_;
  delete[] coarse_projection_grid_;
  delete[] coarse_projection_grid_num_;
}

void CoarseDistanceMap::makeDistanceMap(
    std::vector<FrameHessian *> frameHessians, FrameHessian *frame) {
  int w1 = w_[1];
  int h1 = h_[1];
  int wh1 = w1 * h1;
  for (int i = 0; i < wh1; i++)
    fwdWarpedIDDistFinal[i] = 1000;

  // make coarse tracking templates for latstRef.
  int numItems = 0;

  for (FrameHessian *fh : frameHessians) {
    if (frame == fh)
      continue;

    SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
    Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
    Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

    for (PointHessian *ph : fh->pointHessians) {
      assert(ph->status == PointHessian::ACTIVE);
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;
      if (!(u > 0 && v > 0 && u < w_[1] && v < h_[1]))
        continue;
      fwdWarpedIDDistFinal[u + w1 * v] = 0;
      bfs_list1_[numItems] = Eigen::Vector2i(u, v);
      numItems++;
    }
  }

  growDistBFS(numItems);
}

void CoarseDistanceMap::makeInlierVotes(
    std::vector<FrameHessian *> frameHessians) {}

void CoarseDistanceMap::growDistBFS(int bfsNum) {
  assert(w_[0] != 0);
  int w1 = w_[1], h1 = h_[1];
  for (int k = 1; k < 40; k++) {
    int bfsNum2 = bfsNum;
    std::swap<Eigen::Vector2i *>(bfs_list1_, bfs_list2_);
    bfsNum = 0;

    if (k % 2 == 0) {
      for (int i = 0; i < bfsNum2; i++) {
        int x = bfs_list2_[i][0];
        int y = bfs_list2_[i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
          continue;
        int idx = x + y * w1;

        if (fwdWarpedIDDistFinal[idx + 1] > k) {
          fwdWarpedIDDistFinal[idx + 1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x + 1, y);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1] > k) {
          fwdWarpedIDDistFinal[idx - 1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x - 1, y);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx + w1] > k) {
          fwdWarpedIDDistFinal[idx + w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - w1] > k) {
          fwdWarpedIDDistFinal[idx - w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x, y - 1);
          bfsNum++;
        }
      }
    } else {
      for (int i = 0; i < bfsNum2; i++) {
        int x = bfs_list2_[i][0];
        int y = bfs_list2_[i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
          continue;
        int idx = x + y * w1;

        if (fwdWarpedIDDistFinal[idx + 1] > k) {
          fwdWarpedIDDistFinal[idx + 1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x + 1, y);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1] > k) {
          fwdWarpedIDDistFinal[idx - 1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x - 1, y);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx + w1] > k) {
          fwdWarpedIDDistFinal[idx + w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - w1] > k) {
          fwdWarpedIDDistFinal[idx - w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x, y - 1);
          bfsNum++;
        }

        if (fwdWarpedIDDistFinal[idx + 1 + w1] > k) {
          fwdWarpedIDDistFinal[idx + 1 + w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1 + w1] > k) {
          fwdWarpedIDDistFinal[idx - 1 + w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1 - w1] > k) {
          fwdWarpedIDDistFinal[idx - 1 - w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx + 1 - w1] > k) {
          fwdWarpedIDDistFinal[idx + 1 - w1] = k;
          bfs_list1_[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
          bfsNum++;
        }
      }
    }
  }
}

void CoarseDistanceMap::addIntoDistFinal(int u, int v) {
  if (w_[0] == 0)
    return;
  bfs_list1_[0] = Eigen::Vector2i(u, v);
  fwdWarpedIDDistFinal[u + w_[1] * v] = 0;
  growDistBFS(1);
}

void CoarseDistanceMap::makeK(CalibHessian *HCalib) {
  w_[0] = wG[0];
  h_[0] = hG[0];

  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w_[level] = w_[0] >> level;
    h_[level] = h_[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < pyrLevelsUsed; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
  }
}

} // namespace dso
