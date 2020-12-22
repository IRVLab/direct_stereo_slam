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

#include "PoseEstimator.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "IOWrapper/ImageRW.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <algorithm>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

#define DEBUG_PLOT false
#define DEBUG_PRINT false

namespace dso {

template <int b, typename T>
T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
  const int padT = 1 + ((1 << b) / sizeof(T));
  T *ptr = new T[size + padT];
  rawPtrVec.push_back(ptr);
  T *alignedPtr = (T *)((((uintptr_t)(ptr + padT)) >> b) << b);
  return alignedPtr;
}

PoseEstimator::PoseEstimator(int ww, int hh) : ref_aff_g2l_(0, 0) {
  // warped buffers
  buf_warped_idepth_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_u_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_v_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_dx_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_dy_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_residual_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_weight_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);
  buf_warped_ref_color_ = allocAligned<4, float>(ww * hh, ptr_to_delete_);

  new_frame_ = 0;
  w_[0] = h_[0] = 0;
}

PoseEstimator::~PoseEstimator() {
  for (float *ptr : ptr_to_delete_)
    delete[] ptr;
  ptr_to_delete_.clear();
}

void PoseEstimator::makeK(const std::vector<float> &cam) {
  w_[0] = wG[0];
  h_[0] = hG[0];

  fx_[0] = cam[0];
  fy_[0] = cam[1];
  cx_[0] = cam[2];
  cy_[0] = cam[3];

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w_[level] = w_[0] >> level;
    h_[level] = h_[0] >> level;
    fx_[level] = fx_[level - 1] * 0.5;
    fy_[level] = fy_[level - 1] * 0.5;
    cx_[level] = (cx_[0] + 0.5) / ((int)1 << level) - 0.5;
    cy_[level] = (cy_[0] + 0.5) / ((int)1 << level) - 0.5;
  }
}

void PoseEstimator::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out,
                              const SE3 &refToNew, AffLight aff_g2l) {
  acc_.initialize();

  __m128 fxl = _mm_set1_ps(fx_[lvl]);
  __m128 fyl = _mm_set1_ps(fy_[lvl]);
  __m128 b0 = _mm_set1_ps(ref_aff_g2l_.b);
  __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(
      ref_ab_exposure_, new_frame_->ab_exposure, ref_aff_g2l_, aff_g2l)[0]));

  __m128 one = _mm_set1_ps(1);
  __m128 minusOne = _mm_set1_ps(-1);
  __m128 zero = _mm_set1_ps(0);

  int n = buf_warped_n_;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx_ + i), fxl);
    __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy_ + i), fyl);
    __m128 u = _mm_load_ps(buf_warped_u_ + i);
    __m128 v = _mm_load_ps(buf_warped_v_ + i);
    __m128 id = _mm_load_ps(buf_warped_idepth_ + i);

    acc_.updateSSE_eighted(
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
        _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_ref_color_ + i))),
        minusOne, _mm_load_ps(buf_warped_residual_ + i),
        _mm_load_ps(buf_warped_weight_ + i));
  }

  acc_.finish();
  H_out = acc_.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
  b_out = acc_.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

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

Vec6 PoseEstimator::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l,
                            float cutoffTH, bool plot_img) {
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

  Mat33f R = refToNew.rotationMatrix().cast<float>();
  Vec3f t = refToNew.translation().cast<float>();
  Vec2f affLL =
      AffLight::fromToVecExposure(ref_ab_exposure_, new_frame_->ab_exposure,
                                  ref_aff_g2l_, aff_g2l)
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
    resImage->setBlack();
    for (int i = 0; i < h_[lvl] * w_[lvl]; i++) {
      int c = new_frame_->dIp[lvl][i][0] * 0.9f;
      if (c > 255)
        c = 255;
      resImage->at(i) = Vec3b(c, c, c);
    }
  }

  for (size_t i = 0; i < pts_.size(); i++) {
    float x = pts_[i].first(0);
    float y = pts_[i].first(1);
    float z = pts_[i].first(2);
    float u0 = x / z;
    float v0 = y / z;
    float Ku0 = fxl * u0 + cxl;
    float Kv0 = fyl * v0 + cyl;

    Vec3f pt = R * Vec3f(x, y, z) + t;
    float u = pt[0] / pt[2];
    float v = pt[1] / pt[2];
    float Ku = fxl * u + cxl;
    float Kv = fyl * v + cyl;
    float new_idepth = 1 / pt[2];

    if (lvl == 0 && i % 32 == 0) {
      // translation only (positive)
      Vec3f ptT = Vec3f(x, y, 1) + t;
      float uT = ptT[0] / ptT[2];
      float vT = ptT[1] / ptT[2];
      float KuT = fxl * uT + cxl;
      float KvT = fyl * vT + cyl;

      // translation only (negative)
      Vec3f ptT2 = Vec3f(x, y, 1) - t;
      float uT2 = ptT2[0] / ptT2[2];
      float vT2 = ptT2[1] / ptT2[2];
      float KuT2 = fxl * uT2 + cxl;
      float KvT2 = fyl * vT2 + cyl;

      // translation and rotation (negative)
      Vec3f pt3 = R * Vec3f(x, y, 1) - t;
      float u3 = pt3[0] / pt3[2];
      float v3 = pt3[1] / pt3[2];
      float Ku3 = fxl * u3 + cxl;
      float Kv3 = fyl * v3 + cyl;

      // translation and rotation (positive)
      // already have it.

      sumSquaredShiftT += (KuT - Ku0) * (KuT - Ku0) + (KvT - Kv0) * (KvT - Kv0);
      sumSquaredShiftT +=
          (KuT2 - Ku0) * (KuT2 - Ku0) + (KvT2 - Kv0) * (KvT2 - Kv0);
      sumSquaredShiftRT += (Ku - Ku0) * (Ku - Ku0) + (Kv - Kv0) * (Kv - Kv0);
      sumSquaredShiftRT +=
          (Ku3 - Ku0) * (Ku3 - Ku0) + (Kv3 - Kv0) * (Kv3 - Kv0);
      sumSquaredShiftNum += 2;
    }

    if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
      continue;

    float refColor = pts_[i].second[lvl];
    Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
    if (!std::isfinite((float)hitColor[0]))
      continue;
    float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
    // printf("%.1f - (%.1f * %.1f + %.1f) = %.1f\n", hitColor[0], affLL[0],
    //        refColor, affLL[1], residual);
    float hw =
        fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    if (plot_img)
      resImage->setPixel4(Ku, Kv, Vec3b(refColor, refColor, refColor));
    if (fabs(residual) > cutoffTH) {
      E += maxEnergy;
      numTermsInE++;
      numSaturated++;
    } else {
      E += hw * residual * residual * (2 - hw);
      numTermsInE++;

      buf_warped_idepth_[numTermsInWarped] = new_idepth;
      buf_warped_u_[numTermsInWarped] = u;
      buf_warped_v_[numTermsInWarped] = v;
      buf_warped_dx_[numTermsInWarped] = hitColor[1];
      buf_warped_dy_[numTermsInWarped] = hitColor[2];
      buf_warped_residual_[numTermsInWarped] = residual;
      buf_warped_weight_[numTermsInWarped] = hw;
      buf_warped_ref_color_[numTermsInWarped] = refColor;
      numTermsInWarped++;
    }
  }

  while (numTermsInWarped % 4 != 0) {
    buf_warped_idepth_[numTermsInWarped] = 0;
    buf_warped_u_[numTermsInWarped] = 0;
    buf_warped_v_[numTermsInWarped] = 0;
    buf_warped_dx_[numTermsInWarped] = 0;
    buf_warped_dy_[numTermsInWarped] = 0;
    buf_warped_residual_[numTermsInWarped] = 0;
    buf_warped_weight_[numTermsInWarped] = 0;
    buf_warped_ref_color_[numTermsInWarped] = 0;
    numTermsInWarped++;
  }
  buf_warped_n_ = numTermsInWarped;

  if (plot_img) {
    IOWrap::displayImage("Loop Pose Residual", resImage, false);
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

bool PoseEstimator::estimate(
    const std::vector<std::pair<Eigen::Vector3d, float *>> &pts,
    float ref_ab_exposure, FrameHessian *new_fh,
    const std::vector<float> &new_cam, int coarsest_lvl,
    Eigen::Matrix4d &ref_to_new, float &pose_error) {
  int maxIterations[] = {10, 20, 50, 50, 50};
  float lambdaExtrapolationLimit = 0.001;
  assert(coarsest_lvl < 5 && coarsest_lvl < pyrLevelsUsed);

  makeK(new_cam);
  pts_ = pts;

  int lastInners[PYR_LEVELS];
  Vec5 lastResiduals;
  lastResiduals.setConstant(NAN);

  new_frame_ = new_fh;

  ref_aff_g2l_ = AffLight();
  ref_ab_exposure_ = ref_ab_exposure;
  AffLight aff_g2l_current = AffLight();

  SE3 refToNew_current(ref_to_new.block<3, 3>(0, 0),
                       ref_to_new.block<3, 1>(0, 3));

  bool haveRepeated = false;

  for (int lvl = coarsest_lvl; lvl >= 0; lvl--) {
    Mat88 H;
    Vec8 b;
    float levelCutoffRepeat = 1;
    Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current,
                          setting_coarseCutoffTH * levelCutoffRepeat);
    while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
      levelCutoffRepeat *= 2;
      resOld = calcRes(lvl, refToNew_current, aff_g2l_current,
                       setting_coarseCutoffTH * levelCutoffRepeat);

      if (!setting_debugout_runquiet)
        printf("INCREASING cutoff to %f (ratio is %f)!\n",
               setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
    }

    calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

    float lambda = 0.01;

    if (DEBUG_PRINT) {
      Vec2f relAff =
          AffLight::fromToVecExposure(ref_ab_exposure_, new_frame_->ab_exposure,
                                      ref_aff_g2l_, aff_g2l_current)
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

      Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new,
                            setting_coarseCutoffTH * levelCutoffRepeat);

      bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

      if (DEBUG_PRINT) {
        Vec2f relAff = AffLight::fromToVecExposure(ref_ab_exposure_,
                                                   new_frame_->ab_exposure,
                                                   ref_aff_g2l_, aff_g2l_new)
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
        calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
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
    lastInners[lvl] = resOld[1];

    if (levelCutoffRepeat > 1 && !haveRepeated) {
      lvl++;
      haveRepeated = true;
    }
  }

  // set!
  ref_to_new = refToNew_current.matrix();
  pose_error = lastResiduals[0];

  // check if the final pose is:
  // 1. affine coefficients are reasonable
  bool aff_good = true;
  if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_current.a) > 1.2)) ||
      (setting_affineOptModeB != 0 && (fabsf(aff_g2l_current.b) > 200)))
    aff_good = false;

  Vec2f relAff =
      AffLight::fromToVecExposure(ref_ab_exposure_, new_frame_->ab_exposure,
                                  ref_aff_g2l_, aff_g2l_current)
          .cast<float>();

  if ((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5)) ||
      (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
    aff_good = false;

  // 2. with small residual;
  bool low_res = pose_error < RES_THRES;

  // 3. with high inner ratio;
  int inlier_percent = 100 * float(lastInners[0]) / pts.size();
  bool enough_inlier = inlier_percent > INNER_PERCENT;

  // // 4. close enough as loop closure
  // auto tfm_se3 = SE3(ref_to_new).log().eval();
  // float t = tfm_se3.head<3>().norm();
  // float r = tfm_se3.tail<3>().norm();
  // bool tfm_close = t < TRANS_THRES && r < ROT_THRES;

  printf("direct: (%5.2f, %3d%%, %s)  ", pose_error, inlier_percent,
         aff_good ? "Y" : "N");

  // if (low_res && aff_good && !(enough_inlier && tfm_close)) {
  //   printf("\n");
  //   calcRes(0, refToNew_current, aff_g2l_current, setting_coarseCutoffTH,
  //   true);
  // }

  if (DEBUG_PLOT) {
    calcRes(1, refToNew_current, aff_g2l_current, setting_coarseCutoffTH, true);
  }

  return aff_good && low_res && enough_inlier;
}

} // namespace dso