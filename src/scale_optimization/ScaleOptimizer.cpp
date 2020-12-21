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

#include "ScaleOptimizer.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <opencv2/core/eigen.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

template <int b, typename T>
T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
  const int padT = 1 + ((1 << b) / sizeof(T));
  T *ptr = new T[size + padT];
  rawPtrVec.push_back(ptr);
  T *alignedPtr = (T *)((((uintptr_t)(ptr + padT)) >> b) << b);
  return alignedPtr;
}

ScaleOptimizer::ScaleOptimizer(Undistort *undistorter1,
                               const std::vector<double> &tfm_vec)
    : frame0_id_(-1), frame0_(0), frame1_(0), undistorter1_(undistorter1) {
  // tranformation form frame0 to frame1
  Eigen::Matrix4d tfm_eigen;
  cv::Mat tfm_stereo_cv = cv::Mat(tfm_vec);
  tfm_stereo_cv = tfm_stereo_cv.reshape(0, 4);
  cv::cv2eigen(tfm_stereo_cv, tfm_eigen);
  tfm_f1_f0_ = SE3(tfm_eigen);

  // camera1 parameters
  w_[0] = (int)undistorter1_->getSize()[0];
  h_[0] = (int)undistorter1_->getSize()[1];
  Eigen::Matrix3f K1 = undistorter1_->getK().cast<float>();
  fx1_[0] = K1(0, 0);
  fy1_[0] = K1(1, 1);
  cx1_[0] = K1(0, 2);
  cy1_[0] = K1(1, 2);
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w_[level] = w_[0] >> level;
    h_[level] = h_[0] >> level;
    fx1_[level] = fx1_[level - 1] * 0.5;
    fy1_[level] = fy1_[level - 1] * 0.5;
    cx1_[level] = (cx1_[0] + 0.5) / ((int)1 << level) - 0.5;
    cy1_[level] = (cy1_[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  // make coarse tracking templates.
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = w_[0] >> lvl;
    int hl = h_[0] >> lvl;

    idepth_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    weight_sums_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    weight_sums_bak_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);

    pc_u_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    pc_v_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    pc_idepth_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
    pc_color_[lvl] = allocAligned<4, float>(wl * hl, ptr_to_delete_);
  }

  // warped buffers
  buf_warped_rx1_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_rx2_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_rx3_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_dx_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_dy_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_residual_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_weight_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
  buf_warped_ref_color_ = allocAligned<4, float>(w_[0] * h_[0], ptr_to_delete_);
}

ScaleOptimizer::~ScaleOptimizer() {
  delete undistorter1_;
  for (float *ptr : ptr_to_delete_)
    delete[] ptr;
  ptr_to_delete_.clear();
}

float ScaleOptimizer::optimize(std::vector<FrameHessian *> frameHessians0,
                               const cv::Mat &img1, CalibHessian *HCalib0,
                               float &scale, int coarsestLvl) {
  assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

  // set frame0
  makeK0(HCalib0);
  setFrame0(frameHessians0);

  // create frame1
  MinimalImageB min_img1((int)img1.cols, (int)img1.rows,
                         (unsigned char *)img1.data);
  ImageAndExposure *undist_img1 =
      undistorter1_->undistort<unsigned char>(&min_img1, 1, 0, 1.0f);
  frame1_ = new FrameHessian();
  FrameShell *shell1 = new FrameShell();
  shell1->camToWorld = SE3();
  shell1->aff_g2l = AffLight(0, 0);
  shell1->timestamp = undist_img1->timestamp;
  shell1->incoming_id = frame0_id_;
  frame1_->shell = shell1;
  frame1_->ab_exposure = undist_img1->exposure_time;
  // only aff b will be used for intensity, assume same for stereo
  frame1_->makeImages(undist_img1->image, HCalib0);

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
    Vec6 resOld =
        calcRes(lvl, scale_current, setting_coarseCutoffTH * levelCutoffRepeat);
    while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
      levelCutoffRepeat *= 2;
      resOld = calcRes(lvl, scale_current,
                       setting_coarseCutoffTH * levelCutoffRepeat);

      if (!setting_debugout_runquiet)
        printf("INCREASING cutoff to %f (ratio is %f)!\n",
               setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
    }

    calcGSSSE(lvl, H, b, scale_current);

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

      Vec6 resNew =
          calcRes(lvl, scale_new, setting_coarseCutoffTH * levelCutoffRepeat);

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
        calcGSSSE(lvl, H, b, scale_new);
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

  delete frame1_;
  delete shell1;
  return last_residuals[0];
}

void ScaleOptimizer::makeK0(CalibHessian *HCalib0) {
  float fx0l = HCalib0->fxl();
  float fy0l = HCalib0->fyl();
  float cx0 = HCalib0->cxl();
  float cy0 = HCalib0->cyl();
  float cx0l = cx0;
  float cy0l = cy0;

  Mat33f K0;
  K0 << fx0l, 0.0, cx0l, 0.0, fy0l, cy0l, 0.0, 0.0, 1.0;
  K0_inv_[0] = K0.inverse();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    fx0l = fx0l * 0.5;
    fy0l = fy0l * 0.5;
    cx0l = (cx0 + 0.5) / ((int)1 << level) - 0.5;
    cy0l = (cy0 + 0.5) / ((int)1 << level) - 0.5;
    K0 << fx0l, 0.0, cx0l, 0.0, fy0l, cy0l, 0.0, 0.0, 1.0;
    K0_inv_[level] = K0.inverse();
  }
}

void ScaleOptimizer::setFrame0(std::vector<FrameHessian *> frameHessians0) {
  assert(frameHessians0.size() > 0);
  frame0_ = frameHessians0.back();
  makeCoarseDepthL0(frameHessians0);
  frame0_id_ = frame0_->shell->id;
}

void ScaleOptimizer::makeCoarseDepthL0(
    std::vector<FrameHessian *> frameHessians0) {
  // make coarse tracking templates for latstRef.
  memset(idepth_[0], 0, sizeof(float) * w_[0] * h_[0]);
  memset(weight_sums_[0], 0, sizeof(float) * w_[0] * h_[0]);

  for (FrameHessian *fh : frameHessians0) {
    for (PointHessian *ph : fh->pointHessians) {
      if (ph->lastResiduals[0].first != 0 &&
          ph->lastResiduals[0].second == ResState::IN) {
        PointFrameResidual *r = ph->lastResiduals[0].first;
        assert(r->efResidual->isActive() && r->target == frame0_);
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
    float *weightSums_l = weight_sums_[lvl];

    float *idepth_lm = idepth_[lvlm1];
    float *weightSums_lm = weight_sums_[lvlm1];

    for (int y = 0; y < hl; y++)
      for (int x = 0; x < wl; x++) {
        int bidx = 2 * x + 2 * y * wlm1;
        idepth_l[x + y * wl] = idepth_lm[bidx] + idepth_lm[bidx + 1] +
                               idepth_lm[bidx + wlm1] +
                               idepth_lm[bidx + wlm1 + 1];

        weightSums_l[x + y * wl] =
            weightSums_lm[bidx] + weightSums_lm[bidx + 1] +
            weightSums_lm[bidx + wlm1] + weightSums_lm[bidx + wlm1 + 1];
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
    Eigen::Vector3f *dIRefl = frame0_->dIp[lvl];

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

void ScaleOptimizer::calcGSSSE(int lvl, float &H_out, float &b_out,
                               float scale) {
  scale_acc_.initialize();

  __m128 fx1l = _mm_set1_ps(fx1_[lvl]);
  __m128 fy1l = _mm_set1_ps(fy1_[lvl]);

  __m128 s = _mm_set1_ps(scale);
  __m128 tx = _mm_set1_ps(tfm_f1_f0_.translation()[0]);
  __m128 ty = _mm_set1_ps(tfm_f1_f0_.translation()[1]);
  __m128 tz = _mm_set1_ps(tfm_f1_f0_.translation()[2]);

  __m128 one = _mm_set1_ps(1);

  int n = buf_warped_n_;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 dxfx = _mm_mul_ps(_mm_load_ps(buf_warped_dx_ + i), fx1l);
    __m128 dyfy = _mm_mul_ps(_mm_load_ps(buf_warped_dy_ + i), fy1l);
    __m128 rx1 = _mm_load_ps(buf_warped_rx1_ + i);
    __m128 rx2 = _mm_load_ps(buf_warped_rx2_ + i);
    __m128 rx3 = _mm_load_ps(buf_warped_rx3_ + i);

    __m128 deno_sqrt = _mm_add_ps(_mm_mul_ps(s, rx3), tz);
    __m128 deno = _mm_div_ps(one, _mm_mul_ps(deno_sqrt, deno_sqrt));

    __m128 xno = _mm_sub_ps(_mm_mul_ps(rx1, tz), _mm_mul_ps(rx3, tx));
    __m128 yno = _mm_sub_ps(_mm_mul_ps(rx2, tz), _mm_mul_ps(rx3, ty));

    scale_acc_.updateSSE_oneed(
        _mm_add_ps(_mm_mul_ps(dxfx, _mm_mul_ps(deno, xno)),
                   _mm_mul_ps(dyfy, _mm_mul_ps(deno, yno))),
        _mm_load_ps(buf_warped_residual_ + i),
        _mm_load_ps(buf_warped_weight_ + i));
  }

  scale_acc_.finish();
  H_out = scale_acc_.hessian_(0, 0) * (1.0f / n);
  b_out = scale_acc_.hessian_(0, 1) * (1.0f / n);
}

Vec6 ScaleOptimizer::calcRes(int lvl, float scale, float cutoffTH) {
  float E = 0;
  int numTermsInE = 0;
  int numTermsInWarped = 0;
  int numSaturated = 0;

  int wl = w_[lvl];
  int hl = h_[lvl];
  Eigen::Vector3f *dINewl = frame1_->dIp[lvl];
  float fx1l = fx1_[lvl];
  float fy1l = fy1_[lvl];
  float cx1l = cx1_[lvl];
  float cy1l = cy1_[lvl];

  Mat33f rot_f1_f0_K0_i =
      (tfm_f1_f0_.rotationMatrix().cast<float>() * K0_inv_[lvl]);
  Vec3f tsl_f1_f0 = (tfm_f1_f0_.translation()).cast<float>();

  float sumSquaredShiftT = 0;
  float sumSquaredShiftRT = 0;
  float sumSquaredShiftNum = 0;

  float maxEnergy =
      2 * setting_huberTH * cutoffTH -
      setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

  MinimalImageB3 *resImage = 0;
  MinimalImageB3 *projImage = 0;
  if (DEBUG_PLOT) {
    resImage = new MinimalImageB3(wl, hl);
    resImage->setConst(Vec3b(255, 255, 255));
    // resImage->setBlack();
    // for(int i=0;i<h_[lvl]*w_[lvl];i++)
    // {
    //   int c = frame0_->dIp[lvl][i][0]*0.9f;
    //   if(c>255) c=255;
    //   resImage->at(i) = Vec3b(c,c,c);
    // }

    projImage = new MinimalImageB3(wl, hl);
    projImage->setBlack();
    for (int i = 0; i < h_[lvl] * w_[lvl]; i++) {
      int c = frame1_->dIp[lvl][i][0] * 0.9f;
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
      Vec3f ptT = scale * K0_inv_[lvl] * Vec3f(x, y, 1) + tsl_f1_f0 * id;
      float uT = ptT[0] / ptT[2];
      float vT = ptT[1] / ptT[2];
      float KuT = fx1l * uT + cx1l;
      float KvT = fy1l * vT + cy1l;

      // translation only (negative)
      Vec3f ptT2 = scale * K0_inv_[lvl] * Vec3f(x, y, 1) - tsl_f1_f0 * id;
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

    // if(DEBUG_PLOT) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
    if (DEBUG_PLOT)
      projImage->setPixel4(Ku, Kv, Vec3b(0, 0, 255));

    if (fabs(residual) > cutoffTH) {
      if (DEBUG_PLOT)
        resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
      E += maxEnergy;
      numTermsInE++;
      numSaturated++;
    } else {
      if (DEBUG_PLOT)
        resImage->setPixel4(
            lpc_u[i], lpc_v[i],
            Vec3b(residual + 128, residual + 128, residual + 128));
      E += hw * residual * residual * (2 - hw);
      numTermsInE++;

      buf_warped_rx1_[numTermsInWarped] = rx[0];
      buf_warped_rx2_[numTermsInWarped] = rx[1];
      buf_warped_rx3_[numTermsInWarped] = rx[2];
      buf_warped_dx_[numTermsInWarped] = hitColor[1];
      buf_warped_dy_[numTermsInWarped] = hitColor[2];
      buf_warped_residual_[numTermsInWarped] = residual;
      buf_warped_weight_[numTermsInWarped] = hw;
      buf_warped_ref_color_[numTermsInWarped] = lpc_color[i];
      numTermsInWarped++;
    }
  }

  while (numTermsInWarped % 4 != 0) {
    buf_warped_rx1_[numTermsInWarped] = 0;
    buf_warped_rx2_[numTermsInWarped] = 0;
    buf_warped_rx3_[numTermsInWarped] = 0;
    buf_warped_dx_[numTermsInWarped] = 0;
    buf_warped_dy_[numTermsInWarped] = 0;
    buf_warped_residual_[numTermsInWarped] = 0;
    buf_warped_weight_[numTermsInWarped] = 0;
    buf_warped_ref_color_[numTermsInWarped] = 0;
    numTermsInWarped++;
  }
  buf_warped_n_ = numTermsInWarped;

  if (DEBUG_PLOT) {
    IOWrap::displayImage("Scale RES", resImage, false);
    IOWrap::displayImage("Proj", projImage, false);
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

} // namespace dso
