/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "ScaleOptimizer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "IOWrapper/ImageRW.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <algorithm>

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

ScaleOptimizer::ScaleOptimizer(int w_, int h_, const Eigen::Matrix3f &K,
                               const SE3 &T_stereo_, float accept_th_) {
  // make coarse tracking templates.
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = w_ >> lvl;
    int hl = h_ >> lvl;

    idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
    weightSums[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
    weightSums_bak[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

    pc_u[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
    pc_v[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
    pc_idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
    pc_color[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
  }

  // warped buffers
  buf_warped_rx1 = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_rx2 = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_rx3 = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_dx = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_dy = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_residual = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_weight = allocAligned<4, float>(w_ * h_, ptrToDelete);
  buf_warped_refColor = allocAligned<4, float>(w_ * h_, ptrToDelete);

  newFrame = 0;
  lastRef = 0;
  debugPlot = debugPrint = true;
  w[0] = h[0] = 0;
  refFrameID = -1;

  w[0] = w_;
  h[0] = h_;

  fx[0] = K(0, 0);
  fy[0] = K(1, 1);
  cx[0] = K(0, 2);
  cy[0] = K(1, 2);

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  accept_th = accept_th_;
  T_stereo = T_stereo_;
}

void ScaleOptimizer::makeK(CalibHessian *HCalib) {
  float fxl_orig = HCalib->fxl();
  float fyl_orig = HCalib->fyl();
  float cx0_orig = HCalib->cxl();
  float cy0_orig = HCalib->cyl();
  float cxl_orig = cx0_orig;
  float cyl_orig = cy0_orig;

  Mat33f K_orig_l;
  K_orig_l << fxl_orig, 0.0, cxl_orig, 0.0, fyl_orig, cyl_orig, 0.0, 0.0, 1.0;
  Ki_orig[0] = K_orig_l.inverse();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    fxl_orig = fxl_orig * 0.5;
    fyl_orig = fyl_orig * 0.5;
    cxl_orig = (cx0_orig + 0.5) / ((int)1 << level) - 0.5;
    cyl_orig = (cy0_orig + 0.5) / ((int)1 << level) - 0.5;
    K_orig_l << fxl_orig, 0.0, cxl_orig, 0.0, fyl_orig, cyl_orig, 0.0, 0.0, 1.0;
    Ki_orig[level] = K_orig_l.inverse();
  }
}

ScaleOptimizer::~ScaleOptimizer() {
  for (float *ptr : ptrToDelete)
    delete[] ptr;
  ptrToDelete.clear();
}

bool ScaleOptimizer::optimize(FrameHessian *newFrameHessian, float &scale,
                              int coarsestLvl, Vec5 minResForAbort,
                              IOWrap::Output3DWrapper *wrap) {
  debugPlot = setting_render_displayCoarseTrackingFull;
  // debugPlot = true;
  debugPrint = false;

  assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

  lastResiduals.setConstant(NAN);
  lastFlowIndicators.setConstant(1000);

  newFrame = newFrameHessian;
  int maxIterations[] = {10, 20, 50, 50, 50};
  float lambdaExtrapolationLimit = 0.001;

  float scale_current = scale;

  bool haveRepeated = false;

  for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
    float H;
    float b;
    float levelCutoffRepeat = 1;
    Vec6 resOld = calcRes(lvl, T_stereo, scale_current,
                          setting_coarseCutoffTH * levelCutoffRepeat);
    while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
      levelCutoffRepeat *= 2;
      resOld = calcRes(lvl, T_stereo, scale_current,
                       setting_coarseCutoffTH * levelCutoffRepeat);

      if (!setting_debugout_runquiet)
        printf("INCREASING cutoff to %f (ratio is %f)!\n",
               setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
    }

    calcGSSSE(lvl, H, b, T_stereo, scale_current);

    float lambda = 0.01;

    if (debugPrint) {
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

      Vec6 resNew = calcRes(lvl, T_stereo, scale_new,
                            setting_coarseCutoffTH * levelCutoffRepeat);

      bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

      if (debugPrint) {
        printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (inc = "
               "%f)! \t",
               lvl, iteration, lambda, extrapFac,
               (accept ? "ACCEPT" : "REJECT"), resOld[0] / resOld[1],
               resNew[0] / resNew[1], (int)resOld[1], (int)resNew[1], inc);
        std::cout << " New scale " << scale_new << std::endl;
      }

      if (accept) {
        calcGSSSE(lvl, H, b, T_stereo, scale_new);
        resOld = resNew;
        scale_current = scale_new;
        lambda *= 0.5;
      } else {
        lambda *= 4;
        if (lambda < lambdaExtrapolationLimit)
          lambda = lambdaExtrapolationLimit;
      }

      if (!(inc > 1e-3)) {
        if (debugPrint)
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
      printf("SCALE OPT. REPEAT LEVEL!\n");
    }
  }

  printf("Final scale: %f Final res: %f\n", scale_current, lastResiduals[0]);

  if (scale_current < 0 || !(lastResiduals[0] < accept_th)) {
    printf("Scale opt. rejected: coarsestLvl=%d, scale changed: %f -> %f\n",
           coarsestLvl, scale, scale_current);
    return false;
  }

  // set!
  scale = scale_current;

  return true;
}

void ScaleOptimizer::setCoarseTrackingRef(
    std::vector<FrameHessian *> frameHessians) {
  assert(frameHessians.size() > 0);
  lastRef = frameHessians.back();
  makeCoarseDepthL0(frameHessians);

  refFrameID = lastRef->shell->id;

  firstCoarseRMSE = -1;
}

void ScaleOptimizer::makeCoarseDepthL0(
    std::vector<FrameHessian *> frameHessians) {
  // make coarse tracking templates for latstRef.
  memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
  memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

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

        idepth[0][u + w[0] * v] += new_idepth * weight;
        weightSums[0][u + w[0] * v] += weight;
      }
    }
  }

  for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
    int lvlm1 = lvl - 1;
    int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

    float *idepth_l = idepth[lvl];
    float *weightSums_l = weightSums[lvl];

    float *idepth_lm = idepth[lvlm1];
    float *weightSums_lm = weightSums[lvlm1];

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

  // dilate idepth by 1.
  for (int lvl = 0; lvl < 2; lvl++) {
    int numIts = 1;

    for (int it = 0; it < numIts; it++) {
      int wh = w[lvl] * h[lvl] - w[lvl];
      int wl = w[lvl];
      float *weightSumsl = weightSums[lvl];
      float *weightSumsl_bak = weightSums_bak[lvl];
      memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
      float *idepthl =
          idepth[lvl]; // dotnt need to make a temp copy of depth, since I only
                       // read values with weightSumsl>0, and write ones with
                       // weightSumsl<=0.
      for (int i = w[lvl]; i < wh; i++) {
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

  // dilate idepth by 1 (2 on lower levels).
  for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
    int wh = w[lvl] * h[lvl] - w[lvl];
    int wl = w[lvl];
    float *weightSumsl = weightSums[lvl];
    float *weightSumsl_bak = weightSums_bak[lvl];
    memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
    float *idepthl =
        idepth[lvl]; // dotnt need to make a temp copy of depth, since I only
                     // read values with weightSumsl>0, and write ones with
                     // weightSumsl<=0.
    for (int i = w[lvl]; i < wh; i++) {
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
    float *weightSumsl = weightSums[lvl];
    float *idepthl = idepth[lvl];
    Eigen::Vector3f *dIRefl = lastRef->dIp[lvl];

    int wl = w[lvl], hl = h[lvl];

    int lpc_n = 0;
    float *lpc_u = pc_u[lvl];
    float *lpc_v = pc_v[lvl];
    float *lpc_idepth = pc_idepth[lvl];
    float *lpc_color = pc_color[lvl];

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

    pc_n[lvl] = lpc_n;
  }
}

/*
void ScaleOptimizer::calcGSSSE(int lvl, float &H_out, float &b_out, const SE3
&T_stereo, float scale)
{
  H_out=0.0;
  b_out=0.0;

  float fxl = fx[lvl];
  float fyl = fy[lvl];

  float s = scale;
  float tx = T_stereo.translation()[0];
  float ty = T_stereo.translation()[1];
  float tz= T_stereo.translation()[2];

  int n = buf_warped_n;
  assert(n%4==0);
  for(int i=0;i<n;i+=4)
  {
    float dxfx = buf_warped_dx[i]*fxl;
    float dyfy = buf_warped_dy[i]*fyl;
    float rx1 = buf_warped_rx1[i];
    float rx2 = buf_warped_rx2[i];
    float rx3 = buf_warped_rx3[i];

    float deno_sqrt = s*rx3 + tz;
    float deno = 1/deno_sqrt/deno_sqrt;

    float xno = rx1*tz - rx3*tx;
    float yno = rx2*tz - rx3*ty;

    float J = dxfx*deno*xno+dyfy*deno*yno;

    H_out += J*buf_warped_weight[i]*J;
    b_out += J*buf_warped_weight[i]*buf_warped_residual[i];

  }
}
*/

void ScaleOptimizer::calcGSSSE(int lvl, float &H_out, float &b_out,
                               const SE3 &T_stereo, float scale) {
  acc.initialize();

  __m128 fxl = _mm_set1_ps(fx[lvl]);
  __m128 fyl = _mm_set1_ps(fy[lvl]);

  __m128 s = _mm_set1_ps(scale);
  __m128 tx = _mm_set1_ps(T_stereo.translation()[0]);
  __m128 ty = _mm_set1_ps(T_stereo.translation()[1]);
  __m128 tz = _mm_set1_ps(T_stereo.translation()[2]);

  __m128 one = _mm_set1_ps(1);

  int n = buf_warped_n;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 dxfx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);
    __m128 dyfy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);
    __m128 rx1 = _mm_load_ps(buf_warped_rx1 + i);
    __m128 rx2 = _mm_load_ps(buf_warped_rx2 + i);
    __m128 rx3 = _mm_load_ps(buf_warped_rx3 + i);

    __m128 deno_sqrt = _mm_add_ps(_mm_mul_ps(s, rx3), tz);
    __m128 deno = _mm_div_ps(one, _mm_mul_ps(deno_sqrt, deno_sqrt));

    __m128 xno = _mm_sub_ps(_mm_mul_ps(rx1, tz), _mm_mul_ps(rx3, tx));
    __m128 yno = _mm_sub_ps(_mm_mul_ps(rx2, tz), _mm_mul_ps(rx3, ty));

    acc.updateSSE_oneed(_mm_add_ps(_mm_mul_ps(dxfx, _mm_mul_ps(deno, xno)),
                                   _mm_mul_ps(dyfy, _mm_mul_ps(deno, yno))),
                        _mm_load_ps(buf_warped_residual + i),
                        _mm_load_ps(buf_warped_weight + i));
  }

  acc.finish();
  H_out = acc.H(0, 0) * (1.0f / n);
  b_out = acc.H(0, 1) * (1.0f / n);
}

Vec6 ScaleOptimizer::calcRes(int lvl, const SE3 &T_stereo, float scale,
                             float cutoffTH) {
  float E = 0;
  int numTermsInE = 0;
  int numTermsInWarped = 0;
  int numSaturated = 0;

  int wl = w[lvl];
  int hl = h[lvl];
  Eigen::Vector3f *dINewl = newFrame->dIp[lvl];
  float fxl = fx[lvl];
  float fyl = fy[lvl];
  float cxl = cx[lvl];
  float cyl = cy[lvl];

  Mat33f RKi_orig = (T_stereo.rotationMatrix().cast<float>() * Ki_orig[lvl]);
  Vec3f t = (T_stereo.translation()).cast<float>();

  float sumSquaredShiftT = 0;
  float sumSquaredShiftRT = 0;
  float sumSquaredShiftNum = 0;

  float maxEnergy =
      2 * setting_huberTH * cutoffTH -
      setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

  MinimalImageB3 *resImage = 0;
  MinimalImageB3 *projImage = 0;
  if (debugPlot) {
    resImage = new MinimalImageB3(wl, hl);
    resImage->setConst(Vec3b(255, 255, 255));
    // resImage->setBlack();
    // for(int i=0;i<h[lvl]*w[lvl];i++)
    // {
    //   int c = lastRef->dIp[lvl][i][0]*0.9f;
    //   if(c>255) c=255;
    //   resImage->at(i) = Vec3b(c,c,c);
    // }

    projImage = new MinimalImageB3(wl, hl);
    projImage->setBlack();
    for (int i = 0; i < h[lvl] * w[lvl]; i++) {
      int c = newFrame->dIp[lvl][i][0] * 0.9f;
      if (c > 255)
        c = 255;
      projImage->at(i) = Vec3b(c, c, c);
    }
  }

  int nl = pc_n[lvl];
  float *lpc_u = pc_u[lvl];
  float *lpc_v = pc_v[lvl];
  float *lpc_idepth = pc_idepth[lvl];
  float *lpc_color = pc_color[lvl];

  for (int i = 0; i < nl; i++) {
    float id = lpc_idepth[i];
    float x = lpc_u[i];
    float y = lpc_v[i];

    Vec3f pt = scale * RKi_orig * Vec3f(x, y, 1) + t * id;
    float u = pt[0] / pt[2];
    float v = pt[1] / pt[2];
    float Ku = fxl * u + cxl;
    float Kv = fyl * v + cyl;
    float new_idepth = id / pt[2];

    Vec3f rx = RKi_orig * Vec3f(x, y, 1) / id;

    if (lvl == 0 && i % 32 == 0) {
      // translation only (positive)
      Vec3f ptT = scale * Ki_orig[lvl] * Vec3f(x, y, 1) + t * id;
      float uT = ptT[0] / ptT[2];
      float vT = ptT[1] / ptT[2];
      float KuT = fxl * uT + cxl;
      float KvT = fyl * vT + cyl;

      // translation only (negative)
      Vec3f ptT2 = scale * Ki_orig[lvl] * Vec3f(x, y, 1) - t * id;
      float uT2 = ptT2[0] / ptT2[2];
      float vT2 = ptT2[1] / ptT2[2];
      float KuT2 = fxl * uT2 + cxl;
      float KvT2 = fyl * vT2 + cyl;

      // translation and rotation (negative)
      Vec3f pt3 = scale * RKi_orig * Vec3f(x, y, 1) - t * id;
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
    float residual = hitColor[0] - refColor;
    float hw =
        fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    // if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
    if (debugPlot)
      projImage->setPixel4(Ku, Kv, Vec3b(0, 0, 255));

    if (fabs(residual) > cutoffTH) {
      if (debugPlot)
        resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
      E += maxEnergy;
      numTermsInE++;
      numSaturated++;
    } else {
      if (debugPlot)
        resImage->setPixel4(
            lpc_u[i], lpc_v[i],
            Vec3b(residual + 128, residual + 128, residual + 128));
      E += hw * residual * residual * (2 - hw);
      numTermsInE++;

      buf_warped_rx1[numTermsInWarped] = rx[0];
      buf_warped_rx2[numTermsInWarped] = rx[1];
      buf_warped_rx3[numTermsInWarped] = rx[2];
      buf_warped_dx[numTermsInWarped] = hitColor[1];
      buf_warped_dy[numTermsInWarped] = hitColor[2];
      buf_warped_residual[numTermsInWarped] = residual;
      buf_warped_weight[numTermsInWarped] = hw;
      buf_warped_refColor[numTermsInWarped] = lpc_color[i];
      numTermsInWarped++;
    }
  }

  while (numTermsInWarped % 4 != 0) {
    buf_warped_rx1[numTermsInWarped] = 0;
    buf_warped_rx2[numTermsInWarped] = 0;
    buf_warped_rx3[numTermsInWarped] = 0;
    buf_warped_dx[numTermsInWarped] = 0;
    buf_warped_dy[numTermsInWarped] = 0;
    buf_warped_residual[numTermsInWarped] = 0;
    buf_warped_weight[numTermsInWarped] = 0;
    buf_warped_refColor[numTermsInWarped] = 0;
    numTermsInWarped++;
  }
  buf_warped_n = numTermsInWarped;

  if (debugPlot) {
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
