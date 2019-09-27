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

namespace dso {

template <int b, typename T>
T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
  const int padT = 1 + ((1 << b) / sizeof(T));
  T *ptr = new T[size + padT];
  rawPtrVec.push_back(ptr);
  T *alignedPtr = (T *)((((uintptr_t)(ptr + padT)) >> b) << b);
  return alignedPtr;
}

PoseEstimator::PoseEstimator(int ww, int hh) : lastRef_aff_g2l(0, 0) {
  // warped buffers
  buf_warped_idepth = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_u = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_v = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_dx = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_dy = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_residual = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_weight = allocAligned<4, float>(ww * hh, ptrToDelete);
  buf_warped_refColor = allocAligned<4, float>(ww * hh, ptrToDelete);

  newFrame = 0;
  debugPlot = false;
  debugPrint = false;
  w[0] = h[0] = 0;
}

PoseEstimator::~PoseEstimator() {
  for (float *ptr : ptrToDelete)
    delete[] ptr;
  ptrToDelete.clear();
}

void PoseEstimator::makeK(CalibHessian *HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }
}

void PoseEstimator::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out,
                              const SE3 &refToNew, AffLight aff_g2l) {
  acc.initialize();

  __m128 fxl = _mm_set1_ps(fx[lvl]);
  __m128 fyl = _mm_set1_ps(fy[lvl]);
  __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
  __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(
      lastRef_ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
      aff_g2l)[0]));

  __m128 one = _mm_set1_ps(1);
  __m128 minusOne = _mm_set1_ps(-1);
  __m128 zero = _mm_set1_ps(0);

  int n = buf_warped_n;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);
    __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);
    __m128 u = _mm_load_ps(buf_warped_u + i);
    __m128 v = _mm_load_ps(buf_warped_v + i);
    __m128 id = _mm_load_ps(buf_warped_idepth + i);

    acc.updateSSE_eighted(
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
        _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + i))),
        minusOne, _mm_load_ps(buf_warped_residual + i),
        _mm_load_ps(buf_warped_weight + i));
  }

  acc.finish();
  H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
  b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

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

  Mat33f R = refToNew.rotationMatrix().cast<float>();
  Vec3f t = refToNew.translation().cast<float>();
  Vec2f affLL =
      AffLight::fromToVecExposure(lastRef_ab_exposure, newFrame->ab_exposure,
                                  lastRef_aff_g2l, aff_g2l)
          .cast<float>();

  float sumSquaredShiftT = 0;
  float sumSquaredShiftRT = 0;
  float sumSquaredShiftNum = 0;

  float maxEnergy =
      2 * setting_huberTH * cutoffTH -
      setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

  MinimalImageB3 *resImage = 0;
  if (debugPlot || lvl == 0) {
    resImage = new MinimalImageB3(wl, hl);
    resImage->setBlack();
    for (int i = 0; i < h[lvl] * w[lvl]; i++) {
      int c = newFrame->dIp[lvl][i][0] * 0.9f;
      if (c > 255)
        c = 255;
      resImage->at(i) = Vec3b(c, c, c);
    }
  }

  for (int i = 0; i < pointxyzi.rows(); i++) {
    float x = pointxyzi(i, 0);
    float y = pointxyzi(i, 1);
    float z = pointxyzi(i, 2);
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

    float refColor = pointxyzi(i, 3);
    Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);

    if (!std::isfinite((float)hitColor[0]))
      continue;
    float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
    float hw =
        fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    if (fabs(residual) > cutoffTH) {
      if (debugPlot || lvl == 0)
        resImage->setPixel4(Ku, Kv, Vec3b(0, 0, 255));
      E += maxEnergy;
      numTermsInE++;
      numSaturated++;
    } else {
      if (debugPlot || lvl == 0)
        resImage->setPixel4(Ku, Kv, Vec3b(0, residual + 128, 0));

      E += hw * residual * residual * (2 - hw);
      numTermsInE++;

      buf_warped_idepth[numTermsInWarped] = new_idepth;
      buf_warped_u[numTermsInWarped] = u;
      buf_warped_v[numTermsInWarped] = v;
      buf_warped_dx[numTermsInWarped] = hitColor[1];
      buf_warped_dy[numTermsInWarped] = hitColor[2];
      buf_warped_residual[numTermsInWarped] = residual;
      buf_warped_weight[numTermsInWarped] = hw;
      buf_warped_refColor[numTermsInWarped] = refColor;
      numTermsInWarped++;
    }
  }

  while (numTermsInWarped % 4 != 0) {
    buf_warped_idepth[numTermsInWarped] = 0;
    buf_warped_u[numTermsInWarped] = 0;
    buf_warped_v[numTermsInWarped] = 0;
    buf_warped_dx[numTermsInWarped] = 0;
    buf_warped_dy[numTermsInWarped] = 0;
    buf_warped_residual[numTermsInWarped] = 0;
    buf_warped_weight[numTermsInWarped] = 0;
    buf_warped_refColor[numTermsInWarped] = 0;
    numTermsInWarped++;
  }
  buf_warped_n = numTermsInWarped;

  if (debugPlot) {
    IOWrap::displayImage("RES", resImage, false);
    IOWrap::waitKey(0);
    delete resImage;
  } else if (lvl == 0) {
    IOWrap::displayImage("RES", resImage, false);
    IOWrap::waitKey(1);
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

void PoseEstimator::setPointsRef(
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts) {
  pointxyzi = Eigen::MatrixXf(pts.size(), 4);

  for (int i = 0; i < pts.size(); i++) {
    pointxyzi(i, 0) = pts[i].first(0);
    pointxyzi(i, 1) = pts[i].first(1);
    pointxyzi(i, 2) = pts[i].first(2);
    pointxyzi(i, 3) = pts[i].second;
  }
}

void PoseEstimator::estimate(
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts,
    const std::pair<AffLight, float> &affLightExposure,
    FrameHessian *newFrameHessian, CalibHessian *HCalib,
    Eigen::Matrix<double, 4, 4> &lastToNew_out, Mat66 &H_pose,
    Vec5 &lastResiduals, int lastInners[5], int coarsestLvl) {
  int maxIterations[] = {10, 20, 50, 50, 50};
  float lambdaExtrapolationLimit = 0.001;

  assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

  makeK(HCalib);
  setPointsRef(pts);

  lastResiduals.setConstant(NAN);

  newFrame = newFrameHessian;
  AffLight aff_g2l_current = newFrame->aff_g2l();

  Sophus::SE3 refToNew_current(lastToNew_out.block<3, 3>(0, 0),
                               lastToNew_out.block<3, 1>(0, 3));

  lastRef_aff_g2l = affLightExposure.first;
  lastRef_ab_exposure = affLightExposure.second;

  bool haveRepeated = false;

  Mat88 H_current;
  for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
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

    if (debugPrint) {
      Vec2f relAff = AffLight::fromToVecExposure(
                         lastRef_ab_exposure, newFrame->ab_exposure,
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

      Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new,
                            setting_coarseCutoffTH * levelCutoffRepeat);

      bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

      if (debugPrint) {
        Vec2f relAff = AffLight::fromToVecExposure(lastRef_ab_exposure,
                                                   newFrame->ab_exposure,
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
        calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
        H_current = H;
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
        if (debugPrint)
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
      printf("REPEAT LEVEL!\n");
    }
  }

  // set!
  lastToNew_out = refToNew_current.matrix();
  H_pose = H_current.topLeftCorner<6, 6>();
}

} // namespace dso