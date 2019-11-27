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

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FrontEnd.h"
#include "timing.h"

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/CoarseTracker.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>
#include <fstream>
#include <queue>

namespace dso {
int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

FrontEnd::FrontEnd(int prev_kf_size) {
  selection_map_ = new float[wG[0] * hG[0]];

  coarse_distance_map_ = new CoarseDistanceMap(wG[0], hG[0]);
  coarse_tracker_ = new CoarseTracker(wG[0], hG[0]);
  coarse_tracker_for_new_kf_ = new CoarseTracker(wG[0], hG[0]);
  coarse_initializer_ = new CoarseInitializer(wG[0], hG[0]);
  pixel_selector_ = new PixelSelector(wG[0], hG[0]);

  last_coarse_rmse_.setConstant(100);

  current_min_act_dist_ = 2;
  initialized_ = false;

  ef_ = new EnergyFunctional();
  ef_->red = &this->tread_reduce_;

  is_lost_ = false;
  init_failed_ = false;

  last_ref_stop_id_ = 0;

  min_id_jet_vis_debug_ = -1;
  max_id_jet_vis_debug_ = -1;
  min_id_jet_vis_tracker_ = -1;
  max_id_jet_vis_tracker_ = -1;

  scale_optimizer_ = 0;

  cur_pose_ = SE3();
  prev_kf_size_ = prev_kf_size;
}

FrontEnd::~FrontEnd() {
  delete[] selection_map_;

  for (FrameShell *s : all_frame_history_)
    delete s;

  delete coarse_distance_map_;
  delete coarse_tracker_;
  delete coarse_tracker_for_new_kf_;
  delete coarse_initializer_;
  delete pixel_selector_;
  delete ef_;
}

void FrontEnd::setGammaFunction(float *BInv) {
  if (BInv == 0)
    return;

  // copy BInv.
  memcpy(h_calib_.Binv, BInv, sizeof(float) * 256);

  // invert.
  for (int i = 1; i < 255; i++) {
    // find val, such that Binv[val] = i.
    // I dont care about speed for this, so do it the stupid way.

    for (int s = 1; s < 255; s++) {
      if (BInv[s] <= i && BInv[s + 1] >= i) {
        h_calib_.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
        break;
      }
    }
  }
  h_calib_.B[0] = 0;
  h_calib_.B[255] = 255;
}

Vec4 FrontEnd::trackNewCoarse(FrameHessian *fh) {

  assert(all_frame_history_.size() > 0);
  // set pose initialization.

  for (IOWrap::Output3DWrapper *ow : output_wrapper_)
    ow->pushLiveFrame(fh);

  FrameHessian *lastF = coarse_tracker_->lastRef;

  AffLight aff_last_2_l = AffLight(0, 0);

  std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
  if (all_frame_history_.size() == 2)
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
      lastF_2_fh_tries.push_back(SE3());
  else {
    FrameShell *slast = all_frame_history_[all_frame_history_.size() - 2];
    FrameShell *sprelast = all_frame_history_[all_frame_history_.size() - 3];
    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;
    { // lock on global pose consistency!
      boost::unique_lock<boost::mutex> crlock(shell_pose_mutex_);
      slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
      lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
      aff_last_2_l = slast->aff_g2l;
    }
    SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

    // get last delta-movement.
    auto lastF_2_fh_const = fh_2_slast.inverse() * lastF_2_slast;
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() *
                               lastF_2_slast); // assume constant motion.
    lastF_2_fh_tries.push_back(
        fh_2_slast.inverse() * fh_2_slast.inverse() *
        lastF_2_slast); // assume double motion (frame skipped)
    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() *
                               lastF_2_slast); // assume half motion.
    lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
    lastF_2_fh_tries.push_back(SE3());         // assume zero motion FROM KF.

    // just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which
    // is super fast anyway. also, if tracking rails here we loose, so we
    // really, really want to avoid that.
    std::vector<std::vector<float>> rot_signs = {
        {1, 0, 0},   {0, 1, 0},   {0, 0, 1},   {-1, 0, 0},   {0, -1, 0},
        {0, 0, -1},  {1, 1, 0},   {0, 1, 1},   {1, 0, 1},    {-1, 1, 0},
        {0, -1, 1},  {-1, 0, 1},  {1, -1, 0},  {0, 1, -1},   {1, 0, -1},
        {-1, -1, 0}, {0, -1, -1}, {-1, 0, -1}, {-1, -1, -1}, {-1, -1, 1},
        {-1, 1, -1}, {-1, 1, 1},  {1, -1, -1}, {1, -1, 1},   {1, 1, -1},
        {1, 1, 1}};
    for (float rot_delta = 0.02; rot_delta < 0.05; rot_delta += 0.01) {
      for (auto &rs : rot_signs) {
        lastF_2_fh_tries.push_back(
            lastF_2_fh_const *
            SE3(Sophus::Quaterniond(1, rs[0] * rot_delta, rs[1] * rot_delta,
                                    rs[2] * rot_delta),
                Vec3(0, 0, 0)));
      }
    }

    if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
      lastF_2_fh_tries.clear();
      lastF_2_fh_tries.push_back(SE3());
    }
  }

  Vec3 flowVecs = Vec3(100, 100, 100);
  SE3 lastF_2_fh = SE3();
  AffLight aff_g2l = AffLight(0, 0);

  // as long as maxResForImmediateAccept is not reached, I'll continue through
  // the options. I'll keep track of the so-far best achieved residual for each
  // level in achievedRes. If on a coarse level, tracking is WORSE than
  // achievedRes, we will not continue to save time.

  Vec5 achievedRes = Vec5::Constant(NAN);
  bool haveOneGood = false;
  int tryIterations = 0;
  for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
    AffLight aff_g2l_this = aff_last_2_l;
    SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
    bool trackingIsGood = coarse_tracker_->trackNewestCoarse(
        fh, lastF_2_fh_this, aff_g2l_this, pyrLevelsUsed - 1,
        achievedRes); // in each level has to be at least as good as the last
                      // try.
    tryIterations++;

    // if (i != 0) {
    //   printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f
    //   "
    //          "%f): %f %f %f %f %f -> %f %f %f %f %f \n",
    //          i, i, pyrLevelsUsed - 1, aff_g2l_this.a, aff_g2l_this.b,
    //          achievedRes[0], achievedRes[1], achievedRes[2], achievedRes[3],
    //          achievedRes[4], coarse_tracker_->lastResiduals[0],
    //          coarse_tracker_->lastResiduals[1],
    //          coarse_tracker_->lastResiduals[2],
    //          coarse_tracker_->lastResiduals[3],
    //          coarse_tracker_->lastResiduals[4]);
    // }

    // do we have a new winner?
    if (trackingIsGood &&
        std::isfinite((float)coarse_tracker_->lastResiduals[0]) &&
        !(coarse_tracker_->lastResiduals[0] >= achievedRes[0])) {
      // printf("take over. minRes %f -> %f!\n", achievedRes[0],
      // coarse_tracker_->lastResiduals[0]);
      flowVecs = coarse_tracker_->lastFlowIndicators;
      aff_g2l = aff_g2l_this;
      lastF_2_fh = lastF_2_fh_this;
      haveOneGood = true;
    }

    // take over achieved res (always).
    if (haveOneGood) {
      for (int i = 0; i < 5; i++) {
        if (!std::isfinite((float)achievedRes[i]) ||
            achievedRes[i] >
                coarse_tracker_->lastResiduals[i]) // take over if achievedRes
                                                   // is either bigger or NAN.
          achievedRes[i] = coarse_tracker_->lastResiduals[i];
      }
    }

    if (haveOneGood &&
        achievedRes[0] < last_coarse_rmse_[0] * setting_reTrackThreshold)
      break;
  }

  if (!haveOneGood) {
    printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope "
           "we may somehow recover.\n");
    flowVecs = Vec3(0, 0, 0);
    aff_g2l = aff_last_2_l;
    lastF_2_fh = lastF_2_fh_tries[0];
  }

  last_coarse_rmse_ = achievedRes;

  // no lock required, as fh is not used anywhere yet.
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef = lastF->shell;
  fh->shell->aff_g2l = aff_g2l;
  fh->shell->camToWorld =
      fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

  if (coarse_tracker_->firstCoarseRMSE < 0)
    coarse_tracker_->firstCoarseRMSE = achievedRes[0];

  if (!setting_debugout_runquiet)
    printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a,
           aff_g2l.b, fh->ab_exposure, achievedRes[0]);

  return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FrontEnd::traceNewCoarse(FrameHessian *fh) {
  boost::unique_lock<boost::mutex> lock(map_mutex_);

  int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
      trace_skip = 0, trace_badcondition = 0, trace_uninitialized_ = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0) = h_calib_.fxl();
  K(1, 1) = h_calib_.fyl();
  K(0, 2) = h_calib_.cxl();
  K(1, 2) = h_calib_.cyl();

  for (FrameHessian *host : frame_hessians_) // go through all active frames
  {

    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    Vec3f Kt = K * hostToNew.translation().cast<float>();

    Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                            host->aff_g2l(), fh->aff_g2l())
                    .cast<float>();

    for (ImmaturePoint *ph : host->immaturePoints) {
      ph->traceOn(fh, KRKi, Kt, aff, &h_calib_, false);

      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
        trace_good++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
        trace_badcondition++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
        trace_oob++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
        trace_out++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
        trace_skip++;
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
        trace_uninitialized_++;
      trace_total++;
    }
  }
  //  printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d
  //  (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%)
  //  uninit.\n",
  //      trace_total,
  //      trace_good, 100*trace_good/(float)trace_total,
  //      trace_skip, 100*trace_skip/(float)trace_total,
  //      trace_badcondition, 100*trace_badcondition/(float)trace_total,
  //      trace_oob, 100*trace_oob/(float)trace_total,
  //      trace_out, 100*trace_out/(float)trace_total,
  //      trace_uninitialized_, 100*trace_uninitialized_/(float)trace_total);
}

void FrontEnd::activatePointsMT_Reductor(
    std::vector<PointHessian *> *optimized,
    std::vector<ImmaturePoint *> *toOptimize, int min, int max, Vec10 *stats,
    int tid) {
  ImmaturePointTemporaryResidual *tr =
      new ImmaturePointTemporaryResidual[frame_hessians_.size()];
  for (int k = min; k < max; k++) {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
  }
  delete[] tr;
}

void FrontEnd::activatePointsMT() {

  if (ef_->nPoints < setting_desiredPointDensity * 0.66)
    current_min_act_dist_ -= 0.8;
  if (ef_->nPoints < setting_desiredPointDensity * 0.8)
    current_min_act_dist_ -= 0.5;
  else if (ef_->nPoints < setting_desiredPointDensity * 0.9)
    current_min_act_dist_ -= 0.2;
  else if (ef_->nPoints < setting_desiredPointDensity)
    current_min_act_dist_ -= 0.1;

  if (ef_->nPoints > setting_desiredPointDensity * 1.5)
    current_min_act_dist_ += 0.8;
  if (ef_->nPoints > setting_desiredPointDensity * 1.3)
    current_min_act_dist_ += 0.5;
  if (ef_->nPoints > setting_desiredPointDensity * 1.15)
    current_min_act_dist_ += 0.2;
  if (ef_->nPoints > setting_desiredPointDensity)
    current_min_act_dist_ += 0.1;

  if (current_min_act_dist_ < 0)
    current_min_act_dist_ = 0;
  if (current_min_act_dist_ > 4)
    current_min_act_dist_ = 4;

  if (!setting_debugout_runquiet)
    printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
           current_min_act_dist_, (int)(setting_desiredPointDensity),
           ef_->nPoints);

  FrameHessian *newestHs = frame_hessians_.back();

  // make dist map.
  coarse_distance_map_->makeK(&h_calib_);
  coarse_distance_map_->makeDistanceMap(frame_hessians_, newestHs);

  // coarse_tracker_->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint *> toOptimize;
  toOptimize.reserve(20000);

  for (FrameHessian *host : frame_hessians_) // go through all active frames
  {
    if (host == newestHs)
      continue;

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi =
        (coarse_distance_map_->K[1] * fhToNew.rotationMatrix().cast<float>() *
         coarse_distance_map_->Ki[0]);
    Vec3f Kt =
        (coarse_distance_map_->K[1] * fhToNew.translation().cast<float>());

    for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
      ImmaturePoint *ph = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;

      // delete points that have never been traced successfully, or that are
      // outlier on the last trace.
      if (!std::isfinite(ph->idepth_max) ||
          ph->lastTraceStatus == IPS_OUTLIER) {
        //        immature_invalid_deleted++;
        // remove point.
        delete ph;
        host->immaturePoints[i] = 0;
        continue;
      }

      // can activate only if this is true.
      bool canActivate = (ph->lastTraceStatus == IPS_GOOD ||
                          ph->lastTraceStatus == IPS_SKIPPED ||
                          ph->lastTraceStatus == IPS_BADCONDITION ||
                          ph->lastTraceStatus == IPS_OOB) &&
                         ph->lastTracePixelInterval < 8 &&
                         ph->quality > setting_minTraceQuality &&
                         (ph->idepth_max + ph->idepth_min) > 0;

      // if I cannot activate the point, skip it. Maybe also delete it.
      if (!canActivate) {
        // if point will be out afterwards, delete it instead.
        if (ph->host->flaggedForMarginalization ||
            ph->lastTraceStatus == IPS_OOB) {
          //          immature_notReady_deleted++;
          delete ph;
          host->immaturePoints[i] = 0;
        }
        //        immature_notReady_skipped++;
        continue;
      }

      // see if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) +
                  Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;

      if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {

        float dist = coarse_distance_map_->fwdWarpedIDDistFinal[u + wG[1] * v] +
                     (ptp[0] - floorf((float)(ptp[0])));

        if (dist >= current_min_act_dist_ * ph->my_type) {
          coarse_distance_map_->addIntoDistFinal(u, v);
          toOptimize.push_back(ph);
        }
      } else {
        delete ph;
        host->immaturePoints[i] = 0;
      }
    }
  }

  //  printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip
  //  %d)\n",
  //      (int)toOptimize.size(), immature_deleted, immature_notReady,
  //      immature_needMarg, immature_want, immature_margskip);

  std::vector<PointHessian *> optimized;
  optimized.resize(toOptimize.size());

  if (multiThreading)
    tread_reduce_.reduce(boost::bind(&FrontEnd::activatePointsMT_Reductor, this,
                                     &optimized, &toOptimize, _1, _2, _3, _4),
                         0, toOptimize.size(), 50);

  else
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0,
                              0);

  for (unsigned k = 0; k < toOptimize.size(); k++) {
    PointHessian *newpoint = optimized[k];
    ImmaturePoint *ph = toOptimize[k];

    if (newpoint != 0 && newpoint != (PointHessian *)((long)(-1))) {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      newpoint->host->pointHessians.push_back(newpoint);
      ef_->insertPoint(newpoint);
      for (PointFrameResidual *r : newpoint->residuals)
        ef_->insertResidual(r);
      assert(newpoint->efPoint != 0);
      delete ph;
    } else if (newpoint == (PointHessian *)((long)(-1)) ||
               ph->lastTraceStatus == IPS_OOB) {
      delete ph;
      ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
    } else {
      assert(newpoint == 0 || newpoint == (PointHessian *)((long)(-1)));
    }
  }

  for (FrameHessian *host : frame_hessians_) {
    for (int i = 0; i < (int)host->immaturePoints.size(); i++) {
      if (host->immaturePoints[i] == 0) {
        host->immaturePoints[i] = host->immaturePoints.back();
        host->immaturePoints.pop_back();
        i--;
      }
    }
  }
}

void FrontEnd::activatePointsOldFirst() { assert(false); }

void FrontEnd::flagPointsForRemoval() {
  assert(EFIndicesValid);

  std::vector<FrameHessian *> fhsToKeepPoints;
  std::vector<FrameHessian *> fhsToMargPoints;

  // if(setting_margPointVisWindow>0)
  {
    for (int i = ((int)frame_hessians_.size()) - 1;
         i >= 0 && i >= ((int)frame_hessians_.size()); i--)
      if (!frame_hessians_[i]->flaggedForMarginalization)
        fhsToKeepPoints.push_back(frame_hessians_[i]);

    for (int i = 0; i < (int)frame_hessians_.size(); i++)
      if (frame_hessians_[i]->flaggedForMarginalization)
        fhsToMargPoints.push_back(frame_hessians_[i]);
  }

  // ef_->setAdjointsF();
  // ef_->setDeltaF(&h_calib_);
  int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

  for (FrameHessian *host : frame_hessians_) // go through all active frames
  {
    for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
      PointHessian *ph = host->pointHessians[i];
      if (ph == 0)
        continue;

      if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
        host->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        host->pointHessians[i] = 0;
        flag_nores++;
      } else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) ||
                 host->flaggedForMarginalization) {
        flag_oob++;
        if (ph->isInlierNew()) {
          flag_in++;
          int ngoodRes = 0;
          for (PointFrameResidual *r : ph->residuals) {
            r->resetOOB();
            r->linearize(&h_calib_);
            r->efResidual->isLinearized = false;
            r->applyRes(true);
            if (r->efResidual->isActive()) {
              r->efResidual->fixLinearizationF(ef_);
              ngoodRes++;
            }
          }
          if (ph->idepth_hessian > setting_minIdepthH_marg) {
            flag_inin++;
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
            host->pointHessiansMarginalized.push_back(ph);
          } else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
          }

        } else {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

          // printf("drop point in frame %d (%d goodRes, %d activeRes)\n",
          // ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
        }

        host->pointHessians[i] = 0;
      }
    }

    for (int i = 0; i < (int)host->pointHessians.size(); i++) {
      if (host->pointHessians[i] == 0) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }
  }
}

void FrontEnd::addActiveFrame(ImageAndExposure *image, int id) {

  // if(is_lost_) return;

  boost::unique_lock<boost::mutex> lock(track_mutex_);

  // ======================== add into all_frame_history_
  // =========================
  FrameHessian *fh = new FrameHessian();
  FrameShell *shell = new FrameShell();
  shell->camToWorld =
      SE3(); // no lock required, as fh is not used anywhere yet.
  shell->aff_g2l = AffLight(0, 0);
  shell->marginalizedAt = shell->id = all_frame_history_.size();
  shell->timestamp = image->timestamp;
  shell->incoming_id = id;
  fh->shell = shell;
  all_frame_history_.push_back(shell);

  // ======================= make Images / derivatives etc. ====================
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &h_calib_);

  if (!initialized_) {
    // use initializer!
    // first frame set. fh is kept by coarse_initializer_.
    if (coarse_initializer_->frameID < 0) {
      coarse_initializer_->setFirst(&h_calib_, fh);
    } else if (coarse_initializer_->trackFrame(fh,
                                               output_wrapper_)) // if SNAPPED
    {

      initializeFromInitializer(fh);
      lock.unlock();
      deliverTrackedFrame(fh, true);
    } else {
      // if still initializing
      fh->shell->poseValid = false;
      delete fh;
    }
    return;
  } else // do front-end operation.
  {
    // =================] SWAP tracking reference?. =========================
    if (coarse_tracker_for_new_kf_->refFrameID > coarse_tracker_->refFrameID) {
      boost::unique_lock<boost::mutex> crlock(coarse_tracker_swap_mutex_);
      CoarseTracker *tmp = coarse_tracker_;
      coarse_tracker_ = coarse_tracker_for_new_kf_;
      coarse_tracker_for_new_kf_ = tmp;
    }

    Vec4 tres = trackNewCoarse(fh);
    if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) ||
        !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3])) {
      printf("Initial Tracking failed: LOST!\n");
      is_lost_ = true;
      return;
    }

    // printf("tracker %f\n", (double)tres[0]);

    bool needToMakeKF = false;
    if (setting_keyframesPerSecond > 0) {
      needToMakeKF =
          all_frame_history_.size() == 1 ||
          (fh->shell->timestamp - all_keyframes_history_.back()->timestamp) >
              0.95f / setting_keyframesPerSecond;
    } else {
      Vec2 refToFh = AffLight::fromToVecExposure(
          coarse_tracker_->lastRef->ab_exposure, fh->ab_exposure,
          coarse_tracker_->lastRef_aff_g2l, fh->shell->aff_g2l);

      // BRIGHTNESS CHECK
      needToMakeKF = all_frame_history_.size() == 1 ||
                     setting_kfGlobalWeight * setting_maxShiftWeightT *
                                 sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                             setting_kfGlobalWeight * setting_maxShiftWeightR *
                                 sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
                             setting_kfGlobalWeight * setting_maxShiftWeightRT *
                                 sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
                             setting_kfGlobalWeight * setting_maxAffineWeight *
                                 fabs(logf((float)refToFh[0])) >
                         1 ||
                     2 * coarse_tracker_->firstCoarseRMSE < tres[0];
    }

    for (IOWrap::Output3DWrapper *ow : output_wrapper_)
      ow->publishCamPose(fh->shell, &h_calib_);

    cur_pose_ = fh->shell->camToWorld;

    lock.unlock();
    deliverTrackedFrame(fh, needToMakeKF);

    return;
  }
}
void FrontEnd::deliverTrackedFrame(FrameHessian *fh, bool needKF) {
  if (goStepByStep && last_ref_stop_id_ != coarse_tracker_->refFrameID) {
    MinimalImageF3 img(wG[0], hG[0], fh->dI);
    IOWrap::displayImage("frameToTrack", &img);
    while (true) {
      char k = IOWrap::waitKey(0);
      if (k == ' ')
        break;
      handleKey(k);
    }
    last_ref_stop_id_ = coarse_tracker_->refFrameID;
  } else
    handleKey(IOWrap::waitKey(1));

  if (needKF)
    makeKeyFrame(fh);
  else
    makeNonKeyFrame(fh);
}

void FrontEnd::makeNonKeyFrame(FrameHessian *fh) {
  {
    boost::unique_lock<boost::mutex> crlock(shell_pose_mutex_);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld =
        fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  traceNewCoarse(fh);
  delete fh;
}

void FrontEnd::makeKeyFrame(FrameHessian *fh) {
  {
    boost::unique_lock<boost::mutex> crlock(shell_pose_mutex_);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld =
        fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  traceNewCoarse(fh);

  boost::unique_lock<boost::mutex> lock(map_mutex_);

  // ================= Flag Frames to be Marginalized. =========================
  flagFramesForMarginalization(fh);

  // ================ add New Frame to Hessian Struct. =========================
  fh->idx = frame_hessians_.size();
  frame_hessians_.push_back(fh);
  fh->frameID = all_keyframes_history_.size() + prev_kf_size_;
  all_keyframes_history_.push_back(fh->shell);
  ef_->insertFrame(fh, &h_calib_);

  setPrecalcValues();

  // =================== add new residuals for old points ==================
  int numFwdResAdde = 0;
  for (FrameHessian *fh1 : frame_hessians_) // go through all active frames
  {
    if (fh1 == fh)
      continue;
    for (PointHessian *ph : fh1->pointHessians) {
      PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
      r->setState(ResState::IN);
      ph->residuals.push_back(r);
      ef_->insertResidual(r);
      ph->lastResiduals[1] = ph->lastResiduals[0];
      ph->lastResiduals[0] =
          std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
      numFwdResAdde += 1;
    }
  }

  // =========== Activate Points (& flag for marginalization). ==============
  activatePointsMT();
  ef_->makeIDX();

  // =========================== OPTIMIZE ALL =========================

  fh->frameEnergyTH = frame_hessians_.back()->frameEnergyTH;

  auto t0 = std::chrono::steady_clock::now();
  float rmse = optimize(setting_maxOptIterations);
  auto t1 = std::chrono::steady_clock::now();
  pts_count_.emplace_back(ef_->nPoints);
  opt_time_.emplace_back(duration(t0, t1));

  // =============== Figure Out if INITIALIZATION FAILED ============
  if ((all_keyframes_history_.size() == 2 &&
       rmse > 25 * benchmark_initializerSlackFactor) ||
      (all_keyframes_history_.size() == 3 &&
       rmse > 15 * benchmark_initializerSlackFactor) ||
      (all_keyframes_history_.size() == 4 &&
       rmse > 10 * benchmark_initializerSlackFactor)) {
    printf("I THINK INITIALIZATION FAILED: KF: %d, RMSE: %.2f\n",
           all_keyframes_history_.size(), rmse);
    init_failed_ = true;
  }

  if (is_lost_ || init_failed_)
    return;

  // =========================== REMOVE OUTLIER =========================
  removeOutliers();

  // =========================== SCALE OPTIMIZATION =========================
  if (scale_optimizer_ && all_keyframes_history_.size() > 4) {
    float scale_error = optimizeScale();
    scale_errors_.push_back(scale_error);
  } else {
    scale_errors_.push_back(-1.0);
  }

  {
    boost::unique_lock<boost::mutex> crlock(coarse_tracker_swap_mutex_);
    coarse_tracker_for_new_kf_->makeK(&h_calib_);
    coarse_tracker_for_new_kf_->setCoarseTrackingRef(frame_hessians_);

    coarse_tracker_for_new_kf_->debugPlotIDepthMap(
        &min_id_jet_vis_tracker_, &max_id_jet_vis_tracker_, output_wrapper_);
    coarse_tracker_for_new_kf_->debugPlotIDepthMapFloat(output_wrapper_);
  }

  debugPlot("post Optimize");

  // ============= (Activate-)Marginalize Points ===============
  flagPointsForRemoval();
  ef_->dropPointsF();
  getNullspaces(ef_->lastNullspaces_pose, ef_->lastNullspaces_scale,
                ef_->lastNullspaces_affA, ef_->lastNullspaces_affB);
  ef_->marginalizePointsF();

  // ================== add new Immature points & new residuals ============
  t0 = std::chrono::steady_clock::now();
  makeNewTraces(fh, 0);
  t1 = std::chrono::steady_clock::now();
  feature_detect_time_.emplace_back(duration(t0, t1));

  for (IOWrap::Output3DWrapper *ow : output_wrapper_) {
    ow->publishGraph(ef_->connectivityMap);
    ow->publishKeyframes(frame_hessians_, false, &h_calib_);
  }

  // =========================== Marginalize Frames =========================

  for (unsigned int i = 0; i < frame_hessians_.size(); i++)
    if (frame_hessians_[i]->flaggedForMarginalization) {
      marginalizeFrame(frame_hessians_[i], scale_errors_[i]);
      i = 0;
    }
}

void FrontEnd::initializeFromInitializer(FrameHessian *newFrame) {
  boost::unique_lock<boost::mutex> lock(map_mutex_);

  // add firstframe.
  FrameHessian *firstFrame = coarse_initializer_->firstFrame;
  firstFrame->idx = frame_hessians_.size();
  frame_hessians_.push_back(firstFrame);
  firstFrame->frameID = all_keyframes_history_.size() + prev_kf_size_;
  all_keyframes_history_.push_back(firstFrame->shell);
  ef_->insertFrame(firstFrame, &h_calib_);
  setPrecalcValues();

  // int numPointsTotal = makePixelStatus(firstFrame->dI, selection_map_, wG[0],
  // hG[0], setting_desiredDensity); int numPointsTotal =
  // pixel_selector_->makeMaps(firstFrame->dIp,
  // selection_map_,setting_desiredDensity);

  firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

  float sumID = 1e-5, numID = 1e-5;
  for (int i = 0; i < coarse_initializer_->numPoints[0]; i++) {
    sumID += coarse_initializer_->points[0][i].iR;
    numID++;
  }
  float rescaleFactor = 1 / (sumID / numID);

  // randomly sub-select the points I need.
  float keepPercentage =
      setting_desiredPointDensity / coarse_initializer_->numPoints[0];

  if (!setting_debugout_runquiet)
    printf("Initialization: keep %.1f%% (need %d, have %d)!\n",
           100 * keepPercentage, (int)(setting_desiredPointDensity),
           coarse_initializer_->numPoints[0]);

  for (int i = 0; i < coarse_initializer_->numPoints[0]; i++) {
    if (rand() / (float)RAND_MAX > keepPercentage)
      continue;

    Pnt *point = coarse_initializer_->points[0] + i;
    ImmaturePoint *pt =
        new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame,
                          point->my_type, &h_calib_);

    if (!std::isfinite(pt->energyTH)) {
      delete pt;
      continue;
    }

    pt->idepth_max = pt->idepth_min = 1;
    PointHessian *ph = new PointHessian(pt, &h_calib_);
    delete pt;
    if (!std::isfinite(ph->energyTH)) {
      delete ph;
      continue;
    }

    ph->setIdepthScaled(point->iR * rescaleFactor);
    ph->setIdepthZero(ph->idepth);
    ph->hasDepthPrior = true;
    ph->setPointStatus(PointHessian::ACTIVE);

    firstFrame->pointHessians.push_back(ph);
    ef_->insertPoint(ph);
  }

  SE3 firstToNew = coarse_initializer_->thisToNext;
  firstToNew.translation() /= rescaleFactor;

  // really no lock required, as we are initializing.
  {
    boost::unique_lock<boost::mutex> crlock(shell_pose_mutex_);
    firstFrame->shell->camToWorld = cur_pose_;
    firstFrame->shell->aff_g2l = AffLight(0, 0);
    firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),
                                 firstFrame->shell->aff_g2l);
    firstFrame->shell->trackingRef = 0;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = cur_pose_ * firstToNew.inverse();
    newFrame->shell->aff_g2l = AffLight(0, 0);
    newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),
                               newFrame->shell->aff_g2l);
    newFrame->shell->trackingRef = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();
  }

  initialized_ = true;
  printf("INITIALIZE FROM INITIALIZER (%d pts)!\n",
         (int)firstFrame->pointHessians.size());
}

void FrontEnd::makeNewTraces(FrameHessian *newFrame, float *gtDepth) {
  pixel_selector_->allowFast = true;
  // int numPointsTotal = makePixelStatus(newFrame->dI, selection_map_, wG[0],
  // hG[0], setting_desiredDensity);
  int numPointsTotal = pixel_selector_->makeMaps(
      newFrame, selection_map_, setting_desiredImmatureDensity);

  newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
  // fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

  for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
    for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
      int i = x + y * wG[0];
      if (selection_map_[i] == 0)
        continue;

      ImmaturePoint *impt =
          new ImmaturePoint(x, y, newFrame, selection_map_[i], &h_calib_);
      if (!std::isfinite(impt->energyTH))
        delete impt;
      else
        newFrame->immaturePoints.push_back(impt);
    }
  // printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());
}

void FrontEnd::setPrecalcValues() {
  for (FrameHessian *fh : frame_hessians_) {
    fh->targetPrecalc.resize(frame_hessians_.size());
    for (unsigned int i = 0; i < frame_hessians_.size(); i++)
      fh->targetPrecalc[i].set(fh, frame_hessians_[i], &h_calib_);
  }

  ef_->setDeltaF(&h_calib_);
}

/* ========================= Scale optimization ========================== */
void FrontEnd::setScaleOptimizer(ScaleOptimizer *scale_optimizer) {
  scale_optimizer_ = scale_optimizer;
}

void FrontEnd::addStereoImg(cv::Mat stereo_img, int stereo_id) {
  if (scale_optimizer_) {
    stereo_id_img_queue_.push({stereo_id, stereo_img.clone()});
  }
}

float FrontEnd::optimizeScale() {
  static bool scale_trapped = false;
  static int scale_opt_fails = 0;

  if (!scale_optimizer_) {
    return -1;
  }

  // find the corresponding stereo frame
  FrameHessian *last_fh = frame_hessians_.back();
  int stereo_id = stereo_id_img_queue_.front().first;
  cv::Mat stereo_img;
  while (!stereo_id_img_queue_.empty()) {
    stereo_id = stereo_id_img_queue_.front().first;
    if (stereo_id == last_fh->shell->incoming_id) {
      stereo_img = stereo_id_img_queue_.front().second.clone();
      stereo_id_img_queue_.pop();
      break;
    }
    stereo_id_img_queue_.pop();
  }

  if (stereo_id != last_fh->shell->incoming_id) {
    printf("Cannot find stereo frame\n");
    return -1;
  }

  // find the optimal scale
  auto t0 = std::chrono::steady_clock::now();
  float new_scale = 1.0;
  float scale_error = -1;
  if (scale_trapped) {
    scale_error = scale_optimizer_->optimize(
        frame_hessians_, stereo_img, &h_calib_, new_scale, pyrLevelsUsed - 1);
  } else {
    std::vector<float> scale_guess = {0.1, 1, 5, 10, 15, 25, 30, 50};
    for (float cur_scale : scale_guess) {
      float cur_error = scale_optimizer_->optimize(
          frame_hessians_, stereo_img, &h_calib_, cur_scale, pyrLevelsUsed - 1);
      if (cur_error > 0 && (scale_error < 0 || scale_error > cur_error)) {
        scale_error = cur_error;
        new_scale = cur_scale;
      }
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  scale_opt_time_.emplace_back(duration(t0, t1));

  // when scale optimization is working, we don't expect sudden scale change
  if (scale_trapped && fabs(new_scale - 1.0) > 0.5) {
    scale_error = -1;
  }

  // when scale optimization fails continuously, we need to re-initialize scale
  scale_opt_fails = scale_error > 0 ? 0 : scale_opt_fails + 1;
  if (scale_opt_fails > 5) {
    scale_trapped = false;
  }

  // adjust the scale of DSO
  if (scale_error > 0) {
    // std::cout << "Changing scale to " << scale << std::endl;
    for (FrameHessian *fh : frame_hessians_) {
      for (PointHessian *ph : fh->pointHessians) {
        ph->setIdepthScaled(ph->idepth / new_scale);
        ph->setIdepthZero(ph->idepth);
        if (ph->lastResiduals[0].first != 0 &&
            ph->lastResiduals[0].second == ResState::IN) {
          PointFrameResidual *r = ph->lastResiduals[0].first;
          assert(r->efResidual->isActive() && r->target == last_fh);
          r->centerProjectedTo[2] /= new_scale;
        }
      }
    }

    boost::unique_lock<boost::mutex> crlock(shell_pose_mutex_);
    last_fh->shell->camToTrackingRef.translation() *= new_scale;
    last_fh->shell->camToWorld = last_fh->shell->trackingRef->camToWorld *
                                 last_fh->shell->camToTrackingRef;
    last_fh->setEvalPT_scaled(last_fh->shell->camToWorld.inverse(),
                              last_fh->shell->aff_g2l);
    setPrecalcValues();

    if (!scale_trapped) {
      printf("scale trapped to %.2f\n", new_scale);
    }
    scale_trapped = true;
  }

  return scale_error;
}

/* ============================ Loop closure ============================= */
void FrontEnd::setLoopHandler(LoopHandler *loop_handler) {
  loop_handler_ = loop_handler;
}

int FrontEnd::getTotalKFSize() {
  return all_keyframes_history_.size() + prev_kf_size_;
}

/* ============================= Statistics ============================== */
void FrontEnd::printTimeStat() {
  std::cout << "===========VO Time============ " << std::endl;
  std::cout << "feature_detect " << 1000 * average(feature_detect_time_)
            << " * " << feature_detect_time_.size() << std::endl;
  std::cout << "scale_opt " << 1000 * average(scale_opt_time_) << " * "
            << scale_opt_time_.size() << std::endl;
  std::cout << "dso_opt " << 1000 * average(opt_time_) << " * "
            << opt_time_.size() << std::endl;
  std::cout << "pts_count_ " << average(pts_count_) << std::endl;
  std::cout << "============================== " << std::endl;
}

} // namespace dso
