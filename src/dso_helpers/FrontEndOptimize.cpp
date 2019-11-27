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

#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso {

void FrontEnd::linearizeAll_Reductor(
    bool fixLinearization, std::vector<PointFrameResidual *> *toRemove, int min,
    int max, Vec10 *stats, int tid) {
  for (int k = min; k < max; k++) {
    PointFrameResidual *r = active_residuals_[k];
    (*stats)[0] += r->linearize(&h_calib_);

    if (fixLinearization) {
      r->applyRes(true);

      if (r->efResidual->isActive()) {
        if (r->isNew) {
          PointHessian *p = r->point;
          Vec3f ptp_inf =
              r->host->targetPrecalc[r->target->idx].PRE_KRKiTll *
              Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
          Vec3f ptp = ptp_inf +
                      r->host->targetPrecalc[r->target->idx].PRE_KtTll *
                          p->idepth_scaled; // projected point with real depth.
          float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) -
                                (ptp.head<2>() / ptp[2]))
                                   .norm(); // 0.01 = one pixel.

          if (relBS > p->maxRelBaseline)
            p->maxRelBaseline = relBS;

          p->numGoodResiduals++;
        }
      } else {
        toRemove[tid].push_back(active_residuals_[k]);
      }
    }
  }
}

void FrontEnd::applyRes_Reductor(bool copyJacobians, int min, int max,
                                 Vec10 *stats, int tid) {
  for (int k = min; k < max; k++)
    active_residuals_[k]->applyRes(true);
}
void FrontEnd::setNewFrameEnergyTH() {

  // collect all residuals and make decision on TH.
  all_res_vec_.clear();
  all_res_vec_.reserve(active_residuals_.size() * 2);
  FrameHessian *newFrame = frame_hessians_.back();

  for (PointFrameResidual *r : active_residuals_)
    if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) {
      all_res_vec_.push_back(r->state_NewEnergyWithOutlier);
    }

  if (all_res_vec_.size() == 0) {
    newFrame->frameEnergyTH = 12 * 12 * patternNum;
    return; // should never happen, but lets make sure.
  }

  int nthIdx = setting_frameEnergyTHN * all_res_vec_.size();

  assert(nthIdx < (int)all_res_vec_.size());
  assert(setting_frameEnergyTHN < 1);

  std::nth_element(all_res_vec_.begin(), all_res_vec_.begin() + nthIdx,
                   all_res_vec_.end());
  float nthElement = sqrtf(all_res_vec_[nthIdx]);

  newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH =
      26.0f * setting_frameEnergyTHConstWeight +
      newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *=
      setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

  //
  //  int good=0,bad=0;
  //  for(float f : all_res_vec_) if(f<newFrame->frameEnergyTH) good++; else
  //  bad++; printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)!
  //  \n",
  //      meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
  //      good, bad);
}
Vec3 FrontEnd::linearizeAll(bool fixLinearization) {
  double lastEnergyP = 0;
  double lastEnergyR = 0;
  double num = 0;

  std::vector<PointFrameResidual *> toRemove[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++)
    toRemove[i].clear();

  if (multiThreading) {
    tread_reduce_.reduce(boost::bind(&FrontEnd::linearizeAll_Reductor, this,
                                     fixLinearization, toRemove, _1, _2, _3,
                                     _4),
                         0, active_residuals_.size(), 0);
    lastEnergyP = tread_reduce_.stats[0];
  } else {
    Vec10 stats;
    linearizeAll_Reductor(fixLinearization, toRemove, 0,
                          active_residuals_.size(), &stats, 0);
    lastEnergyP = stats[0];
  }

  setNewFrameEnergyTH();

  if (fixLinearization) {

    for (PointFrameResidual *r : active_residuals_) {
      PointHessian *ph = r->point;
      if (ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].second = r->state_state;
      else if (ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].second = r->state_state;
    }

    int nResRemoved = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
      for (PointFrameResidual *r : toRemove[i]) {
        PointHessian *ph = r->point;

        if (ph->lastResiduals[0].first == r)
          ph->lastResiduals[0].first = 0;
        else if (ph->lastResiduals[1].first == r)
          ph->lastResiduals[1].first = 0;

        for (unsigned int k = 0; k < ph->residuals.size(); k++)
          if (ph->residuals[k] == r) {
            ef_->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals, k);
            nResRemoved++;
            break;
          }
      }
    }
    // printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved,
    // (int)active_residuals_.size());
  }

  return Vec3(lastEnergyP, lastEnergyR, num);
}

// applies step to linearization point.
bool FrontEnd::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR,
                                float stepfacA, float stepfacD) {
  //  float meanStepC=0,meanStepP=0,meanStepD=0;
  //  meanStepC += h_calib_.step.norm();

  Vec10 pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<4>(6).setConstant(stepfacA);

  float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

  float sumNID = 0;

  if (setting_solverMode & SOLVER_MOMENTUM) {
    h_calib_.setValue(h_calib_.value_backup + h_calib_.step);
    for (FrameHessian *fh : frame_hessians_) {
      Vec10 step = fh->step;
      step.head<6>() += 0.5f * (fh->step_backup.head<6>());

      fh->setState(fh->state_backup + step);
      sumA += step[6] * step[6];
      sumB += step[7] * step[7];
      sumT += step.segment<3>(0).squaredNorm();
      sumR += step.segment<3>(3).squaredNorm();

      for (PointHessian *ph : fh->pointHessians) {
        float step = ph->step + 0.5f * (ph->step_backup);
        ph->setIdepth(ph->idepth_backup + step);
        sumID += step * step;
        sumNID += fabsf(ph->idepth_backup);
        numID++;

        ph->setIdepthZero(ph->idepth_backup + step);
      }
    }
  } else {
    h_calib_.setValue(h_calib_.value_backup + stepfacC * h_calib_.step);
    for (FrameHessian *fh : frame_hessians_) {
      fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
      sumA += fh->step[6] * fh->step[6];
      sumB += fh->step[7] * fh->step[7];
      sumT += fh->step.segment<3>(0).squaredNorm();
      sumR += fh->step.segment<3>(3).squaredNorm();

      for (PointHessian *ph : fh->pointHessians) {
        ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
        sumID += ph->step * ph->step;
        sumNID += fabsf(ph->idepth_backup);
        numID++;

        ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
      }
    }
  }

  sumA /= frame_hessians_.size();
  sumB /= frame_hessians_.size();
  sumR /= frame_hessians_.size();
  sumT /= frame_hessians_.size();
  sumID /= numID;
  sumNID /= numID;

  if (!setting_debugout_runquiet)
    printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
           sqrtf(sumA) / (0.0005 * setting_thOptIterations),
           sqrtf(sumB) / (0.00005 * setting_thOptIterations),
           sqrtf(sumR) / (0.00005 * setting_thOptIterations),
           sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations));

  EFDeltaValid = false;
  setPrecalcValues();

  return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
         sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
  //
  //  printf("mean steps: %f %f %f!\n",
  //      meanStepC, meanStepP, meanStepD);
}

// sets linearization point.
void FrontEnd::backupState(bool backupLastStep) {
  if (setting_solverMode & SOLVER_MOMENTUM) {
    if (backupLastStep) {
      h_calib_.step_backup = h_calib_.step;
      h_calib_.value_backup = h_calib_.value;
      for (FrameHessian *fh : frame_hessians_) {
        fh->step_backup = fh->step;
        fh->state_backup = fh->get_state();
        for (PointHessian *ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = ph->step;
        }
      }
    } else {
      h_calib_.step_backup.setZero();
      h_calib_.value_backup = h_calib_.value;
      for (FrameHessian *fh : frame_hessians_) {
        fh->step_backup.setZero();
        fh->state_backup = fh->get_state();
        for (PointHessian *ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = 0;
        }
      }
    }
  } else {
    h_calib_.value_backup = h_calib_.value;
    for (FrameHessian *fh : frame_hessians_) {
      fh->state_backup = fh->get_state();
      for (PointHessian *ph : fh->pointHessians)
        ph->idepth_backup = ph->idepth;
    }
  }
}

// sets linearization point.
void FrontEnd::loadSateBackup() {
  h_calib_.setValue(h_calib_.value_backup);
  for (FrameHessian *fh : frame_hessians_) {
    fh->setState(fh->state_backup);
    for (PointHessian *ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup);

      ph->setIdepthZero(ph->idepth_backup);
    }
  }

  EFDeltaValid = false;
  setPrecalcValues();
}

double FrontEnd::calcMEnergy() {
  if (setting_forceAceptStep)
    return 0;
  // calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
  // ef_->makeIDX();
  // ef_->setDeltaF(&h_calib_);
  return ef_->calcMEnergyF();
}

void FrontEnd::printOptRes(const Vec3 &res, double resL, double resM,
                           double resPrior, double LExact, float a, float b) {
  printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n", res[0],
         sqrtf((float)(res[0] / (patternNum * ef_->resInA))), ef_->resInA,
         ef_->resInM, a, b);
}

float FrontEnd::optimize(int mnumOptIts) {

  if (frame_hessians_.size() < 2)
    return 0;
  if (frame_hessians_.size() < 3)
    mnumOptIts = 20;
  if (frame_hessians_.size() < 4)
    mnumOptIts = 15;

  // get statistics and active residuals.

  active_residuals_.clear();
  int numPoints = 0;
  int numLRes = 0;
  for (FrameHessian *fh : frame_hessians_)
    for (PointHessian *ph : fh->pointHessians) {
      for (PointFrameResidual *r : ph->residuals) {
        if (!r->efResidual->isLinearized) {
          active_residuals_.push_back(r);
          r->resetOOB();
        } else
          numLRes++;
      }
      numPoints++;
    }

  if (!setting_debugout_runquiet)
    printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef_->nPoints,
           (int)active_residuals_.size(), numLRes);

  Vec3 lastEnergy = linearizeAll(false);
  double lastEnergyL = calcLEnergy();
  double lastEnergyM = calcMEnergy();

  if (multiThreading)
    tread_reduce_.reduce(
        boost::bind(&FrontEnd::applyRes_Reductor, this, true, _1, _2, _3, _4),
        0, active_residuals_.size(), 50);
  else
    applyRes_Reductor(true, 0, active_residuals_.size(), 0, 0);

  if (!setting_debugout_runquiet) {
    printf("Initial Error       \t");
    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0,
                frame_hessians_.back()->aff_g2l().a,
                frame_hessians_.back()->aff_g2l().b);
  }

  debugPlotTracking();

  double lambda = 1e-1;
  float stepsize = 1;
  VecX previousX = VecX::Constant(CPARS + 8 * frame_hessians_.size(), NAN);
  for (int iteration = 0; iteration < mnumOptIts; iteration++) {
    // solve!
    backupState(iteration != 0);
    // solveSystemNew(0);
    solveSystem(iteration, lambda);
    double incDirChange = (1e-20 + previousX.dot(ef_->lastX)) /
                          (1e-20 + previousX.norm() * ef_->lastX.norm());
    previousX = ef_->lastX;

    if (std::isfinite(incDirChange) &&
        (setting_solverMode & SOLVER_STEPMOMENTUM)) {
      float newStepsize = exp(incDirChange * 1.4);
      if (incDirChange < 0 && stepsize > 1)
        stepsize = 1;

      stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
      if (stepsize > 2)
        stepsize = 2;
      if (stepsize < 0.25)
        stepsize = 0.25;
    }

    bool canbreak =
        doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

    // eval new energy!
    Vec3 newEnergy = linearizeAll(false);
    double newEnergyL = calcLEnergy();
    double newEnergyM = calcMEnergy();

    if (!setting_debugout_runquiet) {
      printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
             (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
              lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)
                 ? "ACCEPT"
                 : "REJECT",
             iteration, log10(lambda), incDirChange, stepsize);
      printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0,
                  frame_hessians_.back()->aff_g2l().a,
                  frame_hessians_.back()->aff_g2l().b);
    }

    if (setting_forceAceptStep ||
        (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
         lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {

      if (multiThreading)
        tread_reduce_.reduce(boost::bind(&FrontEnd::applyRes_Reductor, this,
                                         true, _1, _2, _3, _4),
                             0, active_residuals_.size(), 50);
      else
        applyRes_Reductor(true, 0, active_residuals_.size(), 0, 0);

      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      lambda *= 0.25;
    } else {
      loadSateBackup();
      lastEnergy = linearizeAll(false);
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy();
      lambda *= 1e2;
    }

    if (canbreak && iteration >= setting_minOptIterations)
      break;
  }

  Vec10 newStateZero = Vec10::Zero();
  newStateZero.segment<2>(6) =
      frame_hessians_.back()->get_state().segment<2>(6);

  frame_hessians_.back()->setEvalPT(frame_hessians_.back()->PRE_worldToCam,
                                    newStateZero);
  EFDeltaValid = false;
  EFAdjointsValid = false;
  ef_->setAdjointsF(&h_calib_);
  setPrecalcValues();

  lastEnergy = linearizeAll(true);

  if (!std::isfinite((double)lastEnergy[0]) ||
      !std::isfinite((double)lastEnergy[1]) ||
      !std::isfinite((double)lastEnergy[2])) {
    printf("KF Tracking failed: LOST!\n");
    is_lost_ = true;
  }

  {
    boost::unique_lock<boost::mutex> crlock(shell_pose_mutex_);
    for (FrameHessian *fh : frame_hessians_) {
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l = fh->aff_g2l();
    }
  }

  debugPlotTracking();

  return sqrtf((float)(lastEnergy[0] / (patternNum * ef_->resInA)));
}

void FrontEnd::solveSystem(int iteration, double lambda) {
  ef_->lastNullspaces_forLogging =
      getNullspaces(ef_->lastNullspaces_pose, ef_->lastNullspaces_scale,
                    ef_->lastNullspaces_affA, ef_->lastNullspaces_affB);

  ef_->solveSystemF(iteration, lambda, &h_calib_);
}

double FrontEnd::calcLEnergy() {
  if (setting_forceAceptStep)
    return 0;

  double Ef = ef_->calcLEnergyF_MT();
  return Ef;
}

void FrontEnd::removeOutliers() {
  int numPointsDropped = 0;
  for (FrameHessian *fh : frame_hessians_) {
    for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
      PointHessian *ph = fh->pointHessians[i];
      if (ph == 0)
        continue;

      if (ph->residuals.size() == 0) {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
      }
    }
  }
  ef_->dropPointsF();
}

std::vector<VecX> FrontEnd::getNullspaces(std::vector<VecX> &nullspaces_pose,
                                          std::vector<VecX> &nullspaces_scale,
                                          std::vector<VecX> &nullspaces_affA,
                                          std::vector<VecX> &nullspaces_affB) {
  nullspaces_pose.clear();
  nullspaces_scale.clear();
  nullspaces_affA.clear();
  nullspaces_affB.clear();

  int n = CPARS + frame_hessians_.size() * 8;
  std::vector<VecX> nullspaces_x0_pre;
  for (int i = 0; i < 6; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian *fh : frame_hessians_) {
      nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
      nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
      nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_pose.push_back(nullspace_x0);
  }
  for (int i = 0; i < 2; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian *fh : frame_hessians_) {
      nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) =
          fh->nullspaces_affine.col(i).head<2>();
      nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
      nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    if (i == 0)
      nullspaces_affA.push_back(nullspace_x0);
    if (i == 1)
      nullspaces_affB.push_back(nullspace_x0);
  }

  VecX nullspace_x0(n);
  nullspace_x0.setZero();
  for (FrameHessian *fh : frame_hessians_) {
    nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.push_back(nullspace_x0);
  nullspaces_scale.push_back(nullspace_x0);

  return nullspaces_x0_pre;
}

} // namespace dso
