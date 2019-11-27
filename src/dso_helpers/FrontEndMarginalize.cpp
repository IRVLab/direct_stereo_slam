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
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/ResidualProjections.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso {

inline void
deleteOutOrderFrameHessian(std::vector<FrameHessian *> &frame_hessians,
                           std::vector<float> &scale_errors,
                           const FrameHessian *frame) {
  int i = -1;
  for (unsigned int k = 0; k < frame_hessians.size(); k++) {
    if (frame_hessians[k] == frame) {
      i = k;
      break;
    }
  }
  assert(i != -1);

  for (unsigned int k = i + 1; k < frame_hessians.size(); k++) {
    frame_hessians[k - 1] = frame_hessians[k];
    scale_errors[k - 1] = scale_errors[k];
  }
  frame_hessians.pop_back();
  scale_errors.pop_back();

  // delete frame; moved to LoopHandler::run()
}

void FrontEnd::flagFramesForMarginalization(FrameHessian *newFH) {
  if (setting_minFrameAge > setting_maxFrames) {
    for (int i = setting_maxFrames; i < (int)frame_hessians_.size(); i++) {
      FrameHessian *fh = frame_hessians_[i - setting_maxFrames];
      fh->flaggedForMarginalization = true;
    }
    return;
  }

  int flagged = 0;
  // marginalize all frames that have not enough points.
  for (int i = 0; i < (int)frame_hessians_.size(); i++) {
    FrameHessian *fh = frame_hessians_[i];
    int in = fh->pointHessians.size() + fh->immaturePoints.size();
    int out =
        fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();

    Vec2 refToFh = AffLight::fromToVecExposure(
        frame_hessians_.back()->ab_exposure, fh->ab_exposure,
        frame_hessians_.back()->aff_g2l(), fh->aff_g2l());

    if ((in < setting_minPointsRemaining * (in + out) ||
         fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) &&
        ((int)frame_hessians_.size()) - flagged > setting_minFrames) {
      //      printf("MARGINALIZE frame %d, as only %'d/%'d points remaining
      //      (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated
      //      %d!\n",
      //          fh->frameID, in, in+out,
      //          (int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
      //          (int)fh->pointHessiansMarginalized.size(),
      //          (int)fh->pointHessiansOut.size(), visInLast, outInLast,
      //          fh->statistics_tracesCreatedForThisFrame,
      //          fh->statistics_pointsActivatedForThisFrame);
      fh->flaggedForMarginalization = true;
      flagged++;
    } else {
      //      printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d
      //      %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
      //          fh->frameID, in, in+out,
      //          (int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
      //          (int)fh->pointHessiansMarginalized.size(),
      //          (int)fh->pointHessiansOut.size(), visInLast, outInLast,
      //          fh->statistics_tracesCreatedForThisFrame,
      //          fh->statistics_pointsActivatedForThisFrame);
    }
  }

  // marginalize one.
  if ((int)frame_hessians_.size() - flagged >= setting_maxFrames) {
    double smallestScore = 1;
    FrameHessian *toMarginalize = 0;
    FrameHessian *latest = frame_hessians_.back();

    for (FrameHessian *fh : frame_hessians_) {
      if (fh->frameID > latest->frameID - setting_minFrameAge ||
          fh->frameID == 0)
        continue;
      // if(fh==frame_hessians_.front() == 0) continue;

      double distScore = 0;
      for (FrameFramePrecalc &ffh : fh->targetPrecalc) {
        if (ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 ||
            ffh.target == ffh.host)
          continue;
        distScore += 1 / (1e-5 + ffh.distanceLL);
      }
      distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

      if (distScore < smallestScore) {
        smallestScore = distScore;
        toMarginalize = fh;
      }
    }

    //    printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
    //        toMarginalize->frameID, smallestScore);
    toMarginalize->flaggedForMarginalization = true;
    flagged++;
  }

  //  printf("FRAMES LEFT: ");
  //  for(FrameHessian* fh : frame_hessians_)
  //    printf("%d ", fh->frameID);
  //  printf("\n");
}

void FrontEnd::marginalizeFrame(FrameHessian *frame, float scale_error) {
  // marginalize or remove all this frames points.

  assert((int)frame->pointHessians.size() == 0);

  ef_->marginalizeFrame(frame->efFrame);

  // drop all observations of existing points in that frame.
  float dso_error = 0;
  int energy_count = 0;
  for (FrameHessian *fh : frame_hessians_) {
    if (fh == frame)
      continue;

    for (PointHessian *ph : fh->pointHessians) {
      for (unsigned int i = 0; i < ph->residuals.size(); i++) {
        PointFrameResidual *r = ph->residuals[i];
        if (r->target == frame) {
          if (ph->lastResiduals[0].first == r)
            ph->lastResiduals[0].first = 0;
          else if (ph->lastResiduals[1].first == r)
            ph->lastResiduals[1].first = 0;

          dso_error += r->state_energy;
          energy_count++;

          ef_->dropResidual(r->efResidual);
          deleteOut<PointFrameResidual>(ph->residuals, i);
          break;
        }
      }
    }
  }
  // err = err / count^2 to emphasize on the count
  // *100 is to normalize the error
  dso_error = dso_error / energy_count / energy_count;
  if (energy_count == 0) {
    dso_error = -1;
  }

  {
    std::vector<FrameHessian *> v;
    v.push_back(frame);
    for (IOWrap::Output3DWrapper *ow : output_wrapper_)
      ow->publishKeyframes(v, true, &h_calib_);
  }

  // detect if dso is resetted
  static int prv_existing_kf_size = -1;
  if (prv_existing_kf_size != prev_kf_size_) {
    dso_error = -100;
    prv_existing_kf_size = prev_kf_size_;
  }

  // printf("dso_error: %.2f * %d - scale_error: %.2f \n", dso_error,
  // energy_count,
  //        scale_error);
  loop_handler_->publishKeyframes(frame, &h_calib_, dso_error, scale_error);

  frame->shell->marginalizedAt = frame_hessians_.back()->shell->id;
  frame->shell->movedByOpt = frame->w2c_leftEps().norm();

  deleteOutOrderFrameHessian(frame_hessians_, scale_errors_, frame);
  for (unsigned int i = 0; i < frame_hessians_.size(); i++)
    frame_hessians_[i]->idx = i;

  setPrecalcValues();
  ef_->setAdjointsF(&h_calib_);
}

} // namespace dso
