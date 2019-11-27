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

#pragma once

#include "util/NumType.h"
#include "util/globalCalib.h"
#include "vector"
#include <deque>

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "util/FrameShell.h"
#include "util/IndexThreadReduce.h"
#include "util/NumType.h"
#include "util/Undistort.h"
#include <iostream>

#include "opencv2/core/core.hpp"

#include <math.h>
#include <queue>

#include "loop_closure/LoopHandler.h"
#include "scale_optimization/ScaleOptimizer.h"

namespace dso {
namespace IOWrap {
class Output3DWrapper;
}

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
class ImageAndExposure;
class CoarseDistanceMap;

class EnergyFunctional;

class FrontEnd {
public:
  /* ============================== DSO stuff ============================== */
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FrontEnd(int prev_kf_size = 0);
  virtual ~FrontEnd();

  // adds a new frame, and creates point & residual structs.
  void addActiveFrame(ImageAndExposure *image, int id);

  // marginalizes a frame. drops / marginalizes points & residuals.
  void marginalizeFrame(FrameHessian *frame, float scale_error);

  float optimize(int mnumOptIts);

  void debugPlot(std::string name);

  void printFrameLifetimes();
  // contains pointers to active frames

  std::vector<IOWrap::Output3DWrapper *> output_wrapper_;

  bool is_lost_;
  bool init_failed_;
  bool initialized_;

  void setGammaFunction(float *BInv);

  /* ========================= Scale optimization ========================== */
  void setScaleOptimizer(ScaleOptimizer *scale_optimizer);
  void addStereoImg(cv::Mat stereo_img, int stereo_id);

  /* ============================ Loop closure ============================= */
  SE3 cur_pose_;
  void setLoopHandler(LoopHandler *loop_handler);
  int getTotalKFSize();

  /* ============================= Statistics ============================== */
  void printTimeStat();

private:
  /* ============================== DSO stuff ============================== */
  CalibHessian h_calib_;

  // opt single point
  int optimizePoint(PointHessian *point, int minObs, bool flagOOB);
  PointHessian *
  optimizeImmaturePoint(ImmaturePoint *point, int minObs,
                        ImmaturePointTemporaryResidual *residuals);

  double linAllPointSinle(PointHessian *point, float outlierTHSlack, bool plot);

  // mainPipelineFunctions
  Vec4 trackNewCoarse(FrameHessian *fh);
  void traceNewCoarse(FrameHessian *fh);
  void activatePoints();
  void activatePointsMT();
  void activatePointsOldFirst();
  void flagPointsForRemoval();
  void makeNewTraces(FrameHessian *newFrame, float *gtDepth);
  void initializeFromInitializer(FrameHessian *newFrame);
  void flagFramesForMarginalization(FrameHessian *newFH);

  void removeOutliers();

  // set precalc values.
  void setPrecalcValues();

  // solce. eventually migrate to ef_.
  void solveSystem(int iteration, double lambda);
  Vec3 linearizeAll(bool fixLinearization);
  bool doStepFromBackup(float stepfacC, float stepfacT, float stepfacR,
                        float stepfacA, float stepfacD);
  void backupState(bool backupLastStep);
  void loadSateBackup();
  double calcLEnergy();
  double calcMEnergy();
  void linearizeAll_Reductor(bool fixLinearization,
                             std::vector<PointFrameResidual *> *toRemove,
                             int min, int max, Vec10 *stats, int tid);
  void activatePointsMT_Reductor(std::vector<PointHessian *> *optimized,
                                 std::vector<ImmaturePoint *> *toOptimize,
                                 int min, int max, Vec10 *stats, int tid);
  void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats,
                         int tid);

  void printOptRes(const Vec3 &res, double resL, double resM, double resPrior,
                   double LExact, float a, float b);

  void debugPlotTracking();

  std::vector<VecX> getNullspaces(std::vector<VecX> &nullspaces_pose,
                                  std::vector<VecX> &nullspaces_scale,
                                  std::vector<VecX> &nullspaces_affA,
                                  std::vector<VecX> &nullspaces_affB);

  void setNewFrameEnergyTH();

  // changed by tracker-thread. protected by track_mutex_
  boost::mutex track_mutex_;
  std::vector<FrameShell *> all_frame_history_;
  CoarseInitializer *coarse_initializer_;
  Vec5 last_coarse_rmse_;

  // changed by mapper-thread. protected by map_mutex_
  boost::mutex map_mutex_;
  std::vector<FrameShell *> all_keyframes_history_;

  EnergyFunctional *ef_;
  IndexThreadReduce<Vec10> tread_reduce_;

  float *selection_map_;
  PixelSelector *pixel_selector_;
  CoarseDistanceMap *coarse_distance_map_;

  std::vector<FrameHessian *>
      frame_hessians_; // ONLY changed in marginalizeFrame and addFrame.
  std::vector<PointFrameResidual *> active_residuals_;
  float current_min_act_dist_;

  std::vector<float> all_res_vec_;

  // mutex etc. for tracker exchange.
  boost::mutex coarse_tracker_swap_mutex_;   // if tracker sees that there is a
                                             // new reference, tracker locks
                                             // [coarse_tracker_swap_mutex_] and
                                             // swaps the two.
  CoarseTracker *coarse_tracker_for_new_kf_; // set as as reference. protected
                                             // by [coarse_tracker_swap_mutex_].
  CoarseTracker *coarse_tracker_; // always used to track new frames. protected
                                  // by [track_mutex_].
  float min_id_jet_vis_tracker_, max_id_jet_vis_tracker_;
  float min_id_jet_vis_debug_, max_id_jet_vis_debug_;

  // mutex for camToWorl's in shells (these are always in a good configuration).
  boost::mutex shell_pose_mutex_;

  /*
   * tracking always uses the newest KF as reference.
   *
   */

  void makeKeyFrame(FrameHessian *fh);
  void makeNonKeyFrame(FrameHessian *fh);
  void deliverTrackedFrame(FrameHessian *fh, bool needKF);

  int last_ref_stop_id_;

  /* ========================= Scale optimization ========================== */
  std::queue<std::pair<int, cv::Mat>> stereo_id_img_queue_;
  std::vector<float> scale_errors_;
  ScaleOptimizer *scale_optimizer_;
  float optimizeScale();

  /* ============================ Loop closure ============================= */
  int prev_kf_size_; // previous kf size for increasing kf id
  LoopHandler *loop_handler_;

  /* ============================= Statistics ============================== */
  std::vector<int> pts_count_;
  std::vector<double> feature_detect_time_;
  std::vector<double> scale_opt_time_;
  std::vector<double> opt_time_;
};
} // namespace dso
