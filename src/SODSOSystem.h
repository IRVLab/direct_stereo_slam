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

#pragma once
#define MAX_ACTIVE_FRAMES 100

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
#include <fstream>
#include <iostream>

#include "opencv2/core/core.hpp"

#include <math.h>
#include <queue>

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

class SODSOSystem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SODSOSystem();
  SODSOSystem(int w, int h, const Eigen::Matrix3f &K, const SE3 &T_stereo,
              Undistort *undistorter_, float init_scale_,
              float scale_accept_th);
  virtual ~SODSOSystem();

  void addStereoImg(cv::Mat stereo_img, int stereo_id);

  // adds a new frame, and creates point & residual structs.
  void addActiveFrame(ImageAndExposure *image, int id);

  // marginalizes a frame. drops / marginalizes points & residuals.
  void marginalizeFrame(FrameHessian *frame);
  void blockUntilMappingIsFinished();

  float optimize(int mnumOptIts);

  void printResult(std::string file, std::string ba_time_file,
                   std::string scale_time_file, std::string fps_time_file);

  void debugPlot(std::string name);

  void printFrameLifetimes();
  // contains pointers to active frames

  std::vector<IOWrap::Output3DWrapper *> outputWrapper;

  bool isLost;
  bool initFailed;
  bool initialized;
  bool linearizeOperation;

  SE3 curPose;

  void setGammaFunction(float *BInv);
  void setOriginalCalib(const VecXf &originalCalib, int originalW,
                        int originalH);

private:
  CalibHessian Hcalib;

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

  // solce. eventually migrate to ef.
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

  void printLogLine();
  void printEvalLine();
  void printEigenValLine();
  std::ofstream *calibLog;
  std::ofstream *numsLog;
  std::ofstream *errorsLog;
  std::ofstream *eigenAllLog;
  std::ofstream *eigenPLog;
  std::ofstream *eigenALog;
  std::ofstream *DiagonalLog;
  std::ofstream *variancesLog;
  std::ofstream *nullspacesLog;

  std::ofstream *coarseTrackingLog;

  // statistics
  long int statistics_lastNumOptIts;
  long int statistics_numDroppedPoints;
  long int statistics_numActivatedPoints;
  long int statistics_numCreatedPoints;
  long int statistics_numForceDroppedResBwd;
  long int statistics_numForceDroppedResFwd;
  long int statistics_numMargResFwd;
  long int statistics_numMargResBwd;
  float statistics_lastFineTrackRMSE;

  // =================== changed by tracker-thread. protected by trackMutex
  // ============
  boost::mutex trackMutex;
  std::vector<FrameShell *> allFrameHistory;
  CoarseInitializer *coarseInitializer;
  Vec5 lastCoarseRMSE;

  // ================== changed by mapper-thread. protected by mapMutex
  // ===============
  boost::mutex mapMutex;
  std::vector<FrameShell *> allKeyFramesHistory;

  EnergyFunctional *ef;
  IndexThreadReduce<Vec10> treadReduce;

  float *selectionMap;
  PixelSelector *pixelSelector;
  CoarseDistanceMap *coarseDistanceMap;

  std::vector<FrameHessian *>
      frameHessians; // ONLY changed in marginalizeFrame and addFrame.
  std::vector<PointFrameResidual *> activeResiduals;
  float currentMinActDist;

  std::vector<float> allResVec;

  // mutex etc. for tracker exchange.
  boost::mutex
      coarseTrackerSwapMutex; // if tracker sees that there is a new reference,
                              // tracker locks [coarseTrackerSwapMutex] and
                              // swaps the two.
  CoarseTracker *coarseTracker_forNewKF; // set as as reference. protected by
                                         // [coarseTrackerSwapMutex].
  CoarseTracker *coarseTracker; // always used to track new frames. protected by
                                // [trackMutex].
  float minIdJetVisTracker, maxIdJetVisTracker;
  float minIdJetVisDebug, maxIdJetVisDebug;

  // mutex for camToWorl's in shells (these are always in a good configuration).
  boost::mutex shellPoseMutex;

  /*
   * tracking always uses the newest KF as reference.
   *
   */

  void makeKeyFrame(FrameHessian *fh);
  void makeNonKeyFrame(FrameHessian *fh);
  void deliverTrackedFrame(FrameHessian *fh, bool needKF);
  void mappingLoop();

  // tracking / mapping synchronization. All protected by [trackMapSyncMutex].
  boost::mutex trackMapSyncMutex;
  boost::condition_variable trackedFrameSignal;
  boost::condition_variable mappedFrameSignal;
  std::deque<FrameHessian *> unmappedTrackedFrames;
  int needNewKFAfter; // Otherwise, a new KF is *needed that has ID bigger than
                      // [needNewKFAfter]*.
  boost::thread mappingThread;
  bool runMapping;
  bool needToKetchupMapping;

  int lastRefStopID;

  std::queue<std::pair<int, cv::Mat>> stereo_list; // <id, img>
  Undistort *undistorter;
  dso::ScaleOptimizer *scaleOptimizer;

  float init_scale;
  bool scale_opt_trapped;
  void optimize_scale();

  std::vector<std::pair<int, float>> pts_optTime;
  std::vector<float> scaleOptTime;
  std::vector<float> fps_time;
};
} // namespace dso
