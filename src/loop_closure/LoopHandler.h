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

#pragma once
#include <boost/thread.hpp>
#include <chrono>
#include <flann/flann.hpp>
#include <queue>

#include <g2o/core/block_solver.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include "loop_closure/loop_detection/ScanContext.h"
#include "loop_closure/pangolin_viewer/PangolinLoopViewer.h"
#include "loop_closure/pose_estimation/PoseEstimator.h"

typedef std::vector<std::chrono::duration<long int, std::ratio<1, 1000000000>>>
    TimeVector;

// normalize dso errors to roughly around 1.0
#define DSO_ERROR_SCALE 5.0
#define SCALE_ERROR_SCALE 0.1
#define DIRECT_ERROR_SCALE 0.1
#define ICP_ERROR_SCALE 1.0

// the rotation estimated by DSO is much more accurate than translation
#define POSE_R_WEIGHT 1e4

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

struct LoopEdge {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id_from;
  g2o::SE3Quat measurement;
  Mat66 information;

  LoopEdge(int i, g2o::SE3Quat tfm_t_f, float pose_error, float scale_error)
      : id_from(i), measurement(tfm_t_f) {
    information.setIdentity();
    information *= (1.0 / pose_error);
    information.topLeftCorner<3, 3>() *=
        scale_error > 0 ? (1.0 / scale_error) : 1e-9;
    information.bottomRightCorner<3, 3>() *= POSE_R_WEIGHT;
  }
};

struct LoopFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int kf_id;                      // kF id, for pose graph and visualization
  int incoming_id;                // increasing id, for ground truth
  g2o::SE3Quat tfm_w_c;           // coordinate in pose graph
  Eigen::Vector3d trans_w_c_orig; // original pose for logging
  std::vector<LoopEdge *> edges;  // pose edges associated with current frame

  // loop detection
  SigType signature;           // place signature
  Eigen::Matrix4d tfm_pca_rig; // transformation from rig to pca frame

  // loop correction by dso
  std::vector<std::pair<Eigen::Vector3d, float *>> pts_dso;
  FrameHessian *fh;
  std::vector<float> cam;
  float ab_exposure;

  // loop correction by icp
  std::vector<Eigen::Vector3d> pts_spherical;

  // heuristics for setting edge information
  float dso_error;
  float scale_error;

  // whether has been added to global pose graph
  bool graph_added;

  LoopFrame(FrameHessian *fh,
            const std::vector<std::pair<Eigen::Vector3d, float *>> &pd,
            const std::vector<float> &cam,
            const std::vector<Eigen::Vector3d> &ps, float de, float se)
      : fh(fh), pts_dso(pd), kf_id(fh->frameID),
        incoming_id(fh->shell->incoming_id),
        tfm_w_c(g2o::SE3Quat(fh->shell->camToWorld.rotationMatrix(),
                             fh->shell->camToWorld.translation())),
        trans_w_c_orig(tfm_w_c.translation()), cam(cam),
        ab_exposure(fh->ab_exposure), pts_spherical(ps),
        dso_error(de * DSO_ERROR_SCALE), scale_error(se * SCALE_ERROR_SCALE),
        graph_added(false) {}

  ~LoopFrame() {
    for (auto &edge : edges) {
      delete edge;
    }

    for (auto &p : pts_dso) {
      delete p.second;
    }
  }
};

class LoopHandler {
public:
  LoopHandler(float lidar_range, float scan_context_thres,
              IOWrap::PangolinLoopViewer *pangolin_viewer);
  ~LoopHandler();

  void publishKeyframes(FrameHessian *fh, CalibHessian *HCalib, float dso_error,
                        float scale_error);
  void join();

  void savePose();

  // statistics
  TimeVector pts_generation_time_;
  TimeVector sc_generation_time_;
  TimeVector search_ringkey_time_;
  TimeVector search_sc_time_;
  TimeVector direct_est_time_;
  TimeVector icp_time_;
  TimeVector opt_time_;
  int direct_loop_count_;
  int icp_loop_count_;

private:
  int cur_id_;
  bool running_;
  boost::thread run_thread_;
  void run();

  // loop detection by ScanContext
  float lidar_range_;
  float scan_context_thres_;
  std::unordered_map<int, Eigen::Matrix<double, 6, 1>> id_pose_wc_;
  std::vector<std::pair<int, Eigen::Vector3d>> pts_nearby_;
  flann::Index<flann::L2<float>> *ringkeys_;
  ScanContext *sc_ptr_;

  // loop correction by direct alignment
  PoseEstimator *pose_estimator_;

  // pose graph
  boost::mutex loop_frame_queue_mutex_;
  std::queue<LoopFrame *> loop_frame_queue_;
  std::vector<LoopFrame *> loop_frames_;
  g2o::SparseOptimizer pose_optimizer_;
  IOWrap::PangolinLoopViewer *pangolin_viewer_;
  void optimize();
};

} // namespace dso
