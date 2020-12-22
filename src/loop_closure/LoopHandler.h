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
#include <flann/flann.hpp>
#include <g2o/core/block_solver.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <queue>

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include "loop_closure/pangolin_viewer/PangolinLoopViewer.h"
#include "loop_closure/pose_estimation/PoseEstimator.h"
#include "loop_closure/scan_context/ScanContext.h"

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

struct LoopEdge {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id_from;
  g2o::SE3Quat measurement;
  Mat66 information;

  LoopEdge(int i, Eigen::Matrix4d tfm_t_f, float dso_error, float scale_error)
      : id_from(i), measurement(g2o::SE3Quat(tfm_t_f.block<3, 3>(0, 0),
                                             tfm_t_f.block<3, 1>(0, 3))) {
    // heuristically set information matrix by errors
    information.setIdentity();
    if (dso_error > 0 && scale_error > 0) {
      information *= (1.0 / dso_error);
      information.topLeftCorner<3, 3>() *= (1.0 / scale_error);
    } else {
      information *= 0;
    }
  }
};

struct LoopFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;                        // id for pose graph and visualization
  int incoming_id;               // to find ground truth
  g2o::SE3Quat tfm_w_c;          // coordinate in pose graph
  std::vector<LoopEdge *> edges; // pose edges associated with current frame

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

  LoopFrame(int i, int ii, const dso::SE3 &tfm_w_c,
            const std::vector<std::pair<Eigen::Vector3d, float *>> &pd,
            FrameHessian *f, const std::vector<float> &c, float ae,
            const std::vector<Eigen::Vector3d> &ps, float de, float se)
      : id(i), incoming_id(ii),
        tfm_w_c(g2o::SE3Quat(tfm_w_c.rotationMatrix(), tfm_w_c.translation())),
        pts_dso(pd), fh(f), cam(c), ab_exposure(ae), pts_spherical(ps),
        dso_error(de), scale_error(se), graph_added(false) {}

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
  void printTimeStatAndSavePose();

  void publishKeyframes(FrameHessian *fh, CalibHessian *HCalib, float dso_error,
                        float scale_error);
  void join();

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

  // statistics
  std::vector<double> pts_generation_time_;
  std::vector<double> sc_generation_time_;
  std::vector<double> search_ringkey_time_;
  std::vector<double> search_sc_time_;
  std::vector<double> direct_est_time_;
  std::vector<double> icp_time_;
  std::vector<double> opt_time_;
  int direct_loop_count_;
  int icp_loop_count_;
};

} // namespace dso
