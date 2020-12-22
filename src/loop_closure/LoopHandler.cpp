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

#include "LoopHandler.h"
// loop closure helpers
#include "loop_closure/process_pts/generate_spherical_points.h"
#include "loop_closure/process_pts/icp.h"
#include "loop_closure/scan_context/search_place.h"
#include "timing.h"

#include <fstream>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

namespace dso {

LoopHandler::LoopHandler(float lidar_range, float scan_context_thres,
                         IOWrap::PangolinLoopViewer *pangolin_viewer)
    : cur_id_(-1), lidar_range_(lidar_range),
      scan_context_thres_(scan_context_thres),
      pangolin_viewer_(pangolin_viewer) {
  // place recognition
  sc_ptr_ = new ScanContext();
  flann::Matrix<float> init_data(new float[sc_ptr_->getHeight()], 1,
                                 sc_ptr_->getHeight());
  ringkeys_ = new flann::Index<flann::L2<float>>(init_data,
                                                 flann::KDTreeIndexParams(4));
  ringkeys_->buildIndex();

  pose_estimator_ = new PoseEstimator(wG[0], hG[0]);

  // setup pose_optimizer_
  std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver;
  linearSolver = g2o::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *algorithm =
      new g2o::OptimizationAlgorithmLevenberg(
          g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
  pose_optimizer_.setAlgorithm(algorithm);

  running_ = true;
  run_thread_ = boost::thread(&LoopHandler::run, this);

  direct_loop_count_ = 0;
  icp_loop_count_ = 0;
}

void LoopHandler::printTimeStatAndSavePose() {
  running_ = false;

  std::cout << "======Loop Closure Time======= " << std::endl;
  std::cout << "pts_generation " << 1000 * average(pts_generation_time_)
            << " * " << pts_generation_time_.size() << std::endl;
  std::cout << "sc_generation " << 1000 * average(sc_generation_time_) << " * "
            << sc_generation_time_.size() << std::endl;
  std::cout << "search_ringkey " << 1000 * average(search_ringkey_time_)
            << " * " << search_ringkey_time_.size() << std::endl;
  std::cout << "search_sc " << 1000 * average(search_sc_time_) << " * "
            << search_sc_time_.size() << std::endl;
  std::cout << "direct_est " << 1000 * average(direct_est_time_) << " * "
            << direct_est_time_.size() << std::endl;
  std::cout << "icp " << 1000 * average(icp_time_) << " * " << icp_time_.size()
            << std::endl;
  std::cout << "pose_graph_opt " << 1000 * average(opt_time_) << " * "
            << opt_time_.size() << std::endl;
  std::cout << "direct_loop_count " << direct_loop_count_ << std::endl;
  std::cout << "icp_loop_count " << icp_loop_count_ << std::endl;
  std::cout << "============================== " << std::endl;

  // Export the final pose graph
  std::ofstream posefile;
  posefile.open("id_pose.txt");
  posefile << std::setprecision(6);
  for (auto &lf : loop_frames_) {
    posefile << lf->incoming_id << " ";

    auto t_wc = lf->tfm_w_c.translation();
    posefile << t_wc(0) << " " << t_wc(1) << " " << t_wc(2) << std::endl;
  }
  posefile.close();
}

LoopHandler::~LoopHandler() {
  running_ = false;
  run_thread_.join();

  delete ringkeys_;
  delete sc_ptr_;

  delete pose_estimator_;

  for (LoopFrame *lf : loop_frames_) {
    delete lf;
  }
}

void LoopHandler::join() {
  run_thread_.join();
  printf("JOINED LoopHandler thread!\n");
}

void LoopHandler::optimize() {
  if (loop_frames_.empty()) {
    return;
  }

  // pose_optimizer_.clear();
  // Vertices
  for (LoopFrame *lf : loop_frames_) {
    if (lf->graph_added) {
      continue;
    }

    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(lf->id);
    v->setEstimate(lf->tfm_w_c);
    pose_optimizer_.addVertex(v);
    lf->graph_added = true;

    // no constraint for the first node of each new sequence
    if (lf->dso_error < -1) {
      continue;
    }

    // Edges
    for (LoopEdge *le : lf->edges) {
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      e->setVertex(0, v);
      e->setVertex(1, pose_optimizer_.vertex(le->id_from));
      e->setMeasurement(le->measurement);
      e->setInformation(le->information);
      e->setRobustKernel(new g2o::RobustKernelHuber());
      pose_optimizer_.addEdge(e);
    }
  }

  // fix last vertex
  pose_optimizer_.vertex(loop_frames_.back()->id)->setFixed(true);

  pose_optimizer_.setVerbose(false);
  pose_optimizer_.initializeOptimization();
  pose_optimizer_.computeInitialGuess();
  pose_optimizer_.optimize(25);
}

void LoopHandler::publishKeyframes(FrameHessian *fh, CalibHessian *HCalib,
                                   float dso_error, float scale_error) {
  int prv_id = cur_id_;

  // keep id increasing
  if (prv_id >= fh->frameID) {
    return;
  }

  cur_id_ = fh->frameID;
  SE3 cur_wc = fh->shell->camToWorld;
  float fx = HCalib->fxl();
  float fy = HCalib->fyl();
  float cx = HCalib->cxl();
  float cy = HCalib->cyl();

  // points for loop closure
  std::vector<std::pair<Eigen::Vector3d, float *>> pts_dso;
  std::vector<Eigen::Vector3d> pts_spherical;

  if (lidar_range_ > 0 && scale_error > 0) {
    /* ====================== Extract points ================================ */
    for (PointHessian *p : fh->pointHessiansMarginalized) {
      Eigen::Vector4d p_l((p->u - cx) / fx / p->idepth_scaled,
                          (p->v - cy) / fy / p->idepth_scaled,
                          1 / p->idepth_scaled, 1);
      Eigen::Vector3d p_g = cur_wc.matrix3x4() * p_l;
      pts_nearby_.emplace_back(std::pair<int, Eigen::Vector3d>(cur_id_, p_g));

      float *dIp = new float[PYR_LEVELS];
      for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        float u = (p->u + 0.5) / ((int)1 << lvl) - 0.5;
        float v = (p->v + 0.5) / ((int)1 << lvl) - 0.5;
        dIp[lvl] = getInterpolatedElement31(fh->dIp[lvl], u, v, wG[lvl]);
      }
      pts_dso.emplace_back(
          std::pair<Eigen::Vector3d, float *>(p_l.head<3>(), dIp));
    }

    /* ============= Preprocess points to have sphereical shape ============= */
    id_pose_wc_[cur_id_] = cur_wc.log();
    auto t0 = std::chrono::steady_clock::now();
    generate_spherical_points(pts_nearby_, id_pose_wc_, cur_wc.inverse(),
                              lidar_range_, pts_spherical);
    auto t1 = std::chrono::steady_clock::now();
    pts_generation_time_.emplace_back(duration(t0, t1));
  }

  LoopFrame *cur_frame = new LoopFrame(
      cur_id_, fh->shell->incoming_id, cur_wc, pts_dso, fh, {fx, fy, cx, cy},
      fh->ab_exposure, pts_spherical, dso_error, scale_error);
  boost::unique_lock<boost::mutex> lk_lfq(loop_frame_queue_mutex_);
  loop_frame_queue_.emplace(cur_frame);
}

void LoopHandler::run() {
  std::cout << "Start Loop Thread" << std::endl;
  while (running_) {
    boost::unique_lock<boost::mutex> lk_lfq(loop_frame_queue_mutex_);
    if (loop_frame_queue_.empty()) {
      lk_lfq.unlock();
      usleep(5000);
      continue;
    }
    LoopFrame *cur_frame = loop_frame_queue_.front();
    loop_frame_queue_.pop();

    // for Pangolin visualization
    std::vector<Eigen::Vector3d> lidar_pts = cur_frame->pts_spherical;

    loop_frames_.emplace_back(cur_frame);
    // Connection to previous keyframe
    if (loop_frames_.size() > 1) {
      auto prv_frame = loop_frames_[loop_frames_.size() - 2];
      Eigen::Matrix4d tfm_prv_cur =
          (prv_frame->tfm_w_c.inverse() * cur_frame->tfm_w_c)
              .to_homogeneous_matrix();
      cur_frame->edges.emplace_back(
          new LoopEdge(prv_frame->id, tfm_prv_cur.inverse(),
                       cur_frame->dso_error, cur_frame->scale_error));
    }

    // Loop closure is disabled if lidar_range_ < 0 or scale optimization failed
    if (lidar_range_ < 0 || cur_frame->scale_error < 0) {
      delete cur_frame->fh;
      usleep(5000);
      continue;
    }

    /* == Get ringkey and signature from the aligned points by Scan Context = */
    flann::Matrix<float> ringkey;
    SigType signature;
    Eigen::Matrix4d tfm_pca_rig;
    auto t0 = std::chrono::steady_clock::now();
    sc_ptr_->generate(cur_frame->pts_spherical, ringkey, signature,
                      lidar_range_, tfm_pca_rig);
    auto t1 = std::chrono::steady_clock::now();
    sc_generation_time_.emplace_back(duration(t0, t1));
    cur_frame->tfm_pca_rig = tfm_pca_rig;
    cur_frame->signature = signature;

    /* ======================== Loop Closure ================================ */
    // fast search by ringkey
    std::vector<int> ringkey_candidates;
    t0 = std::chrono::steady_clock::now();
    search_ringkey(ringkey, ringkeys_, ringkey_candidates);
    t1 = std::chrono::steady_clock::now();
    search_ringkey_time_.emplace_back(duration(t0, t1));

    if (!ringkey_candidates.empty()) {
      // search by ScanContext
      t0 = std::chrono::steady_clock::now();
      int matched_idx;
      float sc_diff;
      search_sc(signature, loop_frames_, ringkey_candidates,
                sc_ptr_->getWidth(), matched_idx, sc_diff);
      t1 = std::chrono::steady_clock::now();
      search_sc_time_.emplace_back(duration(t0, t1));

      if (sc_diff < scan_context_thres_) {
        auto matched_frame = loop_frames_[matched_idx];
        printf("%d - %d SC: %.3f ", cur_id_, matched_frame->id, sc_diff);

        // calculate the initial tfm_matched_cur from ScanContext
        Eigen::Matrix4d tfm_cur_matched =
            tfm_pca_rig.inverse() * matched_frame->tfm_pca_rig;

        // first try direct alignment
        Eigen::Matrix4d tfm_cur_matched_direct = tfm_cur_matched;
        t0 = std::chrono::steady_clock::now();
        bool direct_succ = pose_estimator_->estimate(
            matched_frame->pts_dso, matched_frame->ab_exposure, cur_frame->fh,
            cur_frame->cam, tfm_cur_matched_direct, pyrLevelsUsed - 1);
        t1 = std::chrono::steady_clock::now();
        direct_est_time_.emplace_back(duration(t0, t1));

        // try icp if direct alignment failed
        bool icp_succ = false;
        Eigen::Matrix4d tfm_cur_matched_icp = tfm_cur_matched;
        if (!direct_succ) {
          t0 = std::chrono::steady_clock::now();
          icp_succ = icp(matched_frame->pts_spherical, tfm_cur_matched_icp,
                         cur_frame->pts_spherical);
          t1 = std::chrono::steady_clock::now();
          icp_time_.emplace_back(duration(t0, t1));
        }

        if (direct_succ || icp_succ) {
          if (direct_succ) {
            direct_loop_count_++;
            tfm_cur_matched = tfm_cur_matched_direct;
          } else {
            icp_loop_count_++;
            tfm_cur_matched = tfm_cur_matched_icp;
          }
          printf("add loop\n");

          // add the loop constraint
          cur_frame->edges.emplace_back(
              new LoopEdge(matched_frame->id, tfm_cur_matched,
                           cur_frame->dso_error, cur_frame->scale_error));

          // run pose graph optimization
          t0 = std::chrono::steady_clock::now();
          optimize();
          t1 = std::chrono::steady_clock::now();
          opt_time_.emplace_back(duration(t0, t1));

          // update the trajectory
          for (auto &lf : loop_frames_) {
            g2o::VertexSE3 *v =
                (g2o::VertexSE3 *)pose_optimizer_.vertex(lf->id);
            auto new_tfm_wc = v->estimate().matrix();
            lf->tfm_w_c = g2o::SE3Quat(new_tfm_wc.block<3, 3>(0, 0),
                                       new_tfm_wc.block<3, 1>(0, 3));
            if (pangolin_viewer_) {
              pangolin_viewer_->modifyKeyframePoseByKFID(lf->id,
                                                         SE3(new_tfm_wc));
            }
          }

          // merge matech lidar pts for Pangolin visualization
          Eigen::Matrix<double, 4, 1> pt_matched, pt_cur;
          pt_matched(3) = 1.0;
          for (size_t i = 0; i < matched_frame->pts_spherical.size(); i++) {
            pt_matched.head(3) = matched_frame->pts_spherical[i];
            pt_cur = tfm_cur_matched * pt_matched;
            lidar_pts.push_back({pt_cur(0), pt_cur(1), pt_cur(2)});
          }
        } else {
          printf("\n");
        }
      }
    }
    pangolin_viewer_->refreshLidarData(lidar_pts,
                                       cur_frame->pts_spherical.size());
    delete cur_frame->fh;
    usleep(5000);
  }
  std::cout << "Finished Loop Thread" << std::endl;
}

} // namespace dso