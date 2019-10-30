#include "SODSOLoopHandler.h"
#include "loop_closure/place_recognition/process_pts/pts_align.h"
#include "loop_closure/place_recognition/process_pts/pts_preprocess.h"
#include "loop_closure/place_recognition/utils/find_closest_place.h"
#include "loop_closure/place_recognition/utils/get_transformation.h"
#include "loop_closure/place_recognition/utils/icp.h"
#include "loop_closure/place_recognition/utils/merge_point_clouds.h"

#include <fstream>

#define LOOP_MARGIN 50

bool PoseCompare(const IDPose *l, const IDPose *r) {
  return l->incoming_id < r->incoming_id;
}

bool PtCompare(const IDPtIntensity *l, const IDPtIntensity *r) {
  return l->incoming_id < r->incoming_id;
}

namespace dso {

/** conversion code from Euler angles */
Eigen::Matrix<double, 6, 6, Eigen::ColMajor> hessian_quat_from_euler(
    Eigen::Matrix<double, 6, 6, Eigen::ColMajor> &Hess_euler,
    const g2o::Isometry3 &t) {
  double delta = 1e-6;
  double idelta = 1 / (2 * delta);

  Eigen::Matrix<double, 7, 1> t0 = g2o::internal::toVectorQT(t);
  Eigen::Matrix<double, 7, 1> ta = t0;
  Eigen::Matrix<double, 7, 1> tb = t0;

  Eigen::Matrix<double, 6, 6, Eigen::ColMajor> Jac;
  for (int i = 0; i < 6; i++) {
    ta = tb = t0;
    ta[i] -= delta;
    tb[i] += delta;
    Eigen::Matrix<double, 6, 1> ea =
        g2o::internal::toVectorET(g2o::internal::fromVectorQT(ta));
    Eigen::Matrix<double, 6, 1> eb =
        g2o::internal::toVectorET(g2o::internal::fromVectorQT(tb));
    Jac.col(i) = (eb - ea) * idelta;
  }

  return Jac.transpose() * Hess_euler * Jac;
}

SODSOLoopHandler::SODSOLoopHandler()
    : previous_incoming_id(-1), pts_idx(0), lidarRange(45.0), voxelAngle(1.0),
      signature_count(0) {
  sc_ptr = new ScanContext();
  ids = Eigen::VectorXi(500, 1);
  ring_keys = Eigen::MatrixXd(500, sc_ptr->getHeight());
  signatures_structure = Eigen::MatrixXd(500, sc_ptr->getSignatureSize());
  signatures_intensity = Eigen::MatrixXd(500, sc_ptr->getSignatureSize());
  Ts_pca_rig = Eigen::MatrixXd(4 * 500, 4);

  pose_estimator = new PoseEstimator(wG[0], hG[0]);

  // Setup optimizer
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
  linearSolver = g2o::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *algorithm =
      new g2o::OptimizationAlgorithmLevenberg(
          g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(true);

  pcl_viewer = new pcl::visualization::CloudViewer(
      "R: current; G: matched(icp); W: matched(dso)");
}

SODSOLoopHandler::SODSOLoopHandler(float lr, float va) : SODSOLoopHandler() {
  lidarRange = lr;
  voxelAngle = va;
}

SODSOLoopHandler::~SODSOLoopHandler() {
  for (auto pose : poses_history) {
    delete pose;
  }
  for (auto pt : pts_history) {
    delete pt;
  }
  delete sc_ptr;

  delete pose_estimator;

  g2o::VertexSE3 *vlast =
      (g2o::VertexSE3 *)optimizer.vertex(previous_incoming_id);
  vlast->setFixed(true);
  optimizer.save("pose_graph.g2o");
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  optimizer.computeInitialGuess();
  optimizer.optimize(20);
  optimizer.save("pose_graph_optimized.g2o");
  std::cout << "Saved!" << std::endl;
}

void SODSOLoopHandler::addKeyFrame(FrameHessian *fh, CalibHessian *HCalib,
                                   Mat66 poseHessian) {
  // keep incoming id increasing
  if (previous_incoming_id > fh->shell->incoming_id) {
    return;
  }

  // Add new vertex to pose graph
  g2o::VertexSE3 *vfh = new g2o::VertexSE3();
  SE3 fh_wc = fh->shell->camToWorld;
  vfh->setEstimate(g2o::SE3Quat(fh_wc.rotationMatrix(), fh_wc.translation()));
  vfh->setId(fh->shell->incoming_id);
  optimizer.addVertex(vfh);

  // Connection to previous keyframe
  if (previous_incoming_id >= 0 && !pts_spherical_history.empty()) {
    g2o::VertexSE3 *vfh_prv =
        (g2o::VertexSE3 *)optimizer.vertex(previous_incoming_id);
    g2o::EdgeSE3 *edgeToPrv = new g2o::EdgeSE3();
    edgeToPrv->setVertex(0, vfh_prv);
    edgeToPrv->setVertex(1, vfh);
    edgeToPrv->setMeasurementFromState();
    // poseHessian.setIdentity();
    edgeToPrv->setInformation(
        hessian_quat_from_euler(poseHessian, edgeToPrv->measurement()));
    edgeToPrv->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(edgeToPrv);
  }

  previous_incoming_id = fh->shell->incoming_id;

  // Loop closure
  float fx = HCalib->fxl();
  float fy = HCalib->fyl();
  float cx = HCalib->cxl();
  float cy = HCalib->cyl();

  //============= Download poses and points =======================
  for (PointHessian *p : fh->pointHessiansMarginalized) {
    float ave_intensity = 0;
    for (int i = 0; i < patternNum; i++)
      ave_intensity += p->color[i];
    ave_intensity /= patternNum;

    Eigen::Vector4d p_l((p->u - cx) / fx / p->idepth_scaled,
                        (p->v - cy) / fy / p->idepth_scaled,
                        1 / p->idepth_scaled, 1);
    Eigen::Vector3d p_g = fh->shell->camToWorld.matrix3x4() * p_l;
    pts_history.push_back(
        new IDPtIntensity(fh->shell->incoming_id, p_g, ave_intensity));
  }
  poses_history.push_back(new IDPose(
      fh->shell->incoming_id, fh->shell->camToWorld.inverse().matrix3x4()));

  //============= Preprocess points to have sphereical shape ==============
  std::vector<std::pair<Eigen::Vector3d, float>> pts_spherical;
  if (!pts_preprocess(poses_history.back(), pts_nearby, pts_history, pts_idx,
                      pts_spherical, lidarRange, voxelAngle)) {
    return;
  }

  //============= Align spherical points by PCA ===========================
  std::vector<std::pair<Eigen::Vector3d, float>> pts_spherical_aligned;
  Eigen::Matrix<double, 4, 4> T_pca_rig;
  align_points_PCA(pts_spherical, pts_spherical_aligned, T_pca_rig);

  //=== Get ringkey and signature from the aligned points by Scan Context =
  Eigen::VectorXd ring_key, signature_structure, signature_intensity;
  sc_ptr->getSignature(pts_spherical_aligned, ring_key, signature_structure,
                       signature_intensity, lidarRange);

  //============= Find the closest place in history =======================
  if (signature_count > LOOP_MARGIN) {
    std::vector<int> indexes;
    find_closest_place_ring_key(
        ring_key, ring_keys.block(0, 0, signature_count, ring_keys.cols()),
        LOOP_MARGIN, 0.01, indexes);
    if (!indexes.empty()) {
      int idx;
      double difference, yaw;
      bool reverse;
      find_closest_place_sc(
          signature_structure, signature_intensity,
          signatures_structure.block(0, 0, signature_count,
                                     signatures_structure.cols()),
          signatures_intensity.block(0, 0, signature_count,
                                     signatures_intensity.cols()),
          LOOP_MARGIN, sc_ptr->getHeight(), sc_ptr->getWidth(), indexes, idx,
          yaw, reverse, difference);
      // std::cout << "find_closest_place_sc " << difference << std::endl;
      if (difference < 0.2) {
        auto pc_ptr = create_point_clouds(pts_spherical, {255, 0, 0});
        // Calculate T_query_matched
        Eigen::Matrix<double, 4, 4> T_query_matched = get_transformation(
            T_pca_rig, Ts_pca_rig.block<4, 4>(4 * idx, 0), yaw, reverse);

        auto T_query_matched_icp = T_query_matched;
        double icp_score =
            icp(pts_spherical, pts_spherical_history[idx], T_query_matched_icp);
        std::cout << "icp_score " << icp_score << std::endl;
        merge_point_clouds(pc_ptr, pts_spherical_history[idx],
                           T_query_matched_icp, {0, 255, 0});

        if (icp_score < 2) {
          Mat66 poseHessianLoopInit, poseHessianLoopLast;
          Vec5 lastResiduals;
          Vec5 lastInners;
          auto T_query_matched_dso = T_query_matched_icp;
          pose_estimator->estimate(
              pts_spherical_history[idx], affLightExposures[idx], fh, HCalib,
              T_query_matched_dso, poseHessianLoopInit, poseHessianLoopLast,
              lastResiduals, lastInners, pyrLevelsUsed - 1);
          if (!poseHessianLoopInit
                   .allFinite()) { // true when no point < cutoffTH
            poseHessianLoopInit.setIdentity();
          }
          if (!poseHessianLoopLast
                   .allFinite()) { // true when no point < cutoffTH
            poseHessianLoopLast.setIdentity();
          }

          auto T_delta = T_query_matched_dso.inverse() * T_query_matched_icp;
          Sophus::SE3 se3_delta(T_delta.block<3, 3>(0, 0),
                                T_delta.block<3, 1>(0, 3));
          std::cout << "tranlation " << se3_delta.translation().norm()
                    << " rotation " << se3_delta.so3().log().norm()
                    << std::endl;
          std::cout << "lastResiduals " << lastResiduals[0] << " * "
                    << lastInners[0] << std::endl;
          merge_point_clouds(pc_ptr, pts_spherical_history[idx],
                             T_query_matched_dso, {255, 255, 255});

          bool close_pose =
              (icp_score < 1) ||
              (lastInners[0] > 1e-3 && (se3_delta.translation().norm() < 0.3 &&
                                        se3_delta.so3().log().norm() < 0.02));
          bool good_align = lastInners[0] > 0.6 && lastResiduals[0] < 12;
          std::cout << "close_pose: " << close_pose
                    << " good_align: " << good_align << std::endl;
          if (close_pose || good_align) {
            std::cout << "Adding loop constraint" << std::endl;
            auto T_final =
                good_align ? T_query_matched_dso : T_query_matched_icp;
            auto H_final =
                good_align ? poseHessianLoopLast : poseHessianLoopInit;
            // Connection to detected loop closure keyframe
            g2o::VertexSE3 *vfh_loop =
                (g2o::VertexSE3 *)optimizer.vertex(ids(idx));
            g2o::EdgeSE3 *edgeFromLoop = new g2o::EdgeSE3();
            edgeFromLoop->setVertex(0, vfh);
            edgeFromLoop->setVertex(1, vfh_loop);
            edgeFromLoop->setMeasurement(g2o::SE3Quat(
                T_final.block<3, 3>(0, 0), T_final.block<3, 1>(0, 3)));
            auto H_quat =
                hessian_quat_from_euler(H_final, edgeFromLoop->measurement());
            edgeFromLoop->setInformation(H_quat);
            edgeFromLoop->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edgeFromLoop);
          }
          pcl_viewer->showCloud(pc_ptr);
          // IOWrap::waitKey(0);
        }
      }
    }
  }

  //============= Concatenate signatures ==================================
  if (ids.rows() <= signature_count) {
    ids.conservativeResize(ids.rows() + 500);
    ring_keys.conservativeResize(ring_keys.rows() + 500, ring_keys.cols());
    signatures_structure.conservativeResize(signatures_structure.rows() + 500,
                                            signatures_structure.cols());
    signatures_intensity.conservativeResize(signatures_intensity.rows() + 500,
                                            signatures_intensity.cols());
    Ts_pca_rig.conservativeResize(Ts_pca_rig.rows() + 4 * 500,
                                  Ts_pca_rig.cols());
  }
  ids(signature_count) = fh->shell->incoming_id;
  ring_keys.row(signature_count) = ring_key.transpose();
  signatures_structure.row(signature_count) = signature_structure.transpose();
  signatures_intensity.row(signature_count) = signature_intensity.transpose();
  Ts_pca_rig.block<4, 4>(4 * signature_count, 0) = T_pca_rig;
  pts_spherical_history.push_back(pts_spherical);
  affLightExposures.push_back({fh->aff_g2l(), fh->ab_exposure});
  signature_count++;
}

} // namespace dso
