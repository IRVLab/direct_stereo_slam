#include "SODSOLoopHandler.h"
#include "loop_closure/place_recognition/process_pts/pts_align.h"
#include "loop_closure/place_recognition/process_pts/pts_preprocess.h"
#include "loop_closure/place_recognition/utils/find_closest_place.h"
#include "loop_closure/place_recognition/utils/get_transformation.h"

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
  signatures_structure = Eigen::MatrixXd(500, sc_ptr->getSignatureSize());
  signatures_intensity = Eigen::MatrixXd(500, sc_ptr->getSignatureSize());
  Ts_pca_rig = Eigen::MatrixXd(4 * 500, 4);

  pose_estimator = new PoseEstimator(wG[0], hG[0]);

#if COMPARE_PCL
  pcl_viewer = new pcl::visualization::CloudViewer(
      "R: query; W: matched; G: matched transferred");
#endif

  // Setup optimizer
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
  linearSolver = g2o::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *algorithm =
      new g2o::OptimizationAlgorithmLevenberg(
          g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(true);
}

SODSOLoopHandler::SODSOLoopHandler(double lr, double va) {
  SODSOLoopHandler();
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

  delete pcl_viewer;

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

  //============= Get a signature from the aligned points by Scan Context =
  Eigen::VectorXd signature_structure, signature_intensity;
  sc_ptr->getSignature(pts_spherical_aligned, signature_structure,
                       signature_intensity, lidarRange);
  signature_structure = signature_structure / signature_structure.norm(),
  signature_intensity = signature_intensity / signature_intensity.norm();

  //============= Find the closest place in history =======================
  if (signature_count > LOOP_MARGIN) {
    int idx;
    double difference, yaw;
    bool reverse;
    find_closest_place(signature_structure, signature_intensity,
                       signatures_structure.block(0, 0, signature_count,
                                                  signatures_structure.cols()),
                       signatures_intensity.block(0, 0, signature_count,
                                                  signatures_intensity.cols()),
                       LOOP_MARGIN, sc_ptr->getHeight(), sc_ptr->getWidth(),
                       idx, yaw, reverse, difference);
    if (difference < -5) {
      // Calculate T_query_matched
      Eigen::Matrix<double, 4, 4> T_query_matched = get_transformation(
          T_pca_rig, Ts_pca_rig.block<4, 4>(4 * idx, 0), yaw, reverse);
      auto T_query_matched_optimized = T_query_matched;
      Mat66 poseHessianLoop;
      Vec5 lastResiduals;
      int lastInners[5];
      pose_estimator->estimate(pts_spherical_history[idx],
                               affLightExposures[idx], fh, HCalib,
                               T_query_matched_optimized, poseHessianLoop,
                               lastResiduals, lastInners, pyrLevelsUsed - 1);
      // std::cout << "T_query_matched " << std::endl
      //           << T_query_matched << std::endl;
      // std::cout << "T_query_matched_optimized " << std::endl
      //           << T_query_matched_optimized << std::endl;
      std::cout << "lastResiduals " << lastResiduals[0] << " * "
                << lastInners[0] << std::endl;
#if COMPARE_PCL
      auto cloud_ptr =
          merge_point_clouds(pts_spherical, pts_spherical_history[idx],
                             T_query_matched, T_query_matched_optimized);
      pcl_viewer->showCloud(cloud_ptr);
      IOWrap::waitKey(0);
#endif

      if (lastInners[0] > 300 && lastResiduals[0] < 12) {
        std::cout << "Adding loop constraint" << std::endl;
        // Connection to detected loop closure keyframe
        g2o::VertexSE3 *vfh_loop = (g2o::VertexSE3 *)optimizer.vertex(ids(idx));
        g2o::EdgeSE3 *edgeFromLoop = new g2o::EdgeSE3();
        edgeFromLoop->setVertex(0, vfh_loop);
        edgeFromLoop->setVertex(1, vfh);
        // T_query_matched_optimized.setIdentity();
        edgeFromLoop->setMeasurement(
            g2o::SE3Quat(T_query_matched_optimized.block<3, 3>(0, 0),
                         T_query_matched_optimized.block<3, 1>(0, 3)));
        // poseHessianLoop.setIdentity();
        edgeFromLoop->setInformation(hessian_quat_from_euler(
            poseHessianLoop, edgeFromLoop->measurement()));
        edgeFromLoop->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edgeFromLoop);
      }
    }
  }

  //============= Concatenate signatures ==================================
  if (ids.rows() <= signature_count) {
    ids.conservativeResize(ids.rows() + 500);
    signatures_structure.conservativeResize(signatures_structure.rows() + 500,
                                            signatures_structure.cols());
    signatures_intensity.conservativeResize(signatures_intensity.rows() + 500,
                                            signatures_intensity.cols());
    Ts_pca_rig.conservativeResize(Ts_pca_rig.rows() + 4 * 500,
                                  Ts_pca_rig.cols());
  }
  ids(signature_count) = fh->shell->incoming_id;
  signatures_structure.row(signature_count) = signature_structure.transpose();
  signatures_intensity.row(signature_count) = signature_intensity.transpose();
  Ts_pca_rig.block<4, 4>(4 * signature_count, 0) = T_pca_rig;
  pts_spherical_history.push_back(pts_spherical);
  affLightExposures.push_back({fh->aff_g2l(), fh->ab_exposure});
  signature_count++;
}

} // namespace dso
