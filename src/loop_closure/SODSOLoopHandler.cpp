#include "SODSOLoopHandler.h"
#include "loop_closure/place_recognition/process_pts/pts_align.h"
#include "loop_closure/place_recognition/process_pts/pts_preprocess.h"
#include "loop_closure/place_recognition/utils/find_closest_place.h"
#include "loop_closure/place_recognition/utils/get_transformation.h"

#define LOOP_MARGIN 50

bool PoseCompare(const IDPose *l, const IDPose *r) {
  return l->incoming_id < r->incoming_id;
}

bool PtCompare(const IDPtIntensity *l, const IDPtIntensity *r) {
  return l->incoming_id < r->incoming_id;
}

namespace dso {

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
      "R: query; G: matched; B: matched transferred");
#endif
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
}

void SODSOLoopHandler::addKeyFrame(FrameHessian *fh, CalibHessian *HCalib,
                                   Mat66 poseHessian) {
  float fx = HCalib->fxl();
  float fy = HCalib->fyl();
  float cx = HCalib->cxl();
  float cy = HCalib->cyl();

  // keep incoming id increasing
  if (previous_incoming_id > fh->shell->incoming_id) {
    return;
  }

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

  previous_incoming_id = fh->shell->incoming_id;

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
    int idx, yaw_reverse;
    double difference;
    find_closest_place(signature_structure, signature_intensity,
                       signatures_structure.block(0, 0, signature_count,
                                                  signatures_structure.cols()),
                       signatures_intensity.block(0, 0, signature_count,
                                                  signatures_intensity.cols()),
                       LOOP_MARGIN, sc_ptr->getHeight(), sc_ptr->getWidth(),
                       idx, yaw_reverse, difference);
    if (difference < -5) {
      // Calculate T_query_matched
      Eigen::Matrix<double, 4, 4> T_query_matched = get_transformation(
          T_pca_rig, Ts_pca_rig, idx, yaw_reverse, sc_ptr->getWidth());
      std::cout << poses_history.back()->incoming_id << "  " << ids(idx) << " "
                << difference << std::endl;
      auto T_query_matched_optimizted = T_query_matched;
      Mat66 Hessian;
      Vec5 lastResiduals;
      int lastInners[5];
      pose_estimator->estimate(pts_spherical_history[idx],
                               affLightExposures[idx], fh, HCalib,
                               T_query_matched_optimizted, Hessian,
                               lastResiduals, lastInners, pyrLevelsUsed - 1);
      std::cout << "lastInners[0] " << lastInners[0] << std::endl;
      std::cout << "lastResiduals[0] " << lastResiduals[0] << std::endl;
      std::cout << "PoseHessian " << std::log10(poseHessian.determinant())
                << std::endl;
      std::cout << "Hessian " << std::log10(Hessian.determinant()) << std::endl;

      if (lastInners[0] > 300 && lastResiduals[0] < 12 &&
          std::log10(Hessian.determinant()) > 30) {
#if COMPARE_PCL
        auto cloud_ptr =
            merge_point_clouds(pts_spherical, pts_spherical_history[idx],
                               T_query_matched, T_query_matched_optimizted);
        pcl_viewer->showCloud(cloud_ptr);
        IOWrap::waitKey(0);
#endif
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
  ids(signature_count) = poses_history.back()->incoming_id;
  signatures_structure.row(signature_count) = signature_structure.transpose();
  signatures_intensity.row(signature_count) = signature_intensity.transpose();
  Ts_pca_rig.block<4, 4>(4 * signature_count, 0) = T_pca_rig;
  pts_spherical_history.push_back(pts_spherical);
  affLightExposures.push_back({fh->aff_g2l(), fh->ab_exposure});
  signature_count++;
}

} // namespace dso
