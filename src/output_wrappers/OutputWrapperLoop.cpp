#include "OutputWrapperLoop.h"
#include <algorithm>

#define LOOP_MARGIN 10

bool PoseCompare(const IDPose *l, const IDPose *r) {
  return l->incoming_id < r->incoming_id;
}

bool PtCompare(const IDPtIntensity *l, const IDPtIntensity *r) {
  return l->incoming_id < r->incoming_id;
}

namespace dso {
namespace IOWrap {

OutputWrapperLoop::OutputWrapperLoop()
    : previous_incoming_id(-1), pts_idx(0), lidarRange(45.0), voxelAngle(1.0),
      signature_count(0) {
  sc_ptr = new ScanContext();
  id_signature_history =
      Eigen::MatrixXd(500, 1 + 2 * sc_ptr->getSignatureSize());
}

OutputWrapperLoop::OutputWrapperLoop(double lr, double va) {
  OutputWrapperLoop();
  lidarRange = lr;
  voxelAngle = va;
}

OutputWrapperLoop::~OutputWrapperLoop() {
  for (auto pose : poses_history) {
    delete pose;
  }
  for (auto pt : pts_history) {
    delete pt;
  }
  delete sc_ptr;
}

void OutputWrapperLoop::publishKeyframes(std::vector<FrameHessian *> &frames,
                                         bool final, CalibHessian *HCalib) {
  if (!final)
    return;
  float fx = HCalib->fxl();
  float fy = HCalib->fyl();
  float cx = HCalib->cxl();
  float cy = HCalib->cyl();

  //============= Download poses and points =======================
  for (FrameHessian *fh : frames) {
    // keep incoming id increasing
    if (previous_incoming_id > fh->shell->incoming_id) {
      continue;
    }

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
  }

  //============= Preprocess points to have sphereical shape ==============
  std::vector<std::pair<Eigen::Vector3d, float>> pts_spherical;
  if (!pts_preprocess(poses_history.back(), pts_nearby, pts_history, pts_idx,
                      pts_spherical, lidarRange, voxelAngle)) {
    return;
  }

  //============= Align spherical points by PCA ===========================
  std::vector<std::pair<Eigen::Vector3d, float>> pts_spherical_aligned;
  align_points_PCA(pts_spherical, pts_spherical_aligned);

  //============= Get a signature from the aligned points by Scan Context =
  Eigen::VectorXd signature_structure, signature_intensity;
  sc_ptr->getSignature(pts_spherical_aligned, signature_structure,
                       signature_intensity, lidarRange);
  Eigen::VectorXd id_signature(1 + 2 * sc_ptr->getSignatureSize());
  id_signature << poses_history.back()->incoming_id,
      signature_structure / signature_structure.norm(),
      signature_intensity / signature_intensity.norm();

  //============= Find the closest place in history =======================
  int place_id =
      find_closest_place(id_signature.tail(2 * sc_ptr->getSignatureSize()),
                         id_signature_history, LOOP_MARGIN, sc_ptr->getWidth());

  //============= Concatenate signatures ==================================
  id_signature_history.row(signature_count++) = signature.transpose();
}
} // namespace IOWrap

} // namespace dso
