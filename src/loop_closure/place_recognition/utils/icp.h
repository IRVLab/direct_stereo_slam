#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <chrono>

inline double
icp(const std::vector<std::pair<Eigen::Vector3d, float>> &pts_target,
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts_source,
    Eigen::Matrix<double, 4, 4> &T_target_source) {

  pcl::PointCloud<pcl::PointXYZI> pc_target;
  pc_target.width = pts_target.size();
  pc_target.height = 1;
  pc_target.is_dense = false;
  pc_target.points.resize(pts_target.size());

  pcl::PointCloud<pcl::PointXYZI> pc_target_source;
  pc_target_source.width = pts_source.size();
  pc_target_source.height = 1;
  pc_target_source.is_dense = false;
  pc_target_source.points.resize(pts_source.size());

  for (int i = 0; i < pts_target.size(); i++) {
    pc_target.points[i].x = pts_target[i].first[0];
    pc_target.points[i].y = pts_target[i].first[1];
    pc_target.points[i].z = pts_target[i].first[2];
    pc_target.points[i].intensity = pts_target[i].second;
  }

  Eigen::Matrix<double, 4, 1> pt_source, pt_target;
  for (int i = 0; i < pts_source.size(); i++) {
    pt_source.head(3) = pts_source[i].first;
    pt_source(3) = 1.0;
    pt_target = T_target_source * pt_source;
    pc_target_source.points[i].x = pt_target[0];
    pc_target_source.points[i].y = pt_target[1];
    pc_target_source.points[i].z = pt_target[2];
    pc_target_source.points[i].intensity = pts_source[i].second;
  }

  auto pc_target_ptr =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(pc_target);
  auto pc_target_source_ptr =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(pc_target_source);

  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setInputSource(pc_target_source_ptr);
  icp.setInputTarget(pc_target_ptr);
  pcl::PointCloud<pcl::PointXYZI> pc_icp;
  //   auto t0 = std::chrono::high_resolution_clock::now();
  icp.align(pc_icp);
  //   auto t1 = std::chrono::high_resolution_clock::now();
  //   double ttOpt = std::chrono::duration<double>(t1 - t0).count();
  //   std::cout << "icp time: " << ttOpt << std::endl;
  T_target_source =
      icp.getFinalTransformation().cast<double>() * T_target_source;

  return icp.getFitnessScore();
}