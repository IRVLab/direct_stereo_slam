#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr merge_point_clouds(
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts_query,
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts_matched,
    const Eigen::Matrix<double, 4, 4> &T_query_matched,
    const Eigen::Matrix<double, 4, 4> &T_query_matched_optimized) {
  // PCL DEBUG
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.width = pts_query.size() + 2 * pts_matched.size();
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.points.resize(cloud.width * cloud.height);
  int c_idx = 0;
  for (const auto &pt : pts_query) {
    cloud.points[c_idx].x = pt.first[0];
    cloud.points[c_idx].y = pt.first[1];
    cloud.points[c_idx].z = pt.first[2];
    cloud.points[c_idx].r = 255;
    c_idx++;
  }
  for (const auto &pt : pts_matched) {
    Eigen::Matrix<double, 4, 1> pt_query, pt_matched;
    pt_matched.head(3) = pt.first;
    pt_matched(3) = 1.0;
    pt_query = T_query_matched * pt_matched;
    cloud.points[c_idx].x = pt_query[0];
    cloud.points[c_idx].y = pt_query[1];
    cloud.points[c_idx].z = pt_query[2];
    cloud.points[c_idx].r = 255;
    cloud.points[c_idx].g = 255;
    cloud.points[c_idx].b = 255;
    c_idx++;
  }
  for (const auto &pt : pts_matched) {
    Eigen::Matrix<double, 4, 1> pt_query, pt_matched;
    pt_matched.head(3) = pt.first;
    pt_matched(3) = 1.0;
    pt_query = T_query_matched_optimized * pt_matched;
    cloud.points[c_idx].x = pt_query[0];
    cloud.points[c_idx].y = pt_query[1];
    cloud.points[c_idx].z = pt_query[2];
    cloud.points[c_idx].g = 255;
    c_idx++;
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(cloud);

  return cloud_ptr;
}