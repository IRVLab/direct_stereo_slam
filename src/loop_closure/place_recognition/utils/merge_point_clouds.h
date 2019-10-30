#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr
create_point_clouds(const std::vector<std::pair<Eigen::Vector3d, float>> &pts,
                    const vector<int> &rgb) {

  pcl::PointCloud<pcl::PointXYZRGB> pc;
  pc.width = pts.size();
  pc.height = 1;
  pc.is_dense = false;
  pc.points.resize(pts.size());

  for (int i = 0; i < pts.size(); i++) {
    pc.points[i].x = pts[i].first[0];
    pc.points[i].y = pts[i].first[2];
    pc.points[i].z = -pts[i].first[1];
    pc.points[i].r = rgb[0];
    pc.points[i].g = rgb[1];
    pc.points[i].b = rgb[2];
  }

  auto pc_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(pc);

  return pc_ptr;
}

inline void merge_point_clouds(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_ptr,
    const std::vector<std::pair<Eigen::Vector3d, float>> &pts_source,
    const Eigen::Matrix<double, 4, 4> &T_target_source,
    const vector<int> &rgb) {

  int sz = pc_ptr->points.size();
  pc_ptr->width = sz + pts_source.size();
  pc_ptr->points.resize(pc_ptr->width * pc_ptr->height);

  Eigen::Matrix<double, 4, 1> pt_source, pt_target;
  for (int i = 0; i < pts_source.size(); i++) {
    pt_source.head(3) = pts_source[i].first;
    pt_source(3) = 1.0;
    pt_target = T_target_source * pt_source;
    pc_ptr->points[sz + i].x = pt_target[0];
    pc_ptr->points[sz + i].y = pt_target[2];
    pc_ptr->points[sz + i].z = -pt_target[1];
    pc_ptr->points[sz + i].r = rgb[0];
    pc_ptr->points[sz + i].g = rgb[1];
    pc_ptr->points[sz + i].b = rgb[2];
  }
}