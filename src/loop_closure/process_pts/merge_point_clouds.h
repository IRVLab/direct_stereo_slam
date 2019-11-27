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

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr
create_point_clouds(const std::vector<Eigen::Vector3d> &pts,
                    const std::vector<int> &rgb) {

  pcl::PointCloud<pcl::PointXYZRGB> pc;
  pc.width = pts.size();
  pc.height = 1;
  pc.is_dense = false;
  pc.points.resize(pts.size());

  for (size_t i = 0; i < pts.size(); i++) {
    pc.points[i].x = pts[i](0);
    pc.points[i].y = pts[i](2);
    pc.points[i].z = -pts[i](1);
    pc.points[i].r = rgb[0];
    pc.points[i].g = rgb[1];
    pc.points[i].b = rgb[2];
  }

  auto pc_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(pc);

  return pc_ptr;
}

inline void merge_point_clouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_ptr,
                               const std::vector<Eigen::Vector3d> &pts_source,
                               const Eigen::Matrix4d &tfm_target_source,
                               const std::vector<int> &rgb) {

  int sz = pc_ptr->points.size();
  pc_ptr->width = sz + pts_source.size();
  pc_ptr->points.resize(pc_ptr->width * pc_ptr->height);

  Eigen::Matrix<double, 4, 1> pt_source, pt_target;
  for (size_t i = 0; i < pts_source.size(); i++) {
    pt_source.head(3) = pts_source[i];
    pt_source(3) = 1.0;
    pt_target = tfm_target_source * pt_source;
    pc_ptr->points[sz + i].x = pt_target[0];
    pc_ptr->points[sz + i].y = pt_target[2];
    pc_ptr->points[sz + i].z = -pt_target[1];
    pc_ptr->points[sz + i].r = rgb[0];
    pc_ptr->points[sz + i].g = rgb[1];
    pc_ptr->points[sz + i].b = rgb[2];
  }
}