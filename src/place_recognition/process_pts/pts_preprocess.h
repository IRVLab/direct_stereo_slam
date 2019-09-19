#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include "place_recognition/process_pts/pts_filter.h"
#include "place_recognition/utils/PosesPts.h"

#define PTS_HIST 6
#define INIT_FRAME 30

inline void
get_pts_spherical(const IDPose *cur_pose,
                  std::vector<IDPtIntensity *> &pts_nearby,
                  std::vector<std::pair<Eigen::Vector3d, float>> &pts_spherical,
                  double lidarRange, double voxelAngle) {
  std::vector<IDPtIntensity *> new_pts_nearby, pts_spherical_raw;
  for (auto &p : pts_nearby) {
    Eigen::Vector4d p_g(p->pt(0), p->pt(1), p->pt(2), 1.0);
    Eigen::Vector3d p_l = cur_pose->w2c * p_g;

    if (p_l.norm() < lidarRange) {
      pts_spherical_raw.push_back(
          new IDPtIntensity(p->incoming_id, p_l, p->intensity));
    } else if (p_l(2) < 0 ||
               (cur_pose->incoming_id - p->incoming_id) > PTS_HIST)
      continue;

    new_pts_nearby.push_back(p);
  }
  pts_nearby = new_pts_nearby; // update nearby pts

  // filter points
  filterPointsPolar(pts_spherical_raw, lidarRange, {voxelAngle, voxelAngle, 1},
                    pts_spherical, false);

  printf("\rFrame count: %d, Pts (Total: %lu, Sphere: %lu, Filtered: %lu)",
         cur_pose->incoming_id, pts_nearby.size(), pts_spherical_raw.size(),
         pts_spherical.size());
  fflush(stdout);
}

inline bool
pts_preprocess(const IDPose *cur_pose, std::vector<IDPtIntensity *> &pts_nearby,
               const std::vector<IDPtIntensity *> pts_history, int &pts_idx,
               std::vector<std::pair<Eigen::Vector3d, float>> &pts_spherical,
               double lidarRange, double voxelAngle) {
  static int frame_from_reset = 0;

  // reset
  if (cur_pose->w2c.col(3).norm() < 1.0) {
    printf("\nReset at id: %d\n", cur_pose->incoming_id);
    frame_from_reset = 0;
    pts_nearby.clear();
  }

  // retrive new pts
  while (pts_idx < pts_history.size() &&
         std::abs(cur_pose->incoming_id - pts_history[pts_idx]->incoming_id) <
             PTS_HIST) {
    pts_nearby.push_back(pts_history[pts_idx]);
    pts_idx++;
  }

  // accumulate enough pts
  if (frame_from_reset < INIT_FRAME) {
    frame_from_reset++;
    return false;
  }

  // get spherical pts
  pts_spherical.clear();
  get_pts_spherical(cur_pose, pts_nearby, pts_spherical, lidarRange,
                    voxelAngle);

  return true;
}
