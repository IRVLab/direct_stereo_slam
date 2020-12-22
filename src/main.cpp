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

#include <chrono>

#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>

#include "FrontEnd.h"
#include "loop_closure/pangolin_viewer/PangolinLoopViewer.h"

inline void print_average(const std::string &name, const TimeVector &tm_vec) {
  float sum = 0;
  for (size_t i = 0; i < tm_vec.size(); i++) {
    sum += std::chrono::duration_cast<std::chrono::duration<double>>(tm_vec[i])
               .count();
  }

  printf("%s %.2f x %lu\n", name.c_str(), 1000 * sum / tm_vec.size(),
         tm_vec.size());
}

using namespace dso;

class SLAMNode {
private:
  double currentTimeStamp;
  int incomingId;
  FrontEnd *front_end_;
  Undistort *undistorter0_;
  Undistort *undistorter1_;

  // scale optimizer
  std::vector<double> tfm_stereo_;
  float scale_opt_thres_; // set to -1 to disable scale optimization

  // loop closure
  LoopHandler *loop_handler_;

  // statistics: time for each frame
  TimeVector frame_tt_;

  void settingsDefault(int preset, int mode);

public:
  SLAMNode(const std::vector<double> &tfm_stereo, const std::string &calib0,
           const std::string &calib1, const std::string &vignette0,
           const std::string &vignette1, const std::string &gamma0,
           const std::string &gamma1, bool nomt, int preset, int mode,
           float scale_opt_thres, float lidar_range, float scan_context_thres);
  ~SLAMNode();
  void imageMessageCallback(const sensor_msgs::ImageConstPtr &msg0,
                            const sensor_msgs::ImageConstPtr &msg1);
};

void SLAMNode::settingsDefault(int preset, int mode) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 1 || preset == 3) {
    printf("preset=%d is not supported", preset);
    exit(1);
  }
  if (preset == 0) {
    printf("DEFAULT settings:\n"
           "- 2000 active points\n"
           "- 5-7 active frames\n"
           "- 1-6 LM iteration each KF\n"
           "- original image resolution\n");

    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;
  }

  if (preset == 2) {
    printf("FAST settings:\n"
           "- 800 active points\n"
           "- 4-6 active frames\n"
           "- 1-4 LM iteration each KF\n"
           "- 424 x 320 image resolution\n");

    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;
  }

  if (mode == 0) {
    printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
  }
  if (mode == 1) {
    printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
  }
  if (mode == 2) {
    printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_minGradHistAdd = 3;
  }

  printf("==============================================\n");
}

SLAMNode::SLAMNode(const std::vector<double> &tfm_stereo,
                   const std::string &calib0, const std::string &calib1,
                   const std::string &vignette0, const std::string &vignette1,
                   const std::string &gamma0, const std::string &gamma1,
                   bool nomt, int preset, int mode, float scale_opt_thres,
                   float lidar_range, float scan_context_thres)
    : tfm_stereo_(tfm_stereo), scale_opt_thres_(scale_opt_thres) {
  // DSO front end
  settingsDefault(preset, mode);

  multiThreading = !nomt;

  undistorter0_ = Undistort::getUndistorterForFile(calib0, gamma0, vignette0);
  undistorter1_ = Undistort::getUndistorterForFile(calib1, gamma1, vignette1);
  assert((int)undistorter0_->getSize()[0] == (int)undistorter1_->getSize()[0]);
  assert((int)undistorter0_->getSize()[1] == (int)undistorter1_->getSize()[1]);

  setGlobalCalib((int)undistorter0_->getSize()[0],
                 (int)undistorter0_->getSize()[1],
                 undistorter0_->getK().cast<float>());

  front_end_ = new FrontEnd(tfm_stereo_, undistorter1_->getK().cast<float>(),
                            scale_opt_thres_);
  if (undistorter0_->photometricUndist != 0)
    front_end_->setGammaFunction(undistorter0_->photometricUndist->getG());

  // Loop closure
  IOWrap::PangolinLoopViewer *pangolinViewer = 0;
  if (!disableAllDisplay) {
    pangolinViewer = new IOWrap::PangolinLoopViewer(
        (int)undistorter0_->getSize()[0], (int)undistorter0_->getSize()[1]);
    front_end_->output_wrapper_.push_back(pangolinViewer);
  }
  loop_handler_ =
      new LoopHandler(lidar_range, scan_context_thres, pangolinViewer);
  // setLoopHandler is called even if loop closure is disabled
  // because results are recordered by LoopHandler
  // but loop closure is disabled internally if loop closure is disabled
  front_end_->setLoopHandler(loop_handler_);

  incomingId = 0;
  currentTimeStamp = -1.0;
}

SLAMNode::~SLAMNode() {
  loop_handler_->savePose();

  printf("\n\n************** Statistics (ms) ***************\n");
  std::cout << "===========VO Time============ " << std::endl;
  print_average("feature_detect", front_end_->feature_detect_time_);
  print_average("scale_opt", front_end_->scale_opt_time_);
  print_average("dso_opt", front_end_->opt_time_);

  std::cout << "======Loop Closure Time======= " << std::endl;
  print_average("pts_generation", loop_handler_->pts_generation_time_);
  print_average("sc_generation", loop_handler_->sc_generation_time_);
  print_average("search_ringkey", loop_handler_->search_ringkey_time_);
  print_average("search_sc", loop_handler_->search_sc_time_);
  print_average("direct_est", loop_handler_->direct_est_time_);
  print_average("icp", loop_handler_->icp_time_);
  print_average("pose_graph_opt", loop_handler_->opt_time_);
  printf("loop_count: %d (direct) + %d (icp)\n",
         loop_handler_->direct_loop_count_, loop_handler_->icp_loop_count_);

  std::cout << "============================== " << std::endl;
  print_average("per_frame", frame_tt_);

  printf("**********************************************\n\n\n");

  delete undistorter0_;
  delete undistorter1_;
  delete loop_handler_;
  for (auto &ow : front_end_->output_wrapper_) {
    delete ow;
  }
  delete front_end_;
}

void SLAMNode::imageMessageCallback(const sensor_msgs::ImageConstPtr &msg0,
                                    const sensor_msgs::ImageConstPtr &msg1) {
  cv::Mat img0, img1;
  try {
    img0 = cv_bridge::toCvShare(msg0, "mono8")->image;
    img1 = cv_bridge::toCvShare(msg1, "mono8")->image;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  // detect if a new sequence is received, restart if so
  if (currentTimeStamp > 0 &&
      fabs(msg0->header.stamp.toSec() - currentTimeStamp) > 10) {
    front_end_->is_lost_ = true;
  }
  currentTimeStamp = msg0->header.stamp.toSec();

  // reinitialize if necessary
  if (front_end_->init_failed_ || front_end_->is_lost_) {
    auto lastPose = front_end_->cur_pose_;
    int existing_kf_size = front_end_->getTotalKFSize();
    std::vector<IOWrap::Output3DWrapper *> wraps = front_end_->output_wrapper_;
    delete front_end_;

    printf("Reinitializing\n");
    front_end_ = new FrontEnd(tfm_stereo_, undistorter1_->getK().cast<float>(),
                              scale_opt_thres_, existing_kf_size);
    if (undistorter0_->photometricUndist != 0)
      front_end_->setGammaFunction(undistorter0_->photometricUndist->getG());
    front_end_->setLoopHandler(loop_handler_);
    front_end_->output_wrapper_ = wraps;
    front_end_->cur_pose_ = lastPose;
    // setting_fullResetRequested=false;
  }

  MinimalImageB minImg0((int)img0.cols, (int)img0.rows,
                        (unsigned char *)img0.data);
  ImageAndExposure *undistImg0 =
      undistorter0_->undistort<unsigned char>(&minImg0, 1, 0, 1.0f);
  undistImg0->timestamp = msg0->header.stamp.toSec();
  MinimalImageB minImg1((int)img1.cols, (int)img1.rows,
                        (unsigned char *)img1.data);
  ImageAndExposure *undistImg1 =
      undistorter1_->undistort<unsigned char>(&minImg1, 1, 0, 1.0f);

  auto t0 = std::chrono::steady_clock::now();
  front_end_->addActiveStereoFrame(undistImg0, undistImg1, incomingId);
  auto t1 = std::chrono::steady_clock::now();
  frame_tt_.push_back(t1 - t0);

  incomingId++;
  delete undistImg0;
  delete undistImg1;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "direct_stereo_slam");
  ros::NodeHandle nhPriv("~");

  /* *********************** required parameters ************************ */
  // stereo camera parameters
  std::vector<double> tfm_stereo;
  std::string topic0, topic1, calib0, calib1;
  if (!nhPriv.getParam("T_stereo/data", tfm_stereo) ||
      !nhPriv.getParam("topic0", topic0) ||
      !nhPriv.getParam("topic1", topic1) ||
      !nhPriv.getParam("calib0", calib0) ||
      !nhPriv.getParam("calib1", calib1)) {
    ROS_INFO("Fail to get sensor topics/params, exit.");
    return -1;
  }
  std::string vignette0, vignette1, gamma0, gamma1;
  nhPriv.param<std::string>("vignette0", vignette0, "");
  nhPriv.param<std::string>("vignette1", vignette1, "");
  nhPriv.param<std::string>("gamma0", gamma0, "");
  nhPriv.param<std::string>("gamma1", gamma1, "");

  /* *********************** optional parameters ************************ */
  // DSO settings
  bool nomt;
  int preset, mode;
  nhPriv.param("preset", preset, 0);
  nhPriv.param("mode", mode, 1);
  nhPriv.param("quiet", setting_debugout_runquiet, true);
  nhPriv.param("nogui", disableAllDisplay, false);
  nhPriv.param("nomt", nomt, false);

  // scale optimization accept threshold
  // set to -1 to disable scale optimization, i.e., dso mode
  float scale_opt_thres;
  nhPriv.param("scale_opt_thres", scale_opt_thres, 15.0f);

  // loop closure parameters
  float lidar_range; // set to -1 to disable loop closure
  float scan_context_thres;
  nhPriv.param("lidar_range", lidar_range, 40.0f);
  nhPriv.param("scan_context_thres", scan_context_thres, 0.33f);

  // read from a bag file
  std::string bag_path;
  nhPriv.param<std::string>("bag", bag_path, "");

  /* ******************************************************************** */

  SLAMNode slam_node(tfm_stereo, calib0, calib1, vignette0, vignette1, gamma0,
                     gamma1, nomt, preset, mode, scale_opt_thres, lidar_range,
                     scan_context_thres);

  if (!bag_path.empty()) {

    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {topic0, topic1};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    sensor_msgs::ImageConstPtr img0, img1;
    bool img0_updated(false), img1_updated(false);
    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      if (m.getTopic() == topic0) {
        img0 = m.instantiate<sensor_msgs::Image>();
        img0_updated = true;
      }
      if (m.getTopic() == topic1) {
        img1 = m.instantiate<sensor_msgs::Image>();
        img1_updated = true;
      }
      if (img0_updated && img1_updated) {
        assert(fabs(img0->header.stamp.toSec() - img1->header.stamp.toSec()) <
               0.1);
        slam_node.imageMessageCallback(img0, img1);
        img0_updated = img1_updated = false;
      }
    }
    bag.close();
  } else {

    // ROS subscribe to stereo images
    ros::NodeHandle nh;
    auto *cam0_sub =
        new message_filters::Subscriber<sensor_msgs::Image>(nh, topic0, 10000);
    auto *cam1_sub =
        new message_filters::Subscriber<sensor_msgs::Image>(nh, topic1, 10000);
    auto *sync = new message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image>>(
        message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image>(10),
        *cam0_sub, *cam1_sub);
    sync->registerCallback(
        boost::bind(&SLAMNode::imageMessageCallback, &slam_node, _1, _2));
    ros::spin();
  }

  return 0;
}
