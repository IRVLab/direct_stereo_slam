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

// ROS and OpenCV headers
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/Image.h>

#include "FrontEnd.h"
#include "loop_closure/pangolin_viewer/PangolinLoopViewer.h"
#include "timing.h"

using namespace dso;

class SLAMNode {
private:
  double currentTimeStamp;
  int incomingId;
  FrontEnd *frontEnd;
  Undistort *undistorter0;

  // scale optimizer
  ScaleOptimizer *scale_optimizer_;

  // loop closure
  LoopHandler *loop_handler_;

  // statistics: time for each frame
  std::vector<double> frameTime;

  void settingsDefault(int preset, int mode);

public:
  SLAMNode();
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

SLAMNode::SLAMNode() {
  ros::NodeHandle nhPriv("~");

  /* *********************** required parameters ************************ */
  // stereo camera parameters
  std::vector<double> tfm_stereo;
  std::string cam0_topic;
  std::string cam1_topic;
  std::string calib0;
  std::string calib1;
  if (!nhPriv.getParam("T_stereo/data", tfm_stereo) ||
      !nhPriv.getParam("cam0_topic", cam0_topic) ||
      !nhPriv.getParam("cam1_topic", cam1_topic) ||
      !nhPriv.getParam("calib0", calib0) ||
      !nhPriv.getParam("calib1", calib1)) {
    ROS_INFO("Fail to get sensor topics/params, exit.");
    return;
  }
  std::string vignette0 = "";
  std::string vignette1 = "";
  std::string gamma0 = "";
  std::string gamma1 = "";
  nhPriv.param<std::string>("vignette0", vignette0, "");
  nhPriv.param<std::string>("vignette1", vignette1, "");
  nhPriv.param<std::string>("gamma0", gamma0, "");
  nhPriv.param<std::string>("gamma1", gamma1, "");

  /* *********************** optional parameters ************************ */
  // DSO settings
  bool nomt;
  int preset, mode;
  setting_onlyLogKFPoses = false;
  nhPriv.param("preset", preset, 0);
  nhPriv.param("mode", mode, 1);
  nhPriv.param("quiet", setting_debugout_runquiet, true);
  nhPriv.param("nogui", disableAllDisplay, false);
  nhPriv.param("nomt", nomt, false);

  // scale optimization parameters
  float scale_opt_thres; // set to -1 to disable scale optimization
  nhPriv.param("scale_accept_thres", scale_opt_thres, 15.0f);

  // loop closure parameters
  float lidar_range; // set to -1 to disable loop closure
  float scan_context_thres;
  nhPriv.param("lidar_range", lidar_range, 40.0f);
  nhPriv.param("scan_context_thres", scan_context_thres, 0.4f);

  /* ******************************************************************** */

  // DSO front end
  settingsDefault(preset, mode);

  multiThreading = !nomt;

  undistorter0 = Undistort::getUndistorterForFile(calib0, gamma0, vignette0);

  setGlobalCalib((int)undistorter0->getSize()[0],
                 (int)undistorter0->getSize()[1],
                 undistorter0->getK().cast<float>());

  frontEnd = new FrontEnd();

  // Scale optimization
  if (scale_opt_thres > 0) {
    Undistort *undistorter1 = Undistort::getUndistorterForFile(
        calib1, gamma1, vignette1); // will be deleted in ~ScaleOptimizer()
    assert((int)undistorter0->getSize()[0] == (int)undistorter1->getSize()[0]);
    assert((int)undistorter0->getSize()[1] == (int)undistorter1->getSize()[1]);

    scale_optimizer_ =
        new ScaleOptimizer(undistorter1, tfm_stereo, scale_opt_thres);
    frontEnd->setScaleOptimizer(scale_optimizer_);
  } else {
    scale_optimizer_ = 0;
  }

  // Loop closure
  IOWrap::PangolinLoopViewer *pangolinViewer = 0;
  if (!disableAllDisplay) {
    pangolinViewer = new IOWrap::PangolinLoopViewer(
        (int)undistorter0->getSize()[0], (int)undistorter0->getSize()[1]);
    frontEnd->output_wrapper_.push_back(pangolinViewer);
  }
  loop_handler_ =
      new LoopHandler(lidar_range, scan_context_thres, pangolinViewer);
  // setLoopHandler is called even loop closure is disabled
  // because results are recordered by LoopHandler
  // but loop closure is disabled inside LoopHandler
  frontEnd->setLoopHandler(loop_handler_);

  // ROS subscribe to stereo images
  ros::NodeHandle nh;
  auto *cam0_sub = new message_filters::Subscriber<sensor_msgs::Image>(
      nh, cam0_topic, 10000);
  auto *cam1_sub = new message_filters::Subscriber<sensor_msgs::Image>(
      nh, cam1_topic, 10000);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      StereoSyncPolicy;
  auto *sync = new message_filters::Synchronizer<StereoSyncPolicy>(
      StereoSyncPolicy(10), *cam0_sub, *cam1_sub);
  sync->registerCallback(
      boost::bind(&SLAMNode::imageMessageCallback, this, _1, _2));

  incomingId = 0;
  currentTimeStamp = -1.0;
}

SLAMNode::~SLAMNode() {
  printf("\n\n************** Statistics (ms) ***************\n");
  frontEnd->printTimeStat();
  loop_handler_->printTimeStatAndSavePose();
  std::cout << "per_frame (ms) " << 1000 * average(frameTime) << " * "
            << frameTime.size() << std::endl;
  printf("**************************************************\n\n\n");

  delete undistorter0;
  delete scale_optimizer_;
  delete loop_handler_;
  for (auto &ow : frontEnd->output_wrapper_) {
    delete ow;
  }
  delete frontEnd;
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
    frontEnd->is_lost_ = true;
  }
  currentTimeStamp = msg0->header.stamp.toSec();

  // reinitialize if necessary
  if (frontEnd->init_failed_ || frontEnd->is_lost_) {
    auto lastPose = frontEnd->cur_pose_;
    int existing_kf_size = frontEnd->getTotalKFSize();
    std::vector<IOWrap::Output3DWrapper *> wraps = frontEnd->output_wrapper_;
    delete frontEnd;

    printf("Reinitializing\n");
    frontEnd = new FrontEnd(existing_kf_size);
    if (scale_optimizer_) {
      frontEnd->setScaleOptimizer(scale_optimizer_);
    }
    frontEnd->setLoopHandler(loop_handler_);
    frontEnd->output_wrapper_ = wraps;
    frontEnd->cur_pose_ = lastPose;
    // setting_fullResetRequested=false;
  }

  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  if (scale_optimizer_) {
    frontEnd->addStereoImg(img1, incomingId);
  }

  if (undistorter0->photometricUndist != 0)
    frontEnd->setGammaFunction(undistorter0->photometricUndist->getG());

  MinimalImageB minImg((int)img0.cols, (int)img0.rows,
                       (unsigned char *)img0.data);
  ImageAndExposure *undistImg =
      undistorter0->undistort<unsigned char>(&minImg, 1, 0, 1.0f);
  undistImg->timestamp = msg0->header.stamp.toSec();
  frontEnd->addActiveFrame(undistImg, incomingId);
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  frameTime.push_back(duration(t0, t1));
  incomingId++;
  delete undistImg;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "direct_stereo_slam");
  SLAMNode vo_node;
  ros::spin();
  return 0;
}
