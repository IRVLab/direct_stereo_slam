#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"

#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"
#include <boost/thread.hpp>

#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"

#include "SODSOSystem.h"

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include <Eigen/Core>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/String.h>

using namespace dso;

sig_atomic_t stopFlag = 0; // sigint flag

class SODSONode {
private:
  std::string cam0_topic;
  std::string cam1_topic;
  std::string calib0;
  std::string calib1;
  std::string vignette0 = "";
  std::string vignette1 = "";
  std::string gamma0 = "";
  std::string gamma1 = "";

  float init_scale;
  float scale_accept_th;
  SE3 T_stereo_;

  SODSOSystem *so_dso_System;
  Undistort *undistorter0;
  Undistort *undistorter1;
  int frameID;

  float playbackSpeed =
      0; // 0 for linearize (play as fast as possible, while sequentializing
         // tracking & mapping). otherwise, factor on timestamps.
  void settingsDefault(int preset);

public:
  SODSONode();
  ~SODSONode();
  void imageMessageCallback(const sensor_msgs::ImageConstPtr &msg0,
                            const sensor_msgs::ImageConstPtr &msg1);
  void finish();
};

void SODSONode::settingsDefault(int preset) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 0 || preset == 1) {
    printf("DEFAULT settings:\n"
           "- %s real-time enforcing\n"
           "- 2000 active points\n"
           "- 5-7 active frames\n"
           "- 1-6 LM iteration each KF\n"
           "- original image resolution\n",
           preset == 0 ? "no " : "1x");

    playbackSpeed = (preset == 0 ? 0 : 1);
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    setting_logStuff = false;
  }

  if (preset == 2 || preset == 3) {
    printf("FAST settings:\n"
           "- %s real-time enforcing\n"
           "- 800 active points\n"
           "- 4-6 active frames\n"
           "- 1-4 LM iteration each KF\n"
           "- 424 x 320 image resolution\n",
           preset == 2 ? "no " : "5x");

    playbackSpeed = (preset == 2 ? 0 : 1);
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}

SODSONode::SODSONode() {
  ros::NodeHandle nhPriv("~");
  // stereo camera model
  std::vector<double> T_stereo_vec;
  if (!nhPriv.getParam("cam0_topic", cam0_topic) ||
      !nhPriv.getParam("cam1_topic", cam1_topic) ||
      !nhPriv.getParam("calib0", calib0) ||
      !nhPriv.getParam("calib1", calib1) ||
      !nhPriv.getParam("T_stereo/data", T_stereo_vec)) {
    ROS_INFO("Fail to get sensor topics/params, exit.");
    return;
  }

  // stereo pose
  cv::Mat T_stereo_cv = cv::Mat(T_stereo_vec);
  T_stereo_cv = T_stereo_cv.reshape(0, 4);
  Eigen::Matrix<double, 4, 4> T_stereo_eigen;
  cv::cv2eigen(T_stereo_cv, T_stereo_eigen);
  T_stereo_ = SE3(T_stereo_eigen);

  nhPriv.param("init_scale", init_scale, 1.0f);
  nhPriv.param("scale_accept_th", scale_accept_th, 15.0f);

  bool nomt;
  int preset, mode;
  setting_onlyLogKFPoses = false;
  nhPriv.param("preset", preset, 0);
  nhPriv.param("mode", mode, 0);
  nhPriv.param<std::string>("vignette0", vignette0, "");
  nhPriv.param<std::string>("vignette1", vignette1, "");
  nhPriv.param<std::string>("gamma0", gamma0, "");
  nhPriv.param<std::string>("gamma1", gamma1, "");
  nhPriv.param("quiet", setting_debugout_runquiet, true);
  nhPriv.param("nolog", setting_logStuff, true);
  nhPriv.param("nogui", disableAllDisplay, false);
  nhPriv.param("nomt", nomt, false);
  multiThreading = !nomt;
  settingsDefault(preset);
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

  undistorter0 = Undistort::getUndistorterForFile(calib0, gamma0, vignette0);
  undistorter1 = Undistort::getUndistorterForFile(calib1, gamma1, vignette1);

  setGlobalCalib((int)undistorter0->getSize()[0],
                 (int)undistorter0->getSize()[1],
                 undistorter0->getK().cast<float>());

  so_dso_System = new SODSOSystem((int)undistorter1->getSize()[0],
                                  (int)undistorter1->getSize()[1],
                                  undistorter1->getK().cast<float>(), T_stereo_,
                                  undistorter1, init_scale, scale_accept_th);
  so_dso_System->linearizeOperation = (playbackSpeed == 0);

  if (!disableAllDisplay)
    so_dso_System->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
        (int)undistorter0->getSize()[0], (int)undistorter0->getSize()[1]));

  ros::NodeHandle nh;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      StereoSyncPolicy;
  message_filters::Subscriber<sensor_msgs::Image> *cam0_sub;
  message_filters::Subscriber<sensor_msgs::Image> *cam1_sub;
  message_filters::Synchronizer<StereoSyncPolicy> *sync;
  cam0_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, cam0_topic,
                                                                 10000);
  cam1_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, cam1_topic,
                                                                 10000);
  sync = new message_filters::Synchronizer<StereoSyncPolicy>(
      StereoSyncPolicy(10), *cam0_sub, *cam1_sub);
  sync->registerCallback(
      boost::bind(&SODSONode::imageMessageCallback, this, _1, _2));

  frameID = 0;
}

SODSONode::~SODSONode() {
  delete so_dso_System;
  delete undistorter0;
  delete undistorter1;
}

void SODSONode::finish() {
  so_dso_System->blockUntilMappingIsFinished();
  clock_t ended = clock();
  struct timeval tv_end;
  gettimeofday(&tv_end, NULL);

  so_dso_System->printResult("poses.txt", "ba_time.txt", "scale_time.txt",
                             "fps_time.txt");

  // int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
  // double numSecondsProcessed =
  // fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
  // double MilliSecondsTakenSingle =
  // 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC); double MilliSecondsTakenMT
  // = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f +
  // (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
  // printf("\n======================"
  //         "\n%d Frames (%.1f fps)"
  //         "\n%.2fms per frame (single core); "
  //         "\n%.2fms per frame (multi core); "
  //         "\n%.3fx (single core); "
  //         "\n%.3fx (multi core); "
  //         "\n======================\n\n",
  //         numFramesProcessed, numFramesProcessed/numSecondsProcessed,
  //         MilliSecondsTakenSingle/numFramesProcessed,
  //         MilliSecondsTakenMT / (float)numFramesProcessed,
  //         1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
  //         1000 / (MilliSecondsTakenMT / numSecondsProcessed));
  // //so_dso_System->printFrameLifetimes();
  // if(setting_logStuff)
  // {
  //     std::ofstream tmlog;
  //     tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
  //     tmlog <<
  //     1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages())
  //     << " "
  //           << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f +
  //           (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) /
  //           (float)reader->getNumImages() << "\n";
  //     tmlog.flush();
  //     tmlog.close();
  // }
}

void SODSONode::imageMessageCallback(const sensor_msgs::ImageConstPtr &msg0,
                                     const sensor_msgs::ImageConstPtr &msg1) {
  cv::Mat img, stereo_img, conc;
  try {
    img = cv_bridge::toCvShare(msg0, "mono8")->image;
    stereo_img = cv_bridge::toCvShare(msg1, "mono8")->image;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  if (so_dso_System->isLost) {
    printf("Reinitializing\n");
    std::vector<IOWrap::Output3DWrapper *> wraps = so_dso_System->outputWrapper;
    delete so_dso_System;
    for (IOWrap::Output3DWrapper *ow : wraps)
      ow->reset();
    so_dso_System = new SODSOSystem(
        (int)undistorter1->getSize()[0], (int)undistorter1->getSize()[1],
        undistorter1->getK().cast<float>(), T_stereo_, undistorter1, init_scale,
        scale_accept_th);
    so_dso_System->linearizeOperation = (playbackSpeed == 0);
    so_dso_System->outputWrapper = wraps;
    if (undistorter0->photometricUndist != 0)
      so_dso_System->setGammaFunction(undistorter0->photometricUndist->getG());
    // setting_fullResetRequested=false;
  }

  if (undistorter0->photometricUndist != 0)
    so_dso_System->setGammaFunction(undistorter0->photometricUndist->getG());

  MinimalImageB minImg((int)img.cols, (int)img.rows, (unsigned char *)img.data);
  ImageAndExposure *undistImg =
      undistorter0->undistort<unsigned char>(&minImg, 1, 0, 1.0f);
  undistImg->timestamp = msg0->header.stamp.toSec();
  so_dso_System->addStereoImg(stereo_img, frameID);
  so_dso_System->addActiveFrame(undistImg, frameID);
  frameID++;
  delete undistImg;
}

void finishHandler(int sig) { stopFlag = 1; }

int main(int argc, char **argv) {
  ros::init(argc, argv, "so_dso_");
  SODSONode vo_node;

  signal(SIGINT, finishHandler);

  ros::Rate r(10); // 10 hz
  while (stopFlag == 0) {
    ros::spinOnce();
    r.sleep();
  }

  printf("Finishing up\n");
  vo_node.finish();

  return 0;
}
