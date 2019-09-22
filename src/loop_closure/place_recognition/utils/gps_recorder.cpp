#include <fstream>
#include <geometry_msgs/TransformStamped.h>
#include <iomanip>
#include <ros/ros.h>
#include <signal.h>
#include <vector>

sig_atomic_t stopFlag = 0; // sigint flag
std::vector<std::vector<double>> gps_data;

void chatterCallback(const geometry_msgs::TransformStamped::ConstPtr &msg) {
  gps_data.push_back({msg->transform.translation.x,
                      msg->transform.translation.y,
                      msg->transform.translation.z});
}

void finishHandler(int sig) { stopFlag = 1; }

int main(int argc, char **argv) {
  ros::init(argc, argv, "gps_recorder");
  ros::NodeHandle n;

  gps_data.clear();
  ros::Subscriber sub = n.subscribe("gps", 1000, chatterCallback);

  signal(SIGINT, finishHandler);
  ros::Rate r(10); // 10 hz
  while (stopFlag == 0) {
    ros::spinOnce();
    r.sleep();
  }

  std::ofstream gps_file;
  gps_file.open("gps.txt");
  gps_file << std::fixed;
  gps_file << std::setprecision(6);
  printf("Recording GPS data\n");
  for (auto &d : gps_data) {
    gps_file << d[0] << " " << d[1] << " " << d[2] << "\n";
  }

  return 0;
}
