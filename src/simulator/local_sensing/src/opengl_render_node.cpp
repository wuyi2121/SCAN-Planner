#include "opengl_sim.hpp"
// cv_bridge headers became .hpp in Iron; the .h form is gone in Jazzy.
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <deque>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <numeric>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <string>
#include <tr1/unordered_map>
#include <tf2_ros/transform_broadcaster.h>

using namespace Eigen;
using namespace std;

#define MAX_INTENSITY 1
#define MIN_INTENSITY 0.1

string file_name;
deque<double> comp_time_vec;
rclcpp::Node::SharedPtr ros_node;

std::string quad_name;
std::string sensor_type_ = "depth";
int drone_num = 0;
int drone_id = 0;
double lidar_pitch;
opengl_pointcloud_render render;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud, pub_intercloud,
    pub_dyncloud, pub_uavcloud, pub_collisioncloud;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pinhole_depth_pub_;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr camera_pose_pub_, lidar_pose_pub_;
rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr comp_time_pub;
sensor_msgs::msg::PointCloud2 local_map_pcl;
sensor_msgs::msg::PointCloud2 local_depth_pcl;
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_sub;
std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> other_odom_subs;
rclcpp::TimerBase::SharedPtr local_sensing_timer, dynobj_timer;
std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

rclcpp::Time t_init(0, 0, RCL_ROS_TIME);

bool has_odom(false);
bool has_global_map(false);
bool use_global_map_topic(false);

nav_msgs::msg::Odometry odom_;
Eigen::Matrix4f sensor2body, sensor2world;

std::string quadTopic(const std::string &name) {
  std::string prefix = quad_name.empty() ? std::string("quad_0") : quad_name;
  if (prefix.front() != '/') {
    prefix = "/" + prefix;
  }
  return prefix + "/" + name;
}

int output_pcd;
int collisioncheck_enable;
int is_360lidar;
int use_avia_pattern, use_vlp32_pattern, use_minicf_pattern, use_os128_pattern, use_os64_pattern,
use_gaussian_filter;
double livox_linestep;
double sensing_horizon, sensing_rate, estimation_rate, polar_resolution, yaw_fov, vertical_fov,
min_raylength, downsample_res, curvature_limit, hash_cubesize, collision_range;
double x_size, y_size, z_size;
double gl_xl, gl_yl, gl_zl;
double resolution, inv_resolution;
int GLX_SIZE, GLY_SIZE, GLZ_SIZE;

int cam_width = 640;
int cam_height = 480;
double cam_fx = 387.229248046875;
double cam_fy = 387.229248046875;
double cam_cx = 321.04638671875;
double cam_cy = 243.44969177246094;

int plane_interline = 1;

// multi uav variables
// int drone_id = 0;
vector<Eigen::Vector3d> other_uav_pos;
vector<double> other_uav_rcv_time;
vector<vector<PointType>> otheruav_points, otheruav_points_inrender;
vector<vector<int>> otheruav_pointsindex, otheruav_pointsindex_inrender;
pcl::PointCloud<PointType> otheruav_points_vis, dyn_points_vis;
double uav_size[3];
int uav_points_num;
// int drone_num = 0;
int drone_drawpoints_num[3];

int use_uav_extra_model = 1;
pcl::PointCloud<PointType> uav_extra_model;

// dynamic objects variables
int dynobj_enable;
double dynobject_size;
vector<Eigen::Vector3d> dynobj_poss;
vector<int> dynobj_move_modes;
vector<int> dynobj_types;
vector<Eigen::Vector3d> dynobj_dir;
vector<PointType> dynobj_points;
pcl::PointCloud<PointType> dynobj_points_vis;
vector<int> dynobj_pointsindex;
double dyn_velocity;
int dynobject_num;
int dyn_mode;
Eigen::Vector3f map_min, map_max;
int dyn_obs_diff_size_on = 1;
vector<double> dyn_obs_size_vec;
vector<rclcpp::Time> dyn_start_time_vec;
PointType global_mapmin;
PointType global_mapmax;

pcl::PointCloud<PointType> dynamic_input_points;

pcl::PointCloud<PointType> cloud_all_map, local_map_filled, point_in_sensor;
pcl::PointCloud<PointType>::Ptr local_map(new pcl::PointCloud<PointType>);
// pcl::VoxelGrid<PointType> _voxel_sampler;
sensor_msgs::msg::PointCloud2 local_map_pcd, sensor_map_pcd;

// variables for collision checks
pcl::search::KdTree<PointType> _kdtreeLocalMap, kdtree_dyn;
vector<int> pointIdxRadiusSearch;
vector<float> pointRadiusSquaredDistance;
double collision_check_time_sum = 0;
int collision_check_time_count = 0;
// hash map of uav models for collision checks
std::tr1::unordered_map<int, bool> uavpoint_hashmap;
PointType uav_modelmin;
PointType uav_modelmax;
int uavhash_xsize, uavhash_ysize, uavhash_zsize;
pcl::PointCloud<PointType> collision_points;

bool useDepthSensor() {
  return sensor_type_ == "depth";
}

bool useLidarSensor() {
  return sensor_type_ == "lidar";
}

nav_msgs::msg::Odometry makeSensorPoseMsg(const nav_msgs::msg::Odometry &body_odom,
                                     const Eigen::Matrix4f &sensor_pose,
                                     const std::string &child_frame_id) {
  Eigen::Quaternionf q(sensor_pose.block<3, 3>(0, 0));
  q.normalize();

  nav_msgs::msg::Odometry sensor_odom;
  sensor_odom.header = body_odom.header;
  sensor_odom.header.frame_id = "world";
  sensor_odom.child_frame_id = child_frame_id;
  sensor_odom.pose.pose.position.x = sensor_pose(0, 3);
  sensor_odom.pose.pose.position.y = sensor_pose(1, 3);
  sensor_odom.pose.pose.position.z = sensor_pose(2, 3);
  sensor_odom.pose.pose.orientation.w = q.w();
  sensor_odom.pose.pose.orientation.x = q.x();
  sensor_odom.pose.pose.orientation.y = q.y();
  sensor_odom.pose.pose.orientation.z = q.z();
  return sensor_odom;
}

bool preparePinholeMap(const pcl::PointCloud<PointType> &raw_cloud) {
  pcl::PointCloud<PointType> finite_cloud;
  finite_cloud.points.reserve(raw_cloud.points.size());
  for (const auto &point : raw_cloud.points) {
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
      finite_cloud.points.push_back(point);
    }
  }

  if (finite_cloud.empty()) {
    RCLCPP_ERROR(ros_node->get_logger(), "Empty map cloud; cannot publish pinhole depth.");
    return false;
  }

  finite_cloud.width = finite_cloud.points.size();
  finite_cloud.height = 1;
  finite_cloud.is_dense = true;

  pcl::VoxelGrid<PointType> voxel_sampler;
  voxel_sampler.setInputCloud(finite_cloud.makeShared());
  voxel_sampler.setLeafSize(downsample_res, downsample_res, downsample_res);
  voxel_sampler.filter(cloud_all_map);

  if (cloud_all_map.empty()) {
    RCLCPP_ERROR(ros_node->get_logger(), "Downsampled map cloud is empty; cannot publish pinhole depth.");
    return false;
  }

  cloud_all_map.width = cloud_all_map.points.size();
  cloud_all_map.height = 1;
  cloud_all_map.is_dense = true;

  pcl::getMinMax3D(cloud_all_map, global_mapmin, global_mapmax);
  _kdtreeLocalMap.setInputCloud(cloud_all_map.makeShared());

  RCLCPP_INFO(ros_node->get_logger(), "Pinhole depth map ready, points: %zu", cloud_all_map.points.size());
  return true;
}

bool loadPinholeMapFromFile(const std::string &map_file) {
  pcl::PointCloud<PointType> raw_cloud;
  if (pcl::io::loadPCDFile<PointType>(map_file, raw_cloud) == -1) {
    RCLCPP_ERROR(ros_node->get_logger(), "Failed to read PCD file for pinhole depth: %s", map_file.c_str());
    return false;
  }

  return preparePinholeMap(raw_cloud);
}

void publishCameraPose(const nav_msgs::msg::Odometry &body_odom) {
  camera_pose_pub_->publish(makeSensorPoseMsg(body_odom, sensor2world, "camera"));
}

void publishLidarPose(const nav_msgs::msg::Odometry &body_odom, const rclcpp::Time &stamp,
                      const Eigen::Vector3f &pos, const Eigen::Quaternionf &quat) {
  nav_msgs::msg::Odometry lidar_pose;
  lidar_pose.header = body_odom.header;
  lidar_pose.header.stamp = stamp;
  lidar_pose.header.frame_id = "world";
  lidar_pose.child_frame_id = "lidar";
  lidar_pose.pose.pose.position.x = pos.x();
  lidar_pose.pose.pose.position.y = pos.y();
  lidar_pose.pose.pose.position.z = pos.z();
  lidar_pose.pose.pose.orientation.x = quat.x();
  lidar_pose.pose.pose.orientation.y = quat.y();
  lidar_pose.pose.pose.orientation.z = quat.z();
  lidar_pose.pose.pose.orientation.w = quat.w();
  lidar_pose_pub_->publish(lidar_pose);
}

void publishPinholeDepth(const rclcpp::Time &stamp) {
  if (cloud_all_map.empty()) {
    RCLCPP_WARN_THROTTLE(ros_node->get_logger(), *ros_node->get_clock(), 1000,
                         "No pinhole map cloud yet; skip depth publish.");
    return;
  }

  if (cam_width <= 0 || cam_height <= 0 || cam_fx <= 0.0 || cam_fy <= 0.0) {
    RCLCPP_WARN_THROTTLE(ros_node->get_logger(), *ros_node->get_clock(), 1000,
                         "Invalid pinhole camera intrinsics.");
    return;
  }

  cv::Mat depth_image(cam_height, cam_width, CV_32FC1, cv::Scalar(0.0f));

  const Eigen::Matrix4f world2camera = sensor2world.inverse();
  const Eigen::Vector3f camera_pos(sensor2world(0, 3), sensor2world(1, 3), sensor2world(2, 3));
  const double max_depth = std::max(0.1, sensing_horizon);

  PointType search_point;
  search_point.x = camera_pos.x();
  search_point.y = camera_pos.y();
  search_point.z = camera_pos.z();

  std::vector<int> indices;
  std::vector<float> distances;
  _kdtreeLocalMap.radiusSearch(search_point, max_depth, indices, distances);

  for (const int index : indices) {
    const PointType &pt = cloud_all_map.points[index];
    const Eigen::Vector4f pt_world(pt.x, pt.y, pt.z, 1.0f);
    const Eigen::Vector4f pt_camera = world2camera * pt_world;

    const double z = pt_camera.z();
    if (z <= 1e-3 || z > max_depth) {
      continue;
    }

    const int u_center = static_cast<int>(std::round(cam_fx * pt_camera.x() / z + cam_cx));
    const int v_center = static_cast<int>(std::round(cam_fy * pt_camera.y() / z + cam_cy));
    if (u_center < 0 || u_center >= cam_width || v_center < 0 || v_center >= cam_height) {
      continue;
    }

    const int splat_radius =
        std::min(4, std::max(0, static_cast<int>(std::ceil(0.5 * cam_fx * downsample_res / z))));
    const int u_min = std::max(0, u_center - splat_radius);
    const int u_max = std::min(cam_width - 1, u_center + splat_radius);
    const int v_min = std::max(0, v_center - splat_radius);
    const int v_max = std::min(cam_height - 1, v_center + splat_radius);

    for (int v = v_min; v <= v_max; ++v) {
      for (int u = u_min; u <= u_max; ++u) {
        float &depth = depth_image.at<float>(v, u);
        if (depth <= 0.0f || z < depth) {
          depth = static_cast<float>(z);
        }
      }
    }
  }

  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = stamp;
  out_msg.header.frame_id = "camera";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  out_msg.image = depth_image;
  pinhole_depth_pub_->publish(*out_msg.toImageMsg());
}

pcl::PointCloud<PointType> generate_sphere_cloud(double size) {
  pcl::PointCloud<PointType> sphere_cloud;
  PointType temp_point;
  double radius = size / 2;
  double x, y, z;

  // use downsample resolution to compute how many points in a sphere
  int theta_num = 2 * M_PI * radius / downsample_res;
  int fi_num = M_PI * radius / downsample_res;

  for (int i = 0; i < theta_num; i++) {
    for (int j = 0; j < fi_num; j++) {
      x = radius * sin(i * 2 * M_PI / theta_num) * cos(j * M_PI / fi_num);
      y = radius * sin(i * 2 * M_PI / theta_num) * sin(j * M_PI / fi_num);
      z = radius * cos(i * 2 * M_PI / theta_num);
      temp_point.x = x;
      temp_point.y = y;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 1.0 / 2.0;
      sphere_cloud.push_back(temp_point);
    }
  }

  return sphere_cloud;
}

pcl::PointCloud<PointType> generate_box_cloud(double size) {
  pcl::PointCloud<PointType> box_cloud;
  PointType temp_point;
  double x, y, z;

  // use downsample resolution to compute how many points in a box
  int x_num = size / downsample_res;
  int y_num = size / downsample_res;
  int z_num = size / downsample_res;

  // for (int i = 0; i < x_num; i++)
  // {
  //   for (int j = 0; j < y_num; j++)
  //   {
  //     for (int k = 0; k < z_num; k++)
  //     {
  //       x = i * downsample_res - size / 2;
  //       y = j * downsample_res - size / 2;
  //       z = k * downsample_res - size / 2;
  //       temp_point.x = x;
  //       temp_point.y = y;
  //       temp_point.z = z;
  //       temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
  //       box_cloud.push_back(temp_point);
  //     }
  //   }
  // }

  // draw 2 xy plane
  for (int i = 0; i <= x_num; i++) {
    for (int j = 0; j <= y_num; j++) {
      x = i * downsample_res - size / 2;
      y = j * downsample_res - size / 2;
      z = -size / 2;
      temp_point.x = x;
      temp_point.y = y;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
      box_cloud.push_back(temp_point);
      z = size / 2;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
      box_cloud.push_back(temp_point);
    }
  }
  // draw 4 plane
  for (int k = 1; k < z_num; k++) {
    // draw x two lines
    for (int i = 0; i <= x_num; i++) {
      x = i * downsample_res - size / 2;
      y = -size / 2;
      z = k * downsample_res - size / 2;
      temp_point.x = x;
      temp_point.y = y;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
      box_cloud.push_back(temp_point);

      x = i * downsample_res - size / 2;
      y = size / 2;
      z = k * downsample_res - size / 2;
      temp_point.x = x;
      temp_point.y = y;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
      box_cloud.push_back(temp_point);
    }
    for (int j = 0; j <= y_num; j++) {
      x = -size / 2;
      y = j * downsample_res - size / 2;
      z = k * downsample_res - size / 2;
      temp_point.x = x;
      temp_point.y = y;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
      box_cloud.push_back(temp_point);

      x = size / 2;
      y = j * downsample_res - size / 2;
      z = k * downsample_res - size / 2;
      temp_point.x = x;
      temp_point.y = y;
      temp_point.z = z;
      temp_point.intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 2.0 / 2.0;
      box_cloud.push_back(temp_point);
    }
  }
  return box_cloud;
}

void generate_ptclouds_by_pos(Eigen::Vector3d obs_pos, int obs_type,
                              pcl::PointCloud<PointType> &obs_cloud,
                              vector<PointType> &obs_points) {
  // 0 for using uav model, 1 for using sphere model, 2 for using box model
  switch (obs_type) {
  case 0:
    if (use_uav_extra_model) {
      obs_cloud = uav_extra_model;
      for (int i = 0; i < obs_cloud.size(); i++) {
        obs_cloud.points[i].intensity = MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * 0.0 / 2.0;
      }
    } else {
      obs_cloud = generate_box_cloud(dynobject_size);
    }
    break;

  case 1:
    obs_cloud = generate_sphere_cloud(dynobject_size);
    break;

  case 2:
    obs_cloud = generate_box_cloud(dynobject_size);
    break;

  default:
    break;
  }

  for (int i = 0; i < obs_cloud.size(); i++) {
    obs_cloud.points[i].x = obs_cloud.points[i].x + obs_pos(0);
    obs_cloud.points[i].y = obs_cloud.points[i].y + obs_pos(1);
    obs_cloud.points[i].z = obs_cloud.points[i].z + obs_pos(2);
    obs_points.push_back(obs_cloud.points[i]);
  }
}

void dynobjGenerate() {
  if (dynobj_enable == 1) {
    dynobj_points.clear();
    dynobj_pointsindex.clear();
    dynobj_points_vis.clear();
    PointType temp_point;
    int dynpt_count = 0;
    double fly_time;
    Eigen::Vector3d gravity_vec;
    gravity_vec << 0, 0, -1;
    Eigen::Vector3d dyntemp_dir_polar;

    // rewrite generate dynamic obstacles
    for (int n = 0; n < dynobject_num; n++) {
      // according to motion mode, compute the positions of dynamic obstacle
      switch (dynobj_move_modes[n]) {
      case 0: {
        // constant speed mode
        fly_time = (ros_node->now() - dyn_start_time_vec[n]).seconds();
        dynobj_poss[n] = dynobj_poss[n] + fly_time * dynobj_dir[n];
        break;
      }

      case 1: {
        // constant gravity acceleration mode
        fly_time = (ros_node->now() - dyn_start_time_vec[n]).seconds();
        dynobj_dir[n] = dynobj_dir[n] + gravity_vec * fly_time;
        dynobj_poss[n] = dynobj_poss[n] + fly_time * dynobj_dir[n];
        break;
      }

      case 2: {
        // random walk mode
        fly_time = (ros_node->now() - dyn_start_time_vec[n]).seconds();
        dyntemp_dir_polar(0) = rand() / double(RAND_MAX) * 3.1415926;
        dyntemp_dir_polar(1) = (rand() / double(RAND_MAX) - 0.5) * 3.1415926;
        dynobj_dir[n](0) = dyn_velocity * sin(dyntemp_dir_polar(1));
        dynobj_dir[n](1) = dyn_velocity * cos(dyntemp_dir_polar(1)) * sin(dyntemp_dir_polar(0));
        dynobj_dir[n](2) = dyn_velocity * cos(dyntemp_dir_polar(1)) * cos(dyntemp_dir_polar(0));
        dynobj_poss[n] = dynobj_poss[n] + fly_time * dynobj_dir[n];
        break;
      }

      default:
        break;
      }

      // if dynamic obstacle is out of bounding box, regenerte one
      if (dynobj_poss[n](0) < map_min(0) || dynobj_poss[n](0) > map_max(0) ||
          dynobj_poss[n](1) < map_min(1) || dynobj_poss[n](1) > map_max(1) ||
          dynobj_poss[n](2) < map_min(2) || dynobj_poss[n](2) > map_max(2)) {
        dynobj_poss[n](0) = rand() / double(RAND_MAX) * (map_max(0) - map_min(0)) + map_min(0);
        dynobj_poss[n](1) = rand() / double(RAND_MAX) * (map_max(1) - map_min(1)) + map_min(1);
        dynobj_poss[n](2) = rand() / double(RAND_MAX) * (map_max(2) - map_min(2)) + map_min(2);
        Eigen::Vector3d dyntemp_dir_polar;
        dyntemp_dir_polar(0) = rand() / double(RAND_MAX) * 3.1415926;
        dyntemp_dir_polar(1) = (rand() / double(RAND_MAX) - 0.5) * 3.1415926;
        dynobj_dir[n](0) = dyn_velocity * sin(dyntemp_dir_polar(1));
        dynobj_dir[n](1) = dyn_velocity * cos(dyntemp_dir_polar(1)) * sin(dyntemp_dir_polar(0));
        dynobj_dir[n](2) = dyn_velocity * cos(dyntemp_dir_polar(1)) * cos(dyntemp_dir_polar(0));
      }
      dyn_start_time_vec[n] = ros_node->now();
      // generate point cloud of dynamic obstacle
      pcl::PointCloud<PointType> dynobj_cloud;
      vector<PointType> temp_dynobj_points;
      generate_ptclouds_by_pos(dynobj_poss[n], dynobj_types[n], dynobj_cloud, temp_dynobj_points);
      dynobj_points_vis = dynobj_points_vis + dynobj_cloud;
      dynobj_points.insert(dynobj_points.end(), temp_dynobj_points.begin(),
                           temp_dynobj_points.end());
      // for (int i = 0; i < temp_dynobj_points.size(); i++)
      // {
      //   dynobj_pointsindex.push_back(dynpt_count + origin_mapptcount);
      //   dynpt_count++;
      // }
    }

    RCLCPP_INFO(ros_node->get_logger(), "dynobj points size = %zu", dynobj_points_vis.points.size());

    dynobj_points_vis.width = dynobj_points_vis.points.size();
    dynobj_points_vis.height = 1;
    dynobj_points_vis.is_dense = true;

    // sensor_msgs::PointCloud2 dynobj_points_pcd;
    // pcl::toROSMsg(dynobj_points_vis, dynobj_points_pcd);
    // dynobj_points_pcd.header = odom_.header;
    // dynobj_points_pcd.header.frame_id = "world";
    // pub_dyncloud.publish(dynobj_points_pcd);

    kdtree_dyn.setInputCloud(dynobj_points_vis.makeShared());
    // has_dyn_map = true;

    // dyn_start_time = ros::Time::now();
  }
}

void rcvOdometryCallback(const nav_msgs::msg::Odometry::ConstSharedPtr message) {
  /*if(!has_global_map)
    return;*/
  has_odom = true;
  odom_ = *message;
  const auto &odom = *message;

  Matrix4f body2world = Matrix4f::Identity();

  Eigen::Vector3f request_position;
  Eigen::Quaternionf pose;
  pose.x() = odom.pose.pose.orientation.x;
  pose.y() = odom.pose.pose.orientation.y;
  pose.z() = odom.pose.pose.orientation.z;
  pose.w() = odom.pose.pose.orientation.w;
  body2world.block<3, 3>(0, 0) = pose.toRotationMatrix();
  body2world(0, 3) = odom.pose.pose.position.x;
  body2world(1, 3) = odom.pose.pose.position.y;
  body2world(2, 3) = odom.pose.pose.position.z;

  // convert to cam pose
  sensor2world = body2world * sensor2body;

  // collision check function
  // running time statistics
  const auto t1 = std::chrono::steady_clock::now();

  if (collisioncheck_enable) {
    collision_points.points.clear();
    PointType searchPoint;
    searchPoint.x = odom.pose.pose.position.x;
    searchPoint.y = odom.pose.pose.position.y;
    searchPoint.z = odom.pose.pose.position.z;
    if (_kdtreeLocalMap.radiusSearch(searchPoint, collision_range, pointIdxRadiusSearch,
                                     pointRadiusSquaredDistance) > 0) {
      if (use_uav_extra_model) {
        // check if the neighbor points are in the uav model
        bool collision = false;
        for (int i = 0; i < pointIdxRadiusSearch.size(); i++) {
          int index = pointIdxRadiusSearch[i];
          PointType p = cloud_all_map.points[index];

          // transform the point to the uav model coordinate
          Eigen::Vector3f p_eigen(p.x, p.y, p.z);
          p_eigen =
          body2world.block<3, 3>(0, 0).transpose() * (p_eigen - body2world.block<3, 1>(0, 3));

          int hash_x = (p_eigen(0) - uav_modelmin.x) / downsample_res;
          int hash_y = (p_eigen(1) - uav_modelmin.y) / downsample_res;
          int hash_z = (p_eigen(2) - uav_modelmin.z) / downsample_res;
          int hash_index = hash_x + hash_y * uavhash_xsize + hash_z * uavhash_xsize * uavhash_ysize;
          if (uavpoint_hashmap.find(hash_index) != uavpoint_hashmap.end()) {
            collision = true;
            RCLCPP_ERROR(ros_node->get_logger(), "ENVIRONMENT COLLISION DETECTED!!!");
            // break;
            collision_points.points.push_back(p);
          }
        }
      } else {
        RCLCPP_ERROR(ros_node->get_logger(), "ENVIRONMENT COLLISION DETECTED!!!");
      }
    }
    if (dynobj_enable || drone_num > 1) {
      if (kdtree_dyn.radiusSearch(searchPoint, collision_range, pointIdxRadiusSearch,
                                  pointRadiusSquaredDistance) > 0) {
        if (use_uav_extra_model) {
          // check if the neighbor points are in the uav model
          bool collision = false;
          for (int i = 0; i < pointIdxRadiusSearch.size(); i++) {
            int index = pointIdxRadiusSearch[i];
            PointType p = cloud_all_map.points[index];
            int hash_x = (p.x - uav_modelmin.x) / downsample_res;
            int hash_y = (p.y - uav_modelmin.y) / downsample_res;
            int hash_z = (p.z - uav_modelmin.z) / downsample_res;
            int hash_index =
            hash_x + hash_y * uavhash_xsize + hash_z * uavhash_xsize * uavhash_ysize;
            if (uavpoint_hashmap.find(hash_index) != uavpoint_hashmap.end()) {
              collision = true;
              RCLCPP_ERROR(ros_node->get_logger(), "DYNAMIC OBSTACLES COLLISION DETECTED!!!");
              // break;
              collision_points.points.push_back(p);
            }
          }
        } else {
          RCLCPP_ERROR(ros_node->get_logger(), "DYNAMIC OBSTACLES COLLISION DETECTED!!!");
        }
      }
    }
  }

  // publish collision points
  sensor_msgs::msg::PointCloud2 collision_points_vis_pcd;
  pcl::toROSMsg(collision_points, collision_points_vis_pcd);
  collision_points_vis_pcd.header.frame_id = "world";
  collision_points_vis_pcd.header.stamp = ros_node->now();
  pub_collisioncloud->publish(collision_points_vis_pcd);

  const auto t2 = std::chrono::steady_clock::now();
  collision_check_time_sum += std::chrono::duration<double>(t2 - t1).count();
  collision_check_time_count++;
  if (collision_check_time_count == 100) {
    // ROS_INFO("collision check time: %f", collision_check_time_sum / collision_check_time_count);
    collision_check_time_sum = 0;
    collision_check_time_count = 0;
  }
}

// decentralize simulation, get other uav odometry and generate point clouds
void multiOdometryCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg, int drone_id) {
  Eigen::Vector3d uav_pos;
  uav_pos(0) = msg->pose.pose.position.x;
  uav_pos(1) = msg->pose.pose.position.y;
  uav_pos(2) = msg->pose.pose.position.z;
  other_uav_pos[drone_id] = uav_pos;
  other_uav_rcv_time[drone_id] = rclcpp::Time(msg->header.stamp).seconds();

  otheruav_points[drone_id].clear();
  otheruav_pointsindex[drone_id].clear();
  otheruav_points_vis.clear();

  PointType temp_point;
  int uavpt_count = 0;

  if (use_uav_extra_model == 0) {
    // generate uav point cloud
    double x_min = uav_pos(0) - 0.5 * uav_size[0];
    double y_min = uav_pos(1) - 0.5 * uav_size[1];
    double z_min = uav_pos(2) - 0.5 * uav_size[2];

    for (int i = 0; i < drone_drawpoints_num[0]; i++) {
      for (int j = 0; j < drone_drawpoints_num[1]; j++) {
        for (int k = 0; k < drone_drawpoints_num[2]; k++) {
          temp_point.x = i * downsample_res + x_min;
          temp_point.y = j * downsample_res + y_min;
          temp_point.z = k * downsample_res + z_min;
          temp_point.intensity =
          ((float)(MAX_INTENSITY - MIN_INTENSITY)) * ((drone_id + 1.0) / (float(drone_num))) +
          MIN_INTENSITY; // set the intensity of the point
          otheruav_points[drone_id].push_back(temp_point);
          // otheruav_pointsindex[drone_id].push_back(uavpt_count + origin_mapptcount + 100000 +
          // drone_id * uav_points_num);
          otheruav_points_vis.push_back(temp_point);
          uavpt_count++;
        }
      }
    }
  } else {
    // read pcd model from file and attach it to UAVs odom
    Eigen::Quaterniond pose;
    pose.x() = msg->pose.pose.orientation.x;
    pose.y() = msg->pose.pose.orientation.y;
    pose.z() = msg->pose.pose.orientation.z;
    pose.w() = msg->pose.pose.orientation.w;
    Eigen::Matrix3d uav_rot = pose.toRotationMatrix();
    for (int i = 0; i < uav_extra_model.points.size(); i++) {
      Eigen::Vector3d model_point;
      model_point(0) = uav_extra_model.points[i].x;
      model_point(1) = uav_extra_model.points[i].y;
      model_point(2) = uav_extra_model.points[i].z;
      model_point = uav_rot * model_point + uav_pos;
      temp_point.x = model_point(0);
      temp_point.y = model_point(1);
      temp_point.z = model_point(2);
      temp_point.intensity =
      ((float)(MAX_INTENSITY - MIN_INTENSITY)) * ((drone_id + 1.0) / (float(drone_num))) +
      MIN_INTENSITY; // set the intensity of the point
      otheruav_points[drone_id].push_back(temp_point);
      otheruav_points_vis.push_back(temp_point);
      uavpt_count++;
    }
  }

  // publish uav point cloud
  // sensor_msgs::PointCloud2 otheruav_points_vis_pcd;
  // pcl::toROSMsg(otheruav_points_vis, otheruav_points_vis_pcd);
  // otheruav_points_vis_pcd.header.frame_id = "world";
  // otheruav_points_vis_pcd.header.stamp = ros::Time::now();
  // pub_uavcloud.publish(otheruav_points_vis_pcd);
}

int comp_time_count = 0;

void rcvGlobalPointCloudCallBack(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud_map) {
  if (has_global_map)
    return;

  RCLCPP_WARN(ros_node->get_logger(), "Global Pointcloud received for local sensing renderer.");

  pcl::PointCloud<PointType> cloud_input;
  const bool has_intensity = std::any_of(
      pointcloud_map->fields.begin(), pointcloud_map->fields.end(),
      [](const sensor_msgs::msg::PointField &field) { return field.name == "intensity"; });
  if (has_intensity)
  {
    pcl::fromROSMsg(*pointcloud_map, cloud_input);
  }
  else
  {
    pcl::PointCloud<pcl::PointXYZ> xyz_cloud;
    pcl::fromROSMsg(*pointcloud_map, xyz_cloud);
    cloud_input.reserve(xyz_cloud.size());
    for (const auto &point : xyz_cloud.points)
    {
      PointType point_with_intensity;
      point_with_intensity.x = point.x;
      point_with_intensity.y = point.y;
      point_with_intensity.z = point.z;
      point_with_intensity.intensity = 0.0F;
      cloud_input.push_back(point_with_intensity);
    }
    cloud_input.width = static_cast<std::uint32_t>(cloud_input.size());
    cloud_input.height = 1;
    cloud_input.is_dense = xyz_cloud.is_dense;
  }
  if (cloud_input.empty()) {
    RCLCPP_WARN(ros_node->get_logger(), "Received empty global pointcloud, waiting for next map.");
    return;
  }

  if (useLidarSensor()) {
    render.read_pointcloud_fromcloud(cloud_input);
  }
  if (!preparePinholeMap(cloud_input)) {
    return;
  }
  has_global_map = true;
}

void renderSensedPoints() {
  const auto t1 = std::chrono::steady_clock::now();

  if (!has_odom || !has_global_map)
    return;

  if (useDepthSensor()) {
    publishCameraPose(odom_);
    publishPinholeDepth(rclcpp::Time(odom_.header.stamp));
    return;
  }

  Eigen::Quaternionf q;
  q.x() = odom_.pose.pose.orientation.x;
  q.y() = odom_.pose.pose.orientation.y;
  q.z() = odom_.pose.pose.orientation.z;
  q.w() = odom_.pose.pose.orientation.w;
  // add pitchW
  double theta = lidar_pitch * M_PI / 180.0;
  Eigen::Matrix3f rot_body2lidar;
  rot_body2lidar << cos(theta), 0, sin(theta), 0, 1, 0, -sin(theta), 0, cos(theta);

  // rotate lidar
  // Eigen::Vector3d eulerAngle_body2lidar(0,0.5236,0);
  // Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle_body2lidar(0),Vector3d::UnitX()));
  // Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle_body2lidar(1),Vector3d::UnitY()));
  // Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle_body2lidar(2),Vector3d::UnitZ()));
  // rot_body2lidar=yawAngle*pitchAngle*rollAngle;
  // rotate lidar end

  const Eigen::Matrix3f rot(q.toRotationMatrix() * rot_body2lidar);
  Eigen::Quaternionf quat(rot);
  // const Eigen::Matrix3f rot(q.toRotationMatrix());
  const Eigen::Vector3f pos(odom_.pose.pose.position.x, odom_.pose.pose.position.y,
                            odom_.pose.pose.position.z);
  const rclcpp::Time time_stamp_(odom_.header.stamp);

  const rclcpp::Time t_pattern = ros_node->now();
  double time_frominit = (t_pattern - t_init).seconds();

  local_map->points.clear();
  dynamic_input_points.points.clear();

  for (int i = 0; i < drone_num; i++) {
    if (i == drone_id) {
      continue;
    }
    for (int j = 0; j < otheruav_points[i].size(); j++) {
      dynamic_input_points.points.push_back(otheruav_points[i][j]);
    }
    // dynamic_input_points = dynamic_input_points + otheruav_points[i];
  }
  dynamic_input_points = dynamic_input_points + dynobj_points_vis;

  if (dynamic_input_points.points.size() > 0) {
    kdtree_dyn.setInputCloud(dynamic_input_points.makeShared());
  }

  // trans and publish the dynamic point cloud
  sensor_msgs::msg::PointCloud2 dynamic_input_points_pcd;
  pcl::toROSMsg(dynamic_input_points, dynamic_input_points_pcd);
  dynamic_input_points_pcd.header.frame_id = "world";
  dynamic_input_points_pcd.header.stamp = ros_node->now();
  pub_dyncloud->publish(dynamic_input_points_pcd);

  // dynamic_input_points = otheruav_points_vis + dyn_points_vis;
  // ROS_INFO("dynamic_input_points size = %d", dynamic_input_points.size());
  render.input_dyn_clouds(dynamic_input_points);

  render.render_pointcloud(local_map, pos, quat, time_frominit);

  const auto t_afterrender = std::chrono::steady_clock::now();
  double comp_time_temp = std::chrono::duration<double>(t_afterrender - t1).count();
  comp_time_vec.push_back(comp_time_temp);
  if (comp_time_count > 20) {
    comp_time_vec.pop_front();
    geometry_msgs::msg::PoseStamped totaltime_pub;
    totaltime_pub.pose.position.x =
    accumulate(comp_time_vec.begin(), comp_time_vec.end(), 0.0) / comp_time_vec.size();
    comp_time_pub->publish(totaltime_pub);
    // ROS_INFO("Temp compute time = %lf, average compute time = %lf",
    // comp_time_temp,totaltime_pub.pose.position.x);
  } else {
    comp_time_count++;
  }

  local_map->width = local_map->points.size();
  local_map->height = 1;
  local_map->is_dense = true;

  pcl::PointCloud<PointType> local_map_vis;
  local_map_vis.points.reserve(local_map->points.size());
  for (const auto &pt : local_map->points) {
    const Eigen::Vector3f pt_vec(pt.x, pt.y, pt.z);
    if ((pt_vec - pos).norm() >= sensing_horizon - 1.0) {
      continue;
    }
    local_map_vis.points.push_back(pt);
  }
  local_map_vis.width = local_map_vis.points.size();
  local_map_vis.height = 1;
  local_map_vis.is_dense = true;

  // std::cout<< "local map size: " << local_map->points.size() << std::endl;

  pcl::toROSMsg(local_map_vis, local_map_pcd);
  local_map_pcd.header = odom_.header;
  local_map_pcd.header.stamp = time_stamp_;
  local_map_pcd.header.frame_id = "world";
  pub_cloud->publish(local_map_pcd);

  // transform
  std::string sensor_frame_id_ = "sensor";
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = time_stamp_;
  transform.header.frame_id = "world";
  transform.child_frame_id = sensor_frame_id_;

  transform.transform.translation.x = pos.x();
  transform.transform.translation.y = pos.y();
  transform.transform.translation.z = pos.z();
  transform.transform.rotation.x = q.x();
  transform.transform.rotation.y = q.y();
  transform.transform.rotation.z = q.z();
  transform.transform.rotation.w = q.w();

  tf_broadcaster->sendTransform(transform);

  Eigen::Matrix4f sensor2world;
  sensor2world << rot(0, 0), rot(0, 1), rot(0, 2), pos.x(), rot(1, 0), rot(1, 1), rot(1, 2),
  pos.y(), rot(2, 0), rot(2, 1), rot(2, 2), pos.z(), 0, 0, 0, 1;
  Eigen::Matrix4f world2sensor;
  world2sensor = sensor2world.inverse();

  // body frame pointcloud
  point_in_sensor.points.clear();
  // pcl::copyPointCloud(local_map_filled, point_in_sensor);
  pcl::transformPointCloud(*local_map, point_in_sensor, world2sensor);
  point_in_sensor.width = point_in_sensor.points.size();
  point_in_sensor.height = 1;
  point_in_sensor.is_dense = true;

  // pcl::toROSMsg(point_in_sensor, sensor_map_pcd);
  // sensor_map_pcd.header.frame_id = sensor_frame_id_;
  // sensor_map_pcd.header.stamp = time_stamp_;
  // pub_intercloud.publish(sensor_map_pcd);
  publishLidarPose(odom_, time_stamp_, pos, quat);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  ros_node = std::make_shared<rclcpp::Node>("opengl_render_node");
  render.setShaderDirectory(
      ament_index_cpp::get_package_share_directory("local_sensing_node") + "/shaders");

  quad_name = ros_node->declare_parameter<std::string>("quadrotor_name", "quad_0");
  sensor_type_ = ros_node->declare_parameter<std::string>("sensor_type", "depth");
  if (!useDepthSensor() && !useLidarSensor()) {
    RCLCPP_ERROR(ros_node->get_logger(), "Unsupported sensor_type '%s'. Expected 'depth' or 'lidar'.", sensor_type_.c_str());
    rclcpp::shutdown();
    return 1;
  }
  RCLCPP_INFO(ros_node->get_logger(), "opengl_render_node sensor_type: %s", sensor_type_.c_str());
  is_360lidar = ros_node->declare_parameter<int>("is_360lidar", 1);
  sensing_horizon = ros_node->declare_parameter<double>("sensing_horizon", 40.0);
  sensing_rate = ros_node->declare_parameter<double>("sensing_rate", 10.0);
  estimation_rate = ros_node->declare_parameter<double>("estimation_rate", 10.0);
  polar_resolution = ros_node->declare_parameter<double>("polar_resolution", 0.2);
  yaw_fov = ros_node->declare_parameter<double>("yaw_fov", 360.0);
  vertical_fov = ros_node->declare_parameter<double>("vertical_fov", 90.0);
  min_raylength = ros_node->declare_parameter<double>("min_raylength", 1.0);
  downsample_res = ros_node->declare_parameter<double>("downsample_res", 0.1);
  livox_linestep = ros_node->declare_parameter<double>("livox_linestep", 1.4);
  use_avia_pattern = ros_node->declare_parameter<int>("use_avia_pattern", 0);
  curvature_limit = ros_node->declare_parameter<double>("curvature_limit", 100.0);
  hash_cubesize = ros_node->declare_parameter<double>("hash_cubesize", 5.0);
  use_vlp32_pattern = ros_node->declare_parameter<int>("use_vlp32_pattern", 0);
  use_minicf_pattern = ros_node->declare_parameter<int>("use_minicf_pattern", 1);
  use_os128_pattern = ros_node->declare_parameter<int>("use_os128_pattern", 0);
  use_os64_pattern = ros_node->declare_parameter<int>("use_os64_pattern", 0);
  lidar_pitch = ros_node->declare_parameter<double>("lidar_pitch", 45.0);
  cam_width = ros_node->declare_parameter<int>("cam_width", cam_width);
  cam_height = ros_node->declare_parameter<int>("cam_height", cam_height);
  cam_fx = ros_node->declare_parameter<double>("cam_fx", cam_fx);
  cam_fy = ros_node->declare_parameter<double>("cam_fy", cam_fy);
  cam_cx = ros_node->declare_parameter<double>("cam_cx", cam_cx);
  cam_cy = ros_node->declare_parameter<double>("cam_cy", cam_cy);

  use_gaussian_filter = ros_node->declare_parameter<int>("use_gaussian_filter", 0);

  // dyn parameters
  dynobj_enable = ros_node->declare_parameter<int>("dynamic.enable", 0);
  dynobject_size = ros_node->declare_parameter<double>("dynamic.object_size", 0.0);
  dynobject_num = ros_node->declare_parameter<int>("dynamic.object_count", 0);
  dyn_mode = ros_node->declare_parameter<int>("dynamic.mode", 0);
  dyn_velocity = ros_node->declare_parameter<double>("dynamic.velocity", 0.0);
  use_uav_extra_model = ros_node->declare_parameter<int>("use_uav_extra_model", 0);
  collisioncheck_enable = ros_node->declare_parameter<int>("collision_check.enable", 0);
  collision_range = ros_node->declare_parameter<double>("collision_check.range", 0.3);
  output_pcd = ros_node->declare_parameter<int>("output_pcd", 0);
  use_global_map_topic = ros_node->declare_parameter<bool>("use_global_map_topic", false);
  file_name = ros_node->declare_parameter<std::string>("pcd_map_file", "");
  x_size = ros_node->declare_parameter<double>("map.x_size", 100.0);
  y_size = ros_node->declare_parameter<double>("map.y_size", 100.0);
  z_size = ros_node->declare_parameter<double>("map.z_size", 10.0);
  resolution = ros_node->declare_parameter<double>("map.resolution", 0.1);

  // subscribe other uav pos
  drone_num = ros_node->declare_parameter<int>("uav_num", 1);
  drone_id = ros_node->declare_parameter<int>("drone_id", 0);
  other_uav_pos.resize(drone_num);
  other_uav_rcv_time.resize(drone_num);
  otheruav_points.resize(drone_num);
  otheruav_pointsindex.resize(drone_num);
  for (int i = 0; i < drone_num; i++) {
    if (i == drone_id) {
      continue;
    }
    string topic = "/quad_";
    topic += to_string(i);
    topic += "/body_pose";
    cout << topic << endl;
    other_odom_subs.push_back(ros_node->create_subscription<nav_msgs::msg::Odometry>(
        topic, rclcpp::SensorDataQoS(),
        [i](nav_msgs::msg::Odometry::ConstSharedPtr msg) { multiOdometryCallback(msg, i); }));
  }

  pcl::VoxelGrid<PointType> sor;
  int pcd_read_status;
  if (use_uav_extra_model) {
    string uav_model_path;
    uav_model_path = ament_index_cpp::get_package_share_directory("odom_visualization"); //=
                           //"/home/mars/catkin_ws2/src/Exploration_sim/octomap_mapping/octomap_server"
    uav_model_path.append("/meshes/yunque001.pcd");
    std::cout << "\nFound pkg_path = " << uav_model_path << std::endl;

    pcd_read_status = pcl::io::loadPCDFile<PointType>(uav_model_path, uav_extra_model);
    if (pcd_read_status == -1) {
      cout << "can't read uav extra model file." << endl;
      return 0;
    }
    // downsample uav model
    sor.setInputCloud(uav_extra_model.makeShared());
    sor.setLeafSize(downsample_res, downsample_res, downsample_res);
    sor.filter(uav_extra_model);
  }

  uav_size[0] = 0.4;
  uav_size[1] = 0.4;
  uav_size[2] = 0.3;
  drone_drawpoints_num[0] = (uav_size[0] / downsample_res);
  drone_drawpoints_num[1] = (uav_size[1] / downsample_res);
  drone_drawpoints_num[2] = (uav_size[2] / downsample_res);
  uav_points_num = drone_drawpoints_num[0] * drone_drawpoints_num[1] * drone_drawpoints_num[2];
  RCLCPP_INFO(ros_node->get_logger(), "drone_drawpoints_num = %d,%d,%d, Uav points num = %d", drone_drawpoints_num[0],
              drone_drawpoints_num[1], drone_drawpoints_num[2], uav_points_num);

  // render.setParameters(400,400,250,250,downsample_res,0.1,sensing_horizon,sensing_rate,use_avia_pattern);
  int image_width;
  int image_height;
  if (is_360lidar) {
    image_width = ceil(360.0 / polar_resolution);
  } else {
    image_width = ceil(yaw_fov / polar_resolution);
  }
  image_height = ceil(vertical_fov / polar_resolution);
  render.setParameters(image_width, image_height, 250, 250, downsample_res, polar_resolution,
                       yaw_fov, vertical_fov, 0.1, sensing_horizon, sensing_rate, use_avia_pattern,
                       use_os128_pattern, use_minicf_pattern);

  if (use_global_map_topic) {
    const auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    global_map_sub = ros_node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "global_map", map_qos, rcvGlobalPointCloudCallBack);
  } else {
    if (file_name.empty()) {
      RCLCPP_ERROR(ros_node->get_logger(),
                   "pcd_map_file must name an existing PCD file when use_global_map_topic is false.");
      rclcpp::shutdown();
      return 1;
    }
    if (useLidarSensor()) {
      render.read_pointcloud_fromfile(file_name);
    }
    if (!loadPinholeMapFromFile(file_name)) {
      return 1;
    }
    has_global_map = true;
  }
  // home/dji/kong_ws/src/Exploration_sim/uav_simulator/map_generator/resource/Knowles_merge_01cutoff.pcd
  // /home/dji/meshmap/Knowles_local_sor_001.pcd

  if (collisioncheck_enable) {
    if (cloud_all_map.empty()) {
      if (file_name.empty() || !loadPinholeMapFromFile(file_name)) {
        cout << "can't read global" << endl;
        return 0;
      }
    }

    if (cloud_all_map.empty()) {
      RCLCPP_ERROR(ros_node->get_logger(), "Global map is empty.");
      return 0;
    }

    RCLCPP_INFO(ros_node->get_logger(), "Global map and kdtree ready.");

    if (use_uav_extra_model) {
      // get max and min xyz of uav model
      pcl::getMinMax3D(uav_extra_model, uav_modelmin, uav_modelmax);
      uavhash_xsize = ceil((uav_modelmax.x - uav_modelmin.x) / downsample_res);
      uavhash_ysize = ceil((uav_modelmax.y - uav_modelmin.y) / downsample_res);
      uavhash_zsize = ceil((uav_modelmax.z - uav_modelmin.z) / downsample_res);

      // input the uav model point clouds into voxel grid hash map
      for (int i = 0; i < uav_extra_model.points.size(); i++) {
        PointType p = uav_extra_model.points[i];
        int x = floor((p.x - uav_modelmin.x) / downsample_res);
        int y = floor((p.y - uav_modelmin.y) / downsample_res);
        int z = floor((p.z - uav_modelmin.z) / downsample_res);
        long int hash = x + y * uavhash_xsize + z * uavhash_xsize * uavhash_ysize;
        uavpoint_hashmap[hash] = true;
      }

      RCLCPP_INFO(ros_node->get_logger(), "UAV model read and input hash map done. UAV model size = %d,%d,%d", uavhash_xsize,
                  uavhash_ysize, uavhash_zsize);
    }
  }

  if (dynobj_enable == 1) {
    if (!collisioncheck_enable && cloud_all_map.empty()) {
      if (file_name.empty() || !loadPinholeMapFromFile(file_name)) {
        cout << "can't read global" << endl;
        return 0;
      }
    }

    map_min(0) = global_mapmin.x;
    map_min(1) = global_mapmin.y;
    map_min(2) = global_mapmin.z;
    map_max(0) = global_mapmax.x;
    map_max(1) = global_mapmax.y;
    map_max(2) = global_mapmax.z;

    // ROS_INFO("Global map min x = %f, max x = %f" , global_mapmin.x , global_mapmax.x);

    // dynamic objects initial pos generate
    srand((unsigned)time(NULL));
    for (int i = 0; i < dynobject_num; i++) {
      Eigen::Vector3d dyntemp_pos;
      dyntemp_pos(0) =
      rand() / double(RAND_MAX) * (global_mapmax.x - global_mapmin.x) + global_mapmin.x;
      dyntemp_pos(1) =
      rand() / double(RAND_MAX) * (global_mapmax.y - global_mapmin.y) + global_mapmin.y;
      dyntemp_pos(2) =
      rand() / double(RAND_MAX) * (global_mapmax.z - global_mapmin.z) + global_mapmin.z;
      dynobj_poss.push_back(dyntemp_pos);

      // random generate dynamic object type and motion mode
      int dynobj_type = rand() % 3;
      dynobj_types.push_back(dynobj_type);
      // int dynobj_motion_mode = rand() % 2;
      dynobj_move_modes.push_back(dynobj_type);

      Eigen::Vector3d dyntemp_dir, dyntemp_dir_polar;
      dyntemp_dir_polar(0) = rand() / double(RAND_MAX) * 3.1415926;
      dyntemp_dir_polar(1) = (rand() / double(RAND_MAX) - 0.5) * 3.1415926;
      dyntemp_dir[2] = dyn_velocity * sin(dyntemp_dir_polar(1));
      dyntemp_dir[1] = dyn_velocity * cos(dyntemp_dir_polar(1)) * sin(dyntemp_dir_polar(0));
      dyntemp_dir[0] = dyn_velocity * cos(dyntemp_dir_polar(1)) * cos(dyntemp_dir_polar(0));
      dynobj_dir.push_back(dyntemp_dir);

      if (dyn_obs_diff_size_on) {
        double dynobject_rand_size =
        rand() / double(RAND_MAX) * (dynobject_size - downsample_res) + downsample_res;
        dyn_obs_size_vec.push_back(dynobject_rand_size);
      }

      // dyn_start_time = ros::Time::now();
      dyn_start_time_vec.push_back(ros_node->now());
    }
  }

  // subscribe point cloud
  const auto body_pose_topic = ros_node->declare_parameter<std::string>("body_pose_topic", "body_pose");
  odom_sub = ros_node->create_subscription<nav_msgs::msg::Odometry>(
      body_pose_topic, rclcpp::SensorDataQoS(), rcvOdometryCallback);

  // publishers
  //   pub_dyncloud = nh.advertise<sensor_msgs::PointCloud2>("/pcl_render_node/dyn_cloud", 10);
  // pub_intercloud = nh.advertise<sensor_msgs::PointCloud2>("/pcl_render_node/explored_cloud", 10);
  pub_dyncloud = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>("dyn_cloud", rclcpp::SensorDataQoS());
  if (useDepthSensor()) {
    camera_pose_pub_ = ros_node->create_publisher<nav_msgs::msg::Odometry>(quadTopic("camera_pose"), rclcpp::SensorDataQoS());
    pinhole_depth_pub_ = ros_node->create_publisher<sensor_msgs::msg::Image>("depth", rclcpp::SensorDataQoS());
  } else {
    pub_intercloud = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>("sensor_cloud", rclcpp::SensorDataQoS());
    pub_cloud = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>("cloud", rclcpp::SensorDataQoS());
    lidar_pose_pub_ = ros_node->create_publisher<nav_msgs::msg::Odometry>(quadTopic("lidar_pose"), rclcpp::SensorDataQoS());
  }
  pub_uavcloud = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>("uav_cloud", rclcpp::SensorDataQoS());
  comp_time_pub = ros_node->create_publisher<geometry_msgs::msg::PoseStamped>("simulator_compute_time", 10);
  pub_collisioncloud = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>("collision_cloud", rclcpp::SensorDataQoS());
  double sensing_duration = 1.0 / sensing_rate;

  t_init = ros_node->now();
  tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(ros_node);

  local_sensing_timer = ros_node->create_wall_timer(
      std::chrono::duration<double>(sensing_duration), renderSensedPoints);
  dynobj_timer = ros_node->create_wall_timer(
      std::chrono::duration<double>(sensing_duration), dynobjGenerate);

  sensor2body << 0.0, 0.0, 1.0, 0.0,
      -1.0, 0.0, 0.0, 0.0,
      0.0, -1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0;

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(ros_node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
