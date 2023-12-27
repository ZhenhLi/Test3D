#include <iostream>
#include <open3d/Open3D.h>
// #include <pcl/io/io.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/io/ply_io.h>
// #include <pcl/visualization/cloud_viewer.h>

int testCase1() {
  std::cout << "========== open3d testCase1 start ==========" << std::endl;

  open3d::core::Tensor t = open3d::core::Tensor::Ones(open3d::core::SizeVector({4,4}),
                                                      open3d::core::Float64);
  std::cout << t.ToString() << std::endl;
  Eigen::Matrix4d matrix = open3d::core::eigen_converter::TensorToEigenMatrixXd(t);
  std::cout << t.ToString() << std::endl;

  std::cout << "========== open3d testCase1 done  ==========" << std::endl;
  return 0;
}

// void viewerOneOff(pcl::visualization::PCLVisualizer& viewer) {
//   viewer.setBackgroundColor(1.0, 0.5, 1.0);
// }
int testCase2() {
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  // char filePath[256] = ".ply";
  // if (-1 == pcl::io::loadPLYFile(filePath, *cloud)) {
  //   std::cout << "error input!" << std::endl;
  //   return -1;
  // }

  // std::cout << cloud->points.size() << std::endl;
  // pcl::visualization::CloudViewer viewer("Cloud viewer");

  // viewer.showCloud(cloud);
  // viewer.runOnVisualizationThreadOnce(viewerOneOff);
  // system("pause");
  return 0;
}

int testCase3_upsampling() {

  return 0;
}

int main() {
  testCase1();
  testCase2();

  return 0;
}