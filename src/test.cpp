#include <iostream>
#include <open3d/Open3D.h>

int testCase1() {
  std::cout << "--------- open3d testcase1 start ---" << std::endl;
  auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0);
  sphere->ComputeVertexNormals();
  sphere->PaintUniformColor({0.0, 1.0, 0.0});
  open3d::visualization::DrawGeometries({sphere}); // 球可以显示

  auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
  if (!open3d::io::ReadPointCloud("data3d/Armadillo.ply", *cloudPtr)) {
    return -1;
  }

  //点数
	int pointCount = cloudPtr->points_.size();
	// ui.listWidget->addItem("点数：" + QString::number(pointCount));

  // //包围盒
	// Eigen::Vector3d min_bound = cloud_ptr->GetMinBound();
	// double minX = min_bound(0);
	// double minY = min_bound(1);
	// double minZ = min_bound(2);
	// ui.listWidget->addItem("minX = " + QString::number(minX, 'f', 4) + 
  //           ", minY = " + QString::number(minY, 'f', 4) + ", minZ = " + QString::number(minZ, 'f', 4));
	// Eigen::Vector3d max_bound = cloud_ptr->GetMaxBound();
	// double maxX = max_bound(0);
	// double maxY = max_bound(1);
	// double maxZ = max_bound(2);
	// ui.listWidget->addItem("maxX = " + QString::number(maxX, 'f', 4) + 
  //           ", maxY = " + QString::number(maxY, 'f', 4) + ", maxZ = " + QString::number(maxZ, 'f', 4));

  /* visualize PC */
	open3d::visualization::DrawGeometries({ cloudPtr });

  // open3d::geometry::PointCloud pc1;
  // open3d::io::ReadPointCloud("data3d/Armadillo.ply", pc1);
  // Eigen::Vector3d colorSource(1, 0.706, 0);
  // pc1.PaintUniformColor(colorSource);
  // std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometry_ptrs;
  // auto cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
  // open3d::io::ReadPointCloud("data3d/Armadillo.ply", cloud_ptr);
  // geometry_ptrs.push_back(cloud_ptr);
  // open3d::visualization::DrawGeometries(geometry_ptrs);

  std::cout << "--------- open3d testcase1 done ----" << std::endl;
  return 0;
}

int main() {
  std::cout << "--------- open3d test -------------" << std::endl;
  testCase1();
  return 0;
}