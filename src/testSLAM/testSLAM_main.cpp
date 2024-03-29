#include <iostream>
#include <cmath>
#include <Eigen/Eigen>

int testCase1() {
  // c++新特性

  // 列表初始化

  // 自动类型a推导

  // 循环体

  // lambda表达式

  // tuple 可随心所欲变长的参数模板

  // 实际编程代码例子

  return 0;
}

int testCase2_chat3() {
  // 旋转向量(轴角)： 沿着Z轴旋转45°
  Eigen::AngleAxisd rotation_vector( M_PI / 4, Eigen::Vector3d(0, 0, 1));
  std::cout << "旋转向量的旋转轴 = " << rotation_vector.axis() << "\n旋转向量角度 = " << rotation_vector.angle() << std::endl;

  // 旋转矩阵：沿Z轴旋转45°
  Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
  rotation_matrix << 0.707, -0.707, 0,
                     0.707,  0.707, 0,
                     0,      0,     1;
  std::cout << "旋转矩阵=\n" << rotation_matrix << std::endl;
  
  // 四元数： 沿z轴旋转45°
  Eigen::Quaterniond quat = Eigen::Quaterniond(0, 0, 0.383, 0.924);
  std::cout << "四元数输出方法 1: 四元数 = \n" << quat.coeffs() << std::endl;
  // coeffs的顺序为 x y z w, w为实部，其他为虚部
  std::cout << "四元数输出方法2: 四元数 = \n x=" << quat.x() << "\n y=" << quat.y() << "\n z=" << quat.z() << "\n w=" << quat.w() << std::endl;
  return 0;
}

int main(int argc, char* argv[]) {
  testCase2_chat3();

  return 0;
}