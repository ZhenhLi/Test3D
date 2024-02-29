#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int testCase_Tensor() {
  // libtorch Tensor张量的常用操作总结1  https://blog.csdn.net/shandianfengfan/article/details/118348082
  torch::Tensor a = torch::zeros({3, 5});
  // 1 打印张量信息
  // 打印张量的维度信息
  std::cout << a.sizes() << std::endl; // [3, 5]
  a.print(); // [CPUFloatType [3, 5]]

  // 打印张量的内容
  std::cout << a << std::endl; // 0  0  0  0  0
                               // 0  0  0  0  0
                               // 0  0  0  0  0

  // 2 定义并初始化张量的值
  torch::Tensor b = torch::zeros({5, 7});
  std::cout << b << std::endl;
  torch::Tensor c = torch::ones({3, 4});
  std::cout << c << std::endl;
  torch::Tensor d = torch::eye(5); // 5*5单位张量
  std::cout << d << std::endl;
  torch::Tensor e = torch::full({3, 4}, 10);
  std::cout << e << std::endl;
  // 以另一个张量的形状作为模板
  auto f = torch::full_like(b, 2); // 形状类似于b，填充2
  std::cout << f << std::endl;
  // n行1列，并指定初始化值
  auto g = torch::tensor({1, 2, 3, 4, 5});
  std::cout << g << std::endl;
  // 定义一维张量，并使用随机数初始化
  auto h = torch::randn({3, 4}); // randn 标准正态分布；rand 区间[0，1]的符合均匀分布的随机数初始化；
  std::cout << h << std::endl;
  // 形状 5*5，区间[0, 10]的整型数初始化
  auto j = torch::randint(0, 10, {5, 5});
  std::cout << j << std::endl;
  // 使用数组或某一段内存初始化张量
  int aa[4] = {3, 4, 6, 7};
  auto aaa = torch::from_blob(aa, {2, 2}, torch::kInt);
  std::cout << aaa << std::endl;
  std::vector<float> bb = {3, 4, 6};
  auto bbb = torch::from_blob(bb.data(), {1, 1, 1, 3}, torch::kFloat);
  std::cout << bbb << std::endl;
  // 使用Mat中的数据
  cv::Mat mat = cv::Mat::zeros(5, 5, CV_32FC1);
  auto xx = torch::from_blob(mat.data, {1, 1, 5, 5}, torch::kFloat);
  std::cout << xx << std::endl;
  // 注意三通道cv::Mat不能直接赋指针
  cv::Mat x1 = cv::Mat::ones(5, 5, CV_8UC1) * 1;
  cv::Mat x2 = cv::Mat::ones(5, 5, CV_8UC1) * 2;
  cv::Mat x3 = cv::Mat::ones(5, 5, CV_8UC1) * 3;
  cv::Mat x123 = cv::Mat::zeros(5, 5, CV_8UC3);
  std::vector<cv::Mat> channels;
  channels.push_back(x1);
  channels.push_back(x2);
  channels.push_back(x3);
  cv::merge(channels, x123);
  std::cout << "x123:" << x123 << std::endl;
  auto x_t = torch::from_blob(x123.data, {5, 5, 3}, torch::kByte);
  std::cout << "x_t" << x_t << std::endl;
  x_t = x_t.permute({2, 0, 1});
  std::cout << x_t << std::endl;

  // 3 张量拼接
  torch::Tensor a1 = torch::rand({2, 3});
  torch::Tensor a2 = torch::rand({2, 1});
  torch::Tensor cat1 = torch::cat({a1, a2}, 1); // 按列拼接
  std::cout << a1 << std::endl;
  std::cout << a2 << std::endl;
  std::cout << cat1 << std::endl;
  torch::Tensor b1 = torch::rand({2, 3});
  torch::Tensor b2 = torch::rand({1, 3});
  torch::Tensor cat2 = torch::cat({b1, b2}, 0); // 按列拼接
  std::cout << b1 << std::endl;
  std::cout << b2 << std::endl;
  std::cout << cat2 << std::endl;
  
  // 4 切片与索引
  auto c1 = torch::linspace(1, 75, 75).reshape({3, 5, 5});
  std::cout << c1 << std::endl;
  auto bx = c1.index({"...", 2}); // 对第1 2维度的所有索引，对第三维度索引号范围i-j的数据.
  bx.print();
  std::cout << bx << std::endl;
  auto bx2 = c1.index({"...", 2, 3});
  std::cout << bx2 << std::endl;
  auto bx3 = c1.index({2, "..."});
  std::cout << bx3 << std::endl;
  // auto bx41 << c1.index({2, "...", 2}); // error
  // std::cout << bx41 << std::endl;
  // 通过索引赋值
  auto dx = torch::linspace(1, 4, 4).reshape({2, 2});
  std::cout << dx << std::endl;
  dx.index_put_({1, 1}, 100); // 将第1维度的索引号1、第二维度的索引号1处赋100
  std::cout << dx << std::endl;
  // 切片
  using namespace torch::indexing;
  auto dx2 = c1.index({"...", Slice(1)}); // ?
  // std::cout << dx2 << std::endl;
  float a41[2][3][3] = {{{1, 2, 2}, {3, 4, 4}, {5, 6, 6}},{{1, 2, 2}, {3, 4, 4}, {5, 6, 6}}};
  at::Tensor b41 = at::from_blob(a41, {2, 3, 3}, at::kFloat);
  at::Tensor c41 = b41.index({"...", Slice({None, 2})});
  auto d41 = c41.sizes();
  std::cout << b41 << std::endl;
  std::cout << c41 << std::endl;
  std::cout << d41 << std::endl;

  return 0;
}

int testCase3_() {
  std::cout << "---------- linear ----------" << std::endl;
  torch::Tensor x = torch::randn({2, 3});
  torch::nn::Linear f(3, 6);
  torch::Tensor y = f(x);
  std::cout << y << std::endl;
  torch::Tensor y1 = f->forward(x);
  std::cout << y1 << std::endl;
  return 0;
}
 
int main() {
  torch::Tensor tensor = torch::rand({2, 3}); //生成大小为2*3的随机数矩阵
  std::cout << tensor << std::endl;           //标准输出流打印至屏幕\

  testCase_Tensor();
  testCase3_();
}
