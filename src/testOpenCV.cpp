#include <iostream>
#include <opencv2/opencv.hpp>

int testCase1_showMat() {
  cv::Mat a = cv::Mat::ones(cv::Size(100, 100), CV_8UC3);
  cv::namedWindow("a", CV_WINDOW_NORMAL);
  cv::imshow("a", a);
  cv::waitKey(0);

  return 0;
}

int testCase2_Corners() {
  /**
   * @brief testCase for opencv corners detection
   * @ref https://zhuanlan.zhihu.com/p/623424942
  */

  cv::Mat src1 = cv::imread("/home/lzh/Pictures/dataset2D/laser1.bmp", 1);
  cv::resize(src1, src1, cv::Size(300, 200));
  cv::Mat srcGray1;
  cv::cvtColor(src1, srcGray1, cv::COLOR_BGR2GRAY);
  cv::Mat dst1;
  // Harris corners
  cv::cornerHarris(srcGray1, dst1, 7, 13, 0.04); // src, dst, blockSize角点检测中考虑的邻域大小, 
                                             // ksize Sobel求导中使用的窗口大小
                                             // k - Harris角点检测方程中的自由参数, 取值参数为[0.04, 0.06]
  cv::namedWindow("dst1", CV_WINDOW_NORMAL);
  cv::imshow("dst1", dst1);
  cv::waitKey(0);

  // Shi-Tomasi
  cv::Mat dst2 = src1.clone();
  std::vector<cv::Point2f> corners;
  int kMaxCorners = 1000;
  double kQualityLevel = 0.1;
  double kMinDistance = 1;
  cv::goodFeaturesToTrack(srcGray1, corners, kMaxCorners, kQualityLevel, kMinDistance);
  for (size_t i = 0; i < corners.size(); i++) {
    cv::circle(dst2, corners[i], 2.5, cv::Scalar(0, 255, 0));
  }
  cv::namedWindow("dst2", CV_WINDOW_NORMAL);
  cv::imshow("dst2", dst2);
  cv::waitKey(0);

  // realise corner detection
    // step: sobel dx dy; -> Matrix M -> boxFilter -> every corner response R


  // cornerSubpix()
  // void cornerSubpix(InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria cirteria);

  return 0;
}

int testCase3_svd() {
  // opencv求SVD
  cv::Mat src = cv::imread("/home/lzh/Pictures/dataset2D/laser1.bmp", cv::IMREAD_GRAYSCALE);
  cv::resize(src, src, cv::Size(800, 600));
  cv::Mat src_ = src.clone();
  src.convertTo(src, CV_64FC1);
  cv::Mat U, W, V;
  cv::SVD::compute(src, W, U, V);
  std::cout << "W" << W.rows << ", " << W.cols << std::endl;
  std::cout << "U" << U.rows << ", " << U.cols << std::endl;
  std::cout << "V" << V.rows << ", " << V.cols << std::endl;
  std::cout << "W value >> " << W << std::endl;

  int set_dim = std::min(src.rows, src.cols);
  cv::Mat W_ = cv::Mat(set_dim, set_dim, CV_64FC1, cv::Scalar(0));
  double ratio = 0.01;
  int set_rows = set_dim * ratio;
  for (int i = 0; i < set_rows; i++) {
    W_.at<double>(i, i) = W.at<double>(i, 0);
  }
  cv::Mat dst = U * W_ * V;
  cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
  cv::imshow("dst", dst);
  cv::waitKey(0);
  return 0;
}

// for testcase3 supplement
int testCase3_svd_case2() {
  float b[] ={1, 2, 3, 4, 5, 6, 7, 8, 9};
  cv::Mat a = cv::Mat(3, 3, CV_32FC1, b);
  cv::Mat w, u, vt;
  cv::SVDecomp(a, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  std::cout << w << std::endl;
  std::cout << u << std::endl;
  std::cout << vt << std::endl;
  cv::Mat c = vt.row(2).reshape(1, 3);
  std::cout << c << std::endl;

  return 0;
}

int main() {
  std::cout << "------- opencv test ------------" << std::endl;
//   testCase1_showMat(); // ok
  // testCase2_Corners(); // 提取角点
  testCase3_svd();
  testCase3_svd_case2();
  return 0;
}