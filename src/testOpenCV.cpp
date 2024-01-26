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

int main() {
  std::cout << "------- opencv test ------------" << std::endl;
//   testCase1_showMat(); // ok
  testCase2_Corners();
  return 0;
}