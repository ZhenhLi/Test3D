#include <opencv2/opencv.hpp>
#include <iostream>

namespace cvfilter {

// general image filter
int meanFilter(cv::Mat src, cv::Mat* dst, cv::Size kSize = cv::Size(5, 5));
int medianFilter(cv::Mat src, cv::Mat *dst, int kSize = 5);
int GaussianFilter(cv::Mat src, cv::Mat *dst, cv::Size kSize, double sigmaX);

// outlier filter
int spackleFilter(cv::Mat src, cv::Mat dst, int param);
int threshold(cv::Mat src, cv::Mat dst, int param);
int gradientFilter(cv::Mat src, cv::Mat dst, int param);
int rangeFilter(cv::Mat src, cv::Mat dst, int param);

// depth filling
int fillHoleByLeft(cv::Mat src, cv::Mat dst);
int fillHoleByHorizontalInterpolation(cv::Mat src, cv::Mat dst, int param);
int fillHoleByVerticalInterpolation(cv::Mat src, cv::Mat dst, int param);
int fillHoleByHorizontalAndVerticalInterpolation(cv::Mat src, cv::Mat dst, int param);
int fillHoleByInpaint(cv::Mat src, cv::Mat dst, int param);
















int maxValueFilter(cv::Mat src, cv::Mat *dst, int kSize);
int minValueFilter(cv::Mat src, cv::Mat *dst, int kSize);
int boxFIlter(cv::Mat src, cv::Mat *dst, int kSize);
int Conv(cv::Mat src, cv::Mat *dst, cv::Mat kernel);
int bilateralFilter(cv::Mat src, cv::Mat *dst, /*param*/int param);
int nonLocalMeans(cv::Mat src, cv::Mat *dst, int param);
int WinearFilter(cv::Mat src, cv::Mat *dst, int param);
int GuidanceFilter(cv::Mat src, cv::Mat *dst, int param);
int BM3DFilter(cv::Mat src, cv::Mat *dst, int param);



int fillHolesByNearestInterpolate(cv::Mat src, cv::Mat *dst, int param);
int fillHolesByLeftValue(cv::Mat src, cv::Mat *dst, int param);
int fillHolesByJointBilateralFilterInpainting(cv::Mat src, cv::Mat *dst, int param);
int fillHolesByIPBasic(cv::Mat src, cv::Mat *dst, int param); // ref: In defense of classical image processing: fast depth completion on the CPU.
int fillHolesByLambertianModel(cv::Mat src, cv::Mat *dst, int param); // ref: Lambertian Model-based Normal Guided Depth COmpletion for LiDAR-Camera System. 2021.
// Depth Completion Evaluation https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

// testcase


}