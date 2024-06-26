#include "filter.h"

namespace cvfilter {

int meanFilter(cv::Mat src, cv::Mat* dst, cv::Size kSize) {
  assert(!src.empty());
  assert(dst);
  cv::blur(src, *dst, kSize);
  return 0;
}

int medianFilter(cv::Mat src, cv::Mat *dst, int kSize) {
  assert(!src.empty());
  assert(dst);
  cv::medianBlur(src, *dst, kSize);
  return 0;
}

int GaussianFilter(cv::Mat src, cv::Mat *dst, cv::Size kSize, double sigmaX) {
  assert(!src.empty());
  assert(dst);
  cv::GaussianBlur(src, *dst, kSize, sigmaX);
  return 0;
}

// maxValueFilter
cv::Mat fun1(int i, int j, int kSize, cv::Mat src) {
  int x1, x2, y1, y2;
  x1 = - kSize / 2;
  y1 = - kSize / 2;
  x2 = kSize + 1;
  y2 = kSize + 1;
  cv::Mat temp = cv::Mat::zeros(kSize, kSize, CV_8UC1);
  int count = 0;
  for (int m = x1; m < x2; m++) {
    for (int n = y1; n < y2; n++) {
      if ((i + m < 0) || (i + m > src.rows) || (j + n < 0) || (j + n > src.cols)) {
        temp.ptr<uchar>(m - x1)[n - y1] = src.ptr<uchar>(i)[j];
      } else { 
        temp.ptr<uchar>(m - x1)[n - y1] = src.ptr<uchar>(i+m)[j+n];
      }
      count += 1;
    }
  }
  return temp;
}
int maxValueFilter(cv::Mat src, cv::Mat *dst, int kSize) {
  assert(!src.empty());
  assert(dst);
  
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cv::Mat temp = fun1(i, j, kSize, src);
    //   if ()
    }
  }
  return 0;
}

int boxFilter(cv::Mat src, cv::Mat *dst, cv::Size kSize) {
  assert(!src.empty());
  assert(dst);
  cv::boxFilter(src, *dst, -1, kSize);
  return 0;
}

int bilateralFilter(cv::Mat src, cv::Mat *dst, int d, double sigmaColor, double sigmaSpace) {
  assert(!src.empty());
  assert(dst);
  cv::bilateralFilter(src, *dst, d, sigmaColor, sigmaSpace);
  return 0;
}

int NLMeansFilter(cv::Mat src, cv::Mat* dst, float h, int tempalteWindowSize, int searchWidowSize, float hColor = 3) {
  assert(!src.empty());
  assert(dst);
  if (src.type() == CV_8UC1) {
    cv::fastNlMeansDenoising(src, *dst, h, tempalteWindowSize, searchWidowSize);
  } else if (src.type() == CV_8UC3) {
    cv::fastNlMeansDenoisingColored(src, *dst, h, hColor, tempalteWindowSize, searchWidowSize);
  } else if (src.type() == CV_16SC1) {
    // depth image filter
  }
  return 0;
}


int nonLocalMeans(cv::Mat src, cv::Mat *dst, int param) {
  // 
//   cv::fastNlMeansDenoising(), 使用单个灰度图像
//   cv::fastNlMeansDenoisingColored(), 使用彩色图像
//   cv::fastNlMeansDenoisingMulti(), 用于在短时间内捕获的图像序列（灰度图像）
//  cv::fastNlMeansDenoisingColoredMulti
  cv::Mat mat;
  cv::imread("/home/lzh/Pictures/dataset2D/DIP/DIP3E_CH05_Original_Images/Fig0507(b)(ckt-board-gauss-var-400).tif", -1);

  return 0;
}

int JointBilateralUpsample() {
  // JBU, 联合双边上采样
  // https://blog.csdn.net/zcg1942/article/details/108448764


  return 0;
}

// 自适应中值滤波器
// https://zhuanlan.zhihu.com/p/279602383
uchar adaptiveProcess(const Mat& src, int row, int col, int kernelSize, int maxSize) {
  std::vector<uchar> pixels;
  
  for (int i = -kernelSize / 2; i <= kernelSize / 2; i++) {
    for (int j = -kernelSize / 2; j <= kernelSize / 2; j++) {
      pixels.push_back(src.at<uchar>(row + i, col + b));
    }
  }
  std::sort(pixels.begin(), pixels.end());
  auto min = pixels[0];
  auto max = pixels[kernelSize * kernelSize - 1];
  auto med = pixels[kernelSize * kernelSize / 2];
  auto zxy = src.at<uchar>(row, col);
  if (med > min && med < max) {
    if (zxy > min && zxy < max) {
      return zxy;
    } else {
      return med;
    }
  } else {
    kernelSize += 2;
    if (kernelSize <= maxSize) {
      return (adaptiveProcess(src, row, col, kernelSize, maxSize)); // 增大窗口尺寸，继续自适应过程
    } else {
      return med;
    }
  }
}

void GaussianFilter(const Mat& src, Mat* dst, int kSize, douibel sigma) {
  CV_ASSERT(src.channgles() || src.channels() == 3);
  double **GrassianTemplate = new double *[ksize];
  for (int i = 0; i < kSize; i++) {
    GrassianTemplate[i] = new doiuble [kSize];
  }
  // cv::generate
}

}