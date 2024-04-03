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

int testCase4_calib() {
  // 1 准备标定棋盘格图像
  int boardWidth = 7; // 棋盘格横向内角点数量
  int boardHeight = 7; // 棋盘格纵向内角点数量
  float squareSize = 1.f; // 棋盘格格子大小，单位为米，随便设置，不影响相机内参计算
  cv::Size boardSize(boardWidth, boardHeight);

  std::vector<std::vector<cv::Point3f>> objectPoints;
  std::vector<std::vector<cv::Point2f>> imagePoints;
  std::vector<cv::Point2f> corners;

  // 2 拍摄棋盘图像
  cv::Mat image, gray;
  cv::namedWindow("image", cv::WINDOW_NORMAL);
  std::vector<cv::String> fileNames;
  cv::glob("/home/lzh/Pictures/dataset2D/chessboard/801_93708/data/images/*.png", fileNames);

  for  (size_t i = 0; i < fileNames.size(); i++) {
    image = cv::imread(fileNames[i], cv::IMREAD_COLOR);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 3 读入图像数据，并提取角点
    bool found = cv::findChessboardCorners(image, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH 
                                                                      + cv::CALIB_CB_NORMALIZE_IMAGE 
                                                                      + cv::CALIB_CB_FAST_CHECK);
    if (found) {
      cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS 
                                                                                            + cv::TermCriteria::COUNT, 30, 0.1));
      cv::drawChessboardCorners(image, boardSize, corners, found);
      cv::imshow("image", image);
      cv::waitKey();

      std::vector<cv::Point3f> objectCorners;
      for (int j = 0; j < boardHeight; j++) {
        for (int k = 0; k < boardWidth; k++) {
          objectCorners.push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
        }
      }
      objectPoints.push_back(objectCorners);
      imagePoints.push_back(corners);
    }
  }

  // 4 标定相机
  cv::Mat cameraMatrix, distCoeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  cv::calibrateCamera(objectPoints, imagePoints, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

  std::cout << "Camera matrix:" << std::endl << cameraMatrix << std::endl;
  std::cout << "Distortion coefficients:" << std::endl << distCoeffs << std::endl;

  return 0;
}

int testCase5_IPOneImage() {


  cv::Mat mat = cv::imread("/home/lzh/Pictures/dataset2D/Lena.bmp", -1);
  cv::namedWindow("image", cv::WINDOW_NORMAL);
  cv::imshow("image", mat);
  cv::waitKey(0);


  // 2 空间变换 
  // 2.1 按某点旋转 https://blog.csdn.net/qq_27278957/article/details/88865777
  double angle = 0;
  int len = std::max(mat.cols, mat.rows);
  // cv::Point2f center(len / 2., len / 2.);
  cv::Point2f center(0., 0);
  cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0); // 绕center旋转角度为angle的矩阵
  cv::Size srcSz = mat.size();
  cv::Mat dst;
  cv::warpAffine(mat, dst, rotMat, srcSz); 

  // cv::namedWindow("dst", cv::WINDOW_NORMAL);
  // cv::imshow("dst", dst);
  // cv::waitKey(0);

  // 2.2 平移 https://blog.csdn.net/weixin_38346042/article/details/122595084
  float tx = float(mat.rows) / 4;
  float ty = float(mat.cols) / 4;
  float warpValues[] = {1.0, 0.0, tx, 0.0, 1.0, ty};
  cv::Mat translationMatrix = cv::Mat(2, 3, CV_32F, warpValues);
  cv::Mat translatedImage;
  cv::warpAffine(mat, translatedImage, translationMatrix, mat.size());

  // cv::namedWindow("translatedImage", cv::WINDOW_NORMAL);
  // cv::imshow("translatedImage", translatedImage);
  // cv::waitKey(0);

  // 2.3 尺幅缩放
  cv::Mat resizeImage;
  cv::resize(mat, resizeImage, cv::Size(mat.rows / 2, mat.cols / 2), 0, 0);

  // cv::namedWindow("resizeImage", cv::WINDOW_NORMAL);
  // cv::imshow("resizeImage", resizeImage);
  // cv::waitKey(0);

  // 2.4 仿射变换 https://blog.csdn.net/qxqsunshine/article/details/115110119
  cv::Mat wrapMat;
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  cv::Mat rotMat2(2, 3, CV_32FC1);
  cv::Mat warpMat2(2, 3, CV_32FC1);
  cv::Mat warpDst, warpRotateDst;
  warpDst = cv::Mat::zeros(mat.rows, mat.cols, mat.type());
  srcTri[0] = cv::Point2f(0, 0);
  srcTri[1] = cv::Point2f(mat.cols - 1, 0);
  srcTri[2] = cv::Point2f(0, mat.rows - 1);

  dstTri[0] = cv::Point2f(mat.cols * 0.0, mat.rows * 0.33);
  dstTri[1] = cv::Point2f(mat.cols * 0.85, mat.rows * 0.25);
  dstTri[2] = cv::Point2f(mat.cols * 0.15, mat.rows * 0.7);

  warpMat2 = cv::getAffineTransform(srcTri, dstTri);
  cv::getAffineTransform(srcTri, dstTri);
  cv::warpAffine(mat, warpDst, warpMat2, warpDst.size());

  // cv::namedWindow("warpDst", cv::WINDOW_NORMAL);
  // cv::imshow("warpDst", warpDst);
  // cv::waitKey(0);

  // 2.5 透视变换 https://zhuanlan.zhihu.com/p/387408410
  cv::Mat src = cv::imread("/home/lzh/Pictures/chedaoxian.png", -1);

  cv::Mat perspectiveMat;
  cv::Point2f srcAffinePts[4];
  cv::Point2f dstAffinePts[4];
  srcAffinePts[0].x = 20 * 1020 / 230;  srcAffinePts[0].y = 95 * 647 / 145;
  srcAffinePts[1].x = 210 * 1020 / 230; srcAffinePts[1].y = 95 * 647 / 145;
  srcAffinePts[2].x = 90 * 1020 / 230;  srcAffinePts[2].y = 65 * 647 / 145;
  srcAffinePts[3].x = 140 * 1020 / 230; srcAffinePts[3].y = 65 * 647 / 145;
  // 透视后坐标
  dstAffinePts[0].x = 50;  dstAffinePts[0].y = 780;
  dstAffinePts[1].x = 490; dstAffinePts[1].y = 780;
  dstAffinePts[2].x = 50;  dstAffinePts[2].y = 150;
  dstAffinePts[3].x = 490; dstAffinePts[3].y = 150;
  perspectiveMat = cv::getPerspectiveTransform(srcAffinePts, dstAffinePts);
  cv::Mat perspectiveImage;
  cv::warpPerspective(src, perspectiveImage, perspectiveMat, cv::Size(src.rows, src.cols));
  cv::circle(perspectiveImage, dstAffinePts[0], 9, cv::Scalar(0, 0, 255), 3);
  cv::circle(perspectiveImage, dstAffinePts[1], 9, cv::Scalar(0, 0, 255), 3);
  cv::circle(perspectiveImage, dstAffinePts[2], 9, cv::Scalar(0, 0, 255), 3);
  cv::circle(perspectiveImage, dstAffinePts[3], 9, cv::Scalar(0, 0, 255), 3);
  // cv::namedWindow("perspectiveImage", cv::WINDOW_NORMAL);
  // cv::imshow("perspectiveImage", perspectiveImage);
  // cv::waitKey(0);

  // 3 卷积滤波
  // 3.1 卷积
  cv::Mat mat3;
  mat3 = cv::imread("/home/lzh/Pictures/dataset2D/DIP/DIP3E_CH01_Original_Images/DIP3E_Original_Images_CH01/Fig0107(e)(cygnusloop-Xray).tif", -1);
  cv::namedWindow("mat3", cv::WINDOW_NORMAL);
  cv::imshow("mat3", mat3);
  cv::waitKey(0);

  cv::Mat dst3 = cv::Mat::zeros(mat3.size(), CV_8UC1);

  for (int i = 0; i < mat3.rows - 1; i++) {
    const uchar* previous = mat3.ptr<uchar>(i - 1);
    const uchar* current = mat3.ptr<uchar>(i);
    const uchar* next = mat3.ptr<uchar>(i + 1);
    uchar* output = dst3.ptr<uchar>(i);
    for ( int j = 0; j < mat3.cols - 1; j++) {
      // *output++ = std::saturate_cast<uchar>(5 * current[j] - current[j - 1] - current[j + 1] - previous[j] - next[j]);
      output[j] = 5 * current[j] - current[j - 1] - current[j + 1] - previous[j] - next[j];
    }
  }
  // 对边缘行处理
  dst3.row(0).setTo(cv::Scalar(0));
  dst3.row(dst3.rows - 1).setTo(cv::Scalar(0));
  dst3.col(0).setTo(cv::Scalar(0));
  dst3.col(dst3.cols - 1).setTo(cv::Scalar(0));
  cv::namedWindow("dst3", cv::WINDOW_NORMAL);
  cv::imshow("dst3", dst3);
  cv::waitKey(0);

  // 3.1.2 filter2D
  cv::Mat dst3_1_2 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::Mat kernel3_1 = (cv::Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::filter2D(mat3, dst3_1_2, CV_8UC1, kernel3_1);
  cv::namedWindow("dst3_1_2", cv::WINDOW_NORMAL);
  cv::imshow("dst3_1_2", dst3_1_2);
  cv::waitKey(0);

  // 3.2 均值滤波
  cv::Mat dst3_2 = cv::Mat::zeros(mat3.size(), CV_8UC1);

  for (int i = 0; i < mat3.rows - 1; i++) {
    const uchar* previous = mat3.ptr<uchar>(i - 1);
    const uchar* current = mat3.ptr<uchar>(i);
    const uchar* next = mat3.ptr<uchar>(i + 1);
    uchar* output = dst3_2.ptr<uchar>(i);
    for ( int j = 0; j < mat3.cols - 1; j++) {
      // *output++ = std::saturate_cast<uchar>(5 * current[j] - current[j - 1] - current[j + 1] - previous[j] - next[j]);
      output[j] = (current[j] + current[j - 1] + current[j + 1] + previous[j - 1] + previous[j] + previous[j + 1] + next[j - 1] + next[j] + next[j + 1])/9;
    }
  }
  // 对边缘行处理
  dst3_2.row(0).setTo(cv::Scalar(0));
  dst3_2.row(dst3_2.rows - 1).setTo(cv::Scalar(0));
  dst3_2.col(0).setTo(cv::Scalar(0));
  dst3_2.col(dst3_2.cols - 1).setTo(cv::Scalar(0));
  cv::namedWindow("dst3_2", cv::WINDOW_NORMAL);
  cv::imshow("dst3_2", dst3_2);
  cv::waitKey(0);

  // 3.2.2 blur
  cv::Mat dst3_2_2 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::blur(mat3, dst3_2_2, cv::Size(9, 9), cv::Point(-1, -1)); // size 9, 9
  cv::namedWindow("dst3_2_2", cv::WINDOW_NORMAL);
  cv::imshow("dst3_2_2", dst3_2_2);
  cv::waitKey(0);

  // 3.3 median
  cv::Mat dst3_3 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::medianBlur(mat3, dst3_3, 9);
  cv::namedWindow("dst3_3", cv::WINDOW_NORMAL);
  cv::imshow("dst3_3", dst3_3);
  cv::waitKey(0);

  // 3.4 高斯滤波
  cv::Mat dst3_4 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::GaussianBlur(mat3, dst3_4, cv::Size(9, 9), 0);
  cv::namedWindow("dst3_4", cv::WINDOW_NORMAL);
  cv::imshow("dst3_4", dst3_4);
  cv::waitKey(0);

  // 3.5 方框滤波
  cv::Mat dst3_5 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::GaussianBlur(mat3, dst3_5, cv::Size(9, 9), 0);
  // cv::sepFilter2D();
  cv::boxFilter(mat3, dst3_5, -1, cv::Size(5, 5));
  cv::namedWindow("dst3_5", cv::WINDOW_NORMAL);
  cv::imshow("dst3_5", dst3_5);
  cv::waitKey(0);

  // 3.6 双边滤波
  cv::Mat dst3_6 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::bilateralFilter(mat3, dst3_6, 9, 75, 75);
  cv::namedWindow("dst3_6", cv::WINDOW_NORMAL);
  cv::imshow("dst3_6", dst3_6);
  cv::waitKey(0);

  // 3.7 NLMeans
  cv::Mat dst3_7 = cv::Mat::zeros(mat3.size(), CV_8UC1);
  cv::fastNlMeansDenoising(mat3, dst3_7, 3, 7, 21);
  cv::namedWindow("dst3_7", cv::WINDOW_NORMAL);
  cv::imshow("dst3_7", dst3_7);
  cv::waitKey(0);

  // 3.8 WinearFilter

  return 0;
}

int testCase6_fillHole() {
  // 1 flood fill
  cv::Mat src, dst;
  src = cv::imread("/home/lzh/Pictures/dataset2D/DIP/DIP3E_CH01_Original_Images/DIP3E_Original_Images_CH01/Fig0107(e)(cygnusloop-Xray).tif", -1);
  cv::namedWindow("src", cv::WINDOW_NORMAL);
  cv::imshow("src", src);
  cv::waitKey(0);
  cv::Size m_Size = src.size();
  cv::Mat Temp = cv::Mat::zeros(m_Size.height + 2, m_Size.width + 2, src.type());
  src.copyTo(Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));
  cv::floodFill(Temp, cv::Point(0, 0), cv::Scalar(255));

  cv::Mat cutImg;
  Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cutImg);
  dst = src | (~cutImg);
  cv::namedWindow("dst", cv::WINDOW_NORMAL);
  cv::imshow("dst", dst);
  cv::waitKey(0);

  // 2 left value

  // 3 horizontical interpolation.

  // 4 inpainting
  cv::Mat dst4;
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      if (src.ptr<uchar>(i)[j] == 0) {
        mask.ptr<uchar>(i)[j] = 255;
      }
    }
  }
  cv::inpaint(src, mask, dst, 5, cv::INPAINT_NS);
  cv::namedWindow("dst", cv::WINDOW_NORMAL);
  cv::imshow("dst", dst);
  cv::waitKey(0);
  std::cout << "run fillholes done" << std::endl;
  return 0;
}

int testCase7_PCA() {
  // https://github.com/opencv/opencv/blob/4.x/modules/core/test/test_mat.cpp
  std::cout << "PCA test" << std::endl;

  const cv::Size sz(200, 500);
  double diffPrjEps, diffBackPrjEps, prjEps, backPrjEps, evalEps, evecEps;
  int maxComponents = 100;
  double retainedVariance = 0.95;
  cv::Mat rPoints(sz, CV_32FC1), rTestPoints(sz, CV_32FC1);
  cv::RNG rng(12345);

  rng.fill(rPoints, cv::RNG::UNIFORM, cv::Scalar::all(0.0), cv::Scalar::all(1.0));
  rng.fill(rTestPoints, cv::RNG::UNIFORM, cv::Scalar::all(0.0), cv::Scalar::all(1.0));

  cv::PCA rPCA(rPoints, cv::Mat(), CV_PCA_DATA_AS_ROW, maxComponents), cPCA;

  // 1. check C++ PCA & ROW
  cv::Mat rPrjTestPoints = rPCA.project(rTestPoints);
  cv::Mat rBackPrjTestPoints = rPCA.backProject(rPrjTestPoints);

  cv::Mat avg(1, sz.width, CV_32FC1);
  cv::reduce(rPoints, avg, 0, cv::REDUCE_AVG);
  cv::Mat Q = rPoints - cv::repeat(avg, rPoints.rows, 1);
  cv::Mat Qt = Q.t();
  cv::Mat eval, evec;
  Q = Qt * Q;
  Q = Q / (float)rPoints.rows;

  cv::eigen(Q, eval, evec);

  cv::Mat subEval(maxComponents, 1, eval.type(), eval.ptr());
  cv::Mat subEvec(maxComponents, evec.cols, evec.type(), evec.ptr());

  cv::Mat prjTestPoints, backProjTestPoints, cPoints = rPoints.t(), cTestPoints = rTestPoints.t();
  CvMat _points, _testPoints, _avg, _eval, _evec, _prjTestPoints, _backPrjTestPoints;

  double eigenEps = 1e-4;
  double err;
  for (int i = 0; i < Q.rows; i++) {
    cv::Mat v = evec.row(i).t();
    cv::Mat Qv = Q * v;
    
    cv::Mat lv = eval.at<float>(1.0) * v;
    // err = cvtest::norm(Qv, lv, cv::NORM_L2 | cv::NORM_RELATIVE);
    // std::cout << err << eigenEps << "i =" << i << std::endl;
  }

  return 0;
}

int testCase8_eigen() {
  cv::Mat m = (cv::Mat_<float>(3, 3) << 1, 2, 3, 2, 5, 6, 3, 6, 7); // 输入mat只能是CV_32FC1或CV_64FC1的方阵 
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;

  cv::eigen(m, eigenvalues, eigenvectors);
  std::cout << eigenvalues << std::endl;
  std::cout << eigenvectors << std::endl;

  // 非对称矩阵：cv::eigenNonSymmetric()。

  return 0;
}

int main() {
  std::cout << "------- opencv test ------------" << std::endl;
//   testCase1_showMat(); // ok
  // testCase2_Corners(); // 提取角点
  // testCase3_svd(); // ok
  // testCase3_svd_case2();
  // testCase4_calib(); // error
  // testCase5_IPOneImage();
  // testCase6_fillHole();
  // testCase7_PCA();
  testCase8_eigen();

  std::cout << "-------- opencv test run done -----" << std::endl;
  return 0;
}