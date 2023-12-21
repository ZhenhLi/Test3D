#include <iostream>
#include <Eigen/Eigen>
#include <cfloat>

int testCase1() {

  double a;
  Eigen::Vector3i index1(11, 21, 31);
  a = index1.norm();
  std::cout << "a is " << a << std::endl;

  return 0;
}

void pointSetPCA(const std::vector<Eigen::Vector3f>& pts, 
                  Eigen::Vector3f& centroid,
                  Eigen::Vector3f& normal,
                  float& curvature) {
  assert(pts.size() > 3);
  Eigen::Map<const Eigen::Matrix3Xf>  P(&pts[0].x(), 3, pts.size());

  centroid = P.rowwise().mean();
  Eigen::MatrixXf centered = P.colwise() - centroid;
  Eigen::Matrix3f cov = centered * centered.transpose();

  // eigvecs sorted in increasing order of eigvals
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
  normal = eig.eigenvectors().col(0);
  if (normal(2) > 0)
    normal = - normal;
  Eigen::Vector3f eigVals = eig.eigenvalues();
  curvature = eigVals(0) / eigVals.sum();
  return ;
}

// ICP point to point step
Eigen::Isometry3f pointToPoint(std::vector<Eigen::Vector3f>& src, std::vector<Eigen::Vector3f>& dst) {
  int N = src.size(); 
  assert(N == dst.size());
  Eigen::Map<Eigen::Matrix3Xf> ps(&src[0].x(), 3, N);
  Eigen::Map<Eigen::Matrix3Xf> qs(&dst[0].x(), 3, N);
  Eigen::Vector3f p_dash = ps.rowwise().mean();
  Eigen::Vector3f q_dash = ps.rowwise().mean();
  Eigen::Matrix3Xf ps_centered = ps.colwise() - p_dash;
  Eigen::Matrix3Xf qs_centered = qs.colwise() - q_dash;
  Eigen::Matrix3f K = qs_centered * ps_centered.transpose();
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(K, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0)
    R.col(2) *= -1;
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.linear() = R;
  T.translation() = q_dash - R * p_dash;
  return T;
}

// ICP point to plane
Eigen::Isometry3f pointToPlane(std::vector<Eigen::Vector3f>& src, std::vector<Eigen::Vector3f>& dst, std::vector<Eigen::Vector3f>& nor) {
  assert(src.size() == dst.size() && src.size() == nor.size());
  Eigen::Matrix<float, 6, 6> C;
  C.setZero();
  Eigen::Matrix<float, 6, 1> d;
  d.setZero();
  for (uint i = 0; i < src.size(); ++i) {
    Eigen::Vector3f cro = src[i].cross(nor[i]);
    C.block<3, 3>(0, 0) += cro * cro.transpose();
    C.block<3, 3>(0, 3) += nor[i] * cro.transpose();
    C.block<3, 3>(3, 3) += nor[i] * nor[i].transpose();
    float sum = (src[i] - dst[i]).dot(nor[i]);
    d.head(3) -= cro * sum;
    d.tail(3) -= nor[i] * sum;
  }
  C.block<3, 3>(3, 0) = C.block<3, 3>(0, 3);
  Eigen::Matrix<float, 6, 1> x = C.ldlt().solve(d);
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.linear() = (  Eigen::AngleAxisf(x(0), Eigen::Vector3f::UnitX())
                * Eigen::AngleAxisf(x(1), Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(x(2), Eigen::Vector3f::UnitZ())
                ).toRotationMatrix();
  T.translation() = x.block(3, 0, 3, 1);
  return T;
}

Eigen::Isometry3f alignToOriginAndXAxis(Eigen::Vector3f p, Eigen::Vector3f n) {
  Eigen::Vector3f xAxis = Eigen::Vector3f::UnitX();
  double angle = acos(xAxis.dot(n));
  Eigen::Vector3f axis = (n.cross(xAxis)).normalized();
  // if n parallel to x axis, cross product is [0, 0, 0]
  if (n.y() == 0 && n.z() == 0)
    axis = Eigen::Vector3f::UnitY();
  Eigen::Translation3f tra(-p);
  return Eigen::Isometry3f(Eigen::AngleAxisf(angle, axis) * tra);
}

float planarRotAngle(Eigen::Vector3f p_i, Eigen::Vector3f n_i, Eigen::Vector3f p_j) {
  Eigen::Isometry3f T_ms_g = alignToOriginAndXAxis(p_i, n_i);
  Eigen::Vector3f p_j_image = T_ms_g * p_j;
  // can ignore x coordinate, since we rotate around x axis
  return atan2f(p_j_image.z(), p_j_image.y());
}

bool isPoseSimilar(Eigen::Isometry3f P_i, Eigen::Isometry3f P_j, float t_rot, float t_tra) {
  // Traslation
  float d_tra = (P_i.translation() - P_j.translation()).norm();
  // Rotation
  float d = Eigen::Quaternionf(P_i.linear()).dot(Eigen::Quaternionf(P_j.linear()));
//   float d_rot = rad2degM(std::acos(2 * d * d - 1));
  float d_rot = std::acos(2 * d * d - 1) / M_PI * 180;
  return d_rot <= t_rot && d_tra <= t_tra;
}

int testCase2_pointSetPCA() {
  // ref: "3D Object Reconstruction using Point Pair Features."
  
  return 0;
}

int main() {
  std::cout << "========== test Eigen ==========" << std::endl;

  testCase1();

  return 0;
}