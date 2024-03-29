#include <algorithm>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cfloat>

#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkKdTreePointLocator.h>
#include <vtkPointSource.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkNew.h>

void testCase1_kdtree() {
  int N = 50000;
  vtkNew<vtkPointSource> pointSource;
  pointSource->SetNumberOfPoints(N);
  pointSource->Update();

  vtkPoints *randPts = pointSource->GetOutput()->GetPoints();
  for (vtkIdType i = 0; i < N; i++) {
    double pts[3];
    randPts->GetPoint(i, pts);
    std::cout << pts[0] << "," << pts[1] << "," << pts[2] << std::endl;
  }

  auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
  
  // create the tree
  vtkNew<vtkKdTreePointLocator> pointTree;
  pointTree->SetDataSet(pointSource->GetOutput());
  pointTree->BuildLocator();

  unsigned int k = 1;
  vtkNew<vtkPointSource> testSource;
  testSource->SetNumberOfPoints(1);
  testSource->Update();
  double testPoint[3];
  testSource->GetOutput()->GetPoints()->GetPoint(0, testPoint);
  vtkNew<vtkIdList> result;
  std::cout << "Test Point: " << testPoint[0] << "," << testPoint[1] << "," << testPoint[2] << std::endl;

  pointTree->FindClosestNPoints(k, testPoint, result.GetPointer());

  for (vtkIdType i = 0; i < k; i++) {
    vtkIdType point_ind = result->GetId(i);
    double p[3];
    pointSource->GetOutput()->GetPoint(point_ind, p);
    std::cout << "Closet point " << i << ": Point" << point_ind << ": ("
              << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

  std::cout << "VTK Time:" << t2 - t1 << " ms" << std::endl;
  return ;
}

int main() {
  std::cout << "======== test vtk ===========" << std::endl;

  testCase1_kdtree();
  return 0;
}