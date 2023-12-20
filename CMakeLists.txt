cmake_minimum_required(VERSION 3.0)
project(Test3D)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")

find_package(Open3D REQUIRED)
# Open3D
find_package(Open3D HINTS REQUIRED)
list(APPEND Open3D_LIBRARIES dl)
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")
    link_directories(${Open3D_LIBRARY_DIRS})
endif()
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
# find_package(Ceres REQUIRED)
# list(APPEND EXTRA_NEED_LIBS Ceres::ceres)

option(OpenCV_STATIC "Build static opencv libraries" ON)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found while DepthProc need it")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV2_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

list(APPEND EXTRA_NEED_LIBS ${OpenCV_LIBS})

find_package(PCL 1.7)
if (NOT PCL_FOUND)
    message(FATAL_ERROR "PCL library not found, some tests will not be compiled")
endif()

add_definitions(-DHAVE_PCL)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# list(APPEND EXTRA_NEED_LIBS ${PCL_LIBRARIES})

add_executable(open3d_test src/testOpen3D.cpp)
target_link_libraries(open3d_test ${Open3D_LIBRARIES} ${EXTRA_NEED_LIBS})

# vtk
set(VTK_DIR "path/to/VTK-build")
find_package(VTK REQUIRED)
message("VTK VERSION:${VTK_VERSION}")
message("VTK_LIBRARIES:${VTK_LIBRARIES}")
message("VTK_INCLUDE_DIRS:${VTK_INCLUDE_DIRS}")
message("VTK_USE_FILE:${VTK_USE_FILE}")
include(${VTK_USE_FILE})

add_executable(vtk_test src/testVTK.cpp)
target_link_libraries(vtk_test ${VTK_LIBRARIES} ${EXTRA_NEED_LIBS})


# eigen

# add_executable(eigen_test src/testEigen.cpp)
# target_link_libraries(eigen_test ${VTK_LIBRARIES} ${EXTRA_NEED_LIBS})


# include_directories(${Open3D_INCLUDE_DIRS})
# link_directories(${Open3D_LIBRARY_DIRS})
# add_executable(t3DDemo src/testOpen3D.cpp)
# target_link_libraries(t3DDemo ${Open3D_LIBRARIES})
# target_include_directories(t3DDemo PUBLIC ${Open3D_INCLUDE_DIRS})

# file(GLOB_RECURSE srcs src/*.cpp)
# add_library(${PROJECT_NAME} STATIC ${srcs})
# message(STATUS "srcs = ${srcs}")
# add_executable(t3DDemo ${srcs})
# target_link_libraries(t3DDemo ${Open3D_LIBRARIES})
# target_include_directories(t3DDemo PUBLIC ${Open3D_INCLUDE_DIRS})
# include_directories("usr/include/eigen3")

