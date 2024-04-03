cmake_minimum_required(VERSION 3.0)
project(Test3D)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_BUILD_TYPE Debug)

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

# # vtk
# set(VTK_DIR "path/to/VTK-build")
# find_package(VTK REQUIRED)
# message("VTK VERSION:${VTK_VERSION}")
# message("VTK_LIBRARIES:${VTK_LIBRARIES}")
# message("VTK_INCLUDE_DIRS:${VTK_INCLUDE_DIRS}")
# message("VTK_USE_FILE:${VTK_USE_FILE}")
# include(${VTK_USE_FILE})

# add_executable(vtk_test src/testVTK.cpp)
# target_link_libraries(vtk_test ${VTK_LIBRARIES} ${EXTRA_NEED_LIBS})


# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(eigen_test src/testEigen.cpp)
target_link_libraries(eigen_test ${EIGEN_LIBRARIES} ${EXTRA_NEED_LIBS})

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(opencv_test src/testOpenCV.cpp)
target_link_libraries(opencv_test ${OpenCV_LIBS})

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


# SLAM例程
add_executable(slam_test src/testSLAM/testSLAM_main.cpp)
target_link_libraries(slam_test ${OpenCV_LIBRARIES} ${EIGEN_LIBRARIES} ${Open3D_LIBRARIES} ${EXTRA_NEED_LIBS})


# leetcode刷题
add_executable(leetcode_test leetcode/main.cpp)
target_link_libraries(leetcode_test ${OpenCV_LIBRARIES} ${EIGEN_LIBRARIES} ${Open3D_LIBRARIES} ${EXTRA_NEED_LIBS})

# libtorch
project(test-libtorch)
# set(Torch_DIR /home/lzh/code/libtorch/share/cmake/ATen)
set(Torch_DIR /home/lzh/code/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
# add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)
set(CMKAE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
add_executable(libtorch_test src/testLibTorch.cpp)
target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_test PROPERTY CXX_STANDARD 17)


# include_directories(/home/lzh/code/libtorch/include/
#                     /home/lzh/code/libtorch/include/torch/
#                     /home/lzh/code/libtorch/include/torch/csrc/)

# link_directories(/home/lzh/code/libtorch/lib/)

# add_executable(libtorch_test src/testLibTorch.cpp)

# set(BASE_LIBRARIES 
# libasmjit.a
# libnnapi_backend.so
# libbackend_with_compiler.so 
# libnnpack.a
# libbenchmark.a
#                libnnpack_reference_layers.a
# libbenchmark_main.a
#           libonnx.a
# libc10.so
#                     libonnx_proto.a
# libcaffe2_protos.a           libprotobuf.a
# libclog.a                    libprotobuf-lite.a
# libcpuinfo.a                 libprotoc.a
# libcpuinfo_internals.a       libpthreadpool.a
# libdnnl.a                    libpytorch_qnnpack.a
# libfbgemm.a                  libqnnpack.a
# libfmt.a                     libshm.so
# libfoxi_loader.a             libtensorpipe.a
# libgloo.a                    libtensorpipe_uv.a
# libgmock.a                   libtorchbind_test.so
# libgmock_main.a              libtorch_cpu.so
# libgomp-98b21ff3.so.1        libtorch_global_deps.so
# libgtest.a                   libtorch_python.so
# libgtest_main.a              libtorch.so
# libittnotify.a               libunbox_lib.a
# libjitbackend_test.so        libXNNPACK.a
# libkineto.a
# )

# target_link_libraries(${PROJECT_NAME} ${BASE_LIBRARIES} )


