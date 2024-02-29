cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

# set(CMAKE_CXX_STANDARD 17) #设置为C++11，可能报错
set(Torch_DIR /home/lzh/code/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)
add_executable(example-app example.cpp)
target_link_libraries(example-app ${OpenCV_LIBS} ${TORCH_LIBRARIES} )
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
