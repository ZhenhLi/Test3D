cmake_minimum_required(VERSION 3.0)
project(Test3D)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Open3D REQUIRED)

include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})
# 生成Test3DDemo可执行文件
add_executable(Test3DDemo demo.cpp)
target_link_libraries(Test3DDemo ${Open3D_LIBRARIES})

target_include_directories(Test3DDemo PUBLIC ${Open3D_INCLUDE_DIRS})

# 生成 libAlgorithm.so
# 查找src目录学所有 *.cpp源文件, 并将文件列表保存到 native_srcs变量
file(GLOB_RECURSE native_srcs src/*.cpp)
# 将native_srcs中文件列表以静态库方式加入工程
add_library(${PROJECT_NAME} STATIC ${native_srcs}) # libTest3D.a
message(STATUS "native_srcs = ${native+_srcs}") 
add_library(Algorithm SHARED ${native_srcs}) # libAlgorithm.so
target_link_libraries(
    Algorithm
)
# 将源文件添加到可执行文件中
add_executable(t3DDemo ${native_srcs}) # T3DDemo 可执行文件
target_link_libraries(t3DDemo ${Open3D_LIBRARIES})
target_include_directories(t3DDemo PUBLIC ${Open3D_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
# include_directories("")


# 自动查找头文件路径函数 # https://blog.csdn.net/sep4075/article/details/120001415
macro(FIND_INCLUDE_DIR result curdir)
    file(GLOB_RECURSE children "${curdir}/*.hpp" "${curdir}/*.h") #遍历获取{curdir}中*.hpp和*.h文件列表
    set(dirlist "")
    foreach(child ${children})
        string(REGEX REPLACE "(.*)/.*)" "\\1" LIB_NAME $(child))
        if (IS_DIRECTORY ${LIB_NAME})
            list(FIND dirlist ${LIB_NAME} list_index)
            if($(list_index) LESS 0)
                LIST(APPEND dirlist ${LIB_NAME})
            endif()
        endif()
    endforeach()
    set(${result} ${dirlist})
endmacro()


