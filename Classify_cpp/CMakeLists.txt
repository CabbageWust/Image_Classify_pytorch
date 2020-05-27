cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(Classify_cpp)

# 设置Opencv的CMake路径
set(OpenCV_DIR /usr/local/share/OpenCV)
find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)
if(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()
set(CMAKE_PREFIX_PATH /home/yinliang/anaconda3/envs/pytorch2/lib/python3.6/site-packages/torch)
find_package(Torch REQUIRED)

#设置编译器版本
SET(CMAKE_C_COMPILER g++)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-std=c++11 -fno-stack-protector) # very important key in TK1,otherwise will raise an error call stack smashing detected
    message(STATUS "optional:-std=c++11")
endif(CMAKE_COMPILER_IS_GNUCXX)

add_executable(${PROJECT_NAME} classify.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
SET(CMAKE_BUILD_TYPE DEBUG)
