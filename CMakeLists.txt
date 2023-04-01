cmake_minimum_required(VERSION 3.24)
project(Serial)

set(CMAKE_CXX_STANDARD 17)
include_directories(src)

add_executable(Serial main.cpp src/KMeans.cpp)
