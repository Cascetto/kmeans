cmake_minimum_required(VERSION 3.24)
project(Parallel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
include_directories(src)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(Parallel main.cu)
