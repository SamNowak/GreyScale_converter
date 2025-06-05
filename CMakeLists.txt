cmake_minimum_required(VERSION 3.31)
project(EdgeDection CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(EdgeDection main.cu)

set_target_properties(EdgeDection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
