cmake_minimum_required(VERSION 3.26)

project(conv_demo)

set(ncnn_DIR "/opt/ncnn/build/install/lib/cmake/ncnn")

find_package(ncnn REQUIRED)

add_executable(conv_demo conv_high_low_level_demo.cpp)

target_link_libraries(conv_demo ncnn)

add_executable(inner_product_demo inner_product_low_level_demo.cpp)

target_link_libraries(inner_product_demo ncnn)

add_executable(reshape_low_level_demo reshape_low_level_demo.cpp)

target_link_libraries(reshape_low_level_demo ncnn)
