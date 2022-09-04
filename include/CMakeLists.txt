cmake_minimum_required(VERSION 3.16)

project(nn)

add_subdirectory(data_processing)

target_include_directories(${This} PRIVATE ./)