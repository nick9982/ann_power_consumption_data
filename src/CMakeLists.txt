cmake_minimum_required(VERSION 3.16)

project(nn)

add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(data_processing)
add_subdirectory(neural_network)

target_link_libraries(${PROJECT_NAME} neural_net process_power_consumption)