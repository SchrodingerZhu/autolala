cmake_minimum_required(VERSION 3.28)
project(test)

include(cmake/barvinok.cmake)
include(cmake/mlir.cmake)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test main.cpp)
target_link_libraries(test PUBLIC barvinok)

add_subdirectory(src)
