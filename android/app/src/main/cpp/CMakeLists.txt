cmake_minimum_required(VERSION 3.22.1)
project("inav_core")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(
    inav_core
    SHARED
    inav_jni.cpp
)

find_library(
    log-lib
    log
)

target_link_libraries(
    inav_core
    ${log-lib}
)
