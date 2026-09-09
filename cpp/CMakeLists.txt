cmake_minimum_required(VERSION 3.16)
project(iNAV_Core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optional ONNX Runtime support
find_package(OnnxRuntime QUIET)
if(OnnxRuntime_FOUND)
    add_compile_definitions(HAS_ONNXRUNTIME)
    message(STATUS "ONNX Runtime found. Enabling deep learning inference engine.")
else()
    message(STATUS "Building iNAV core with kinematic dead-reckoning filter.")
endif()

# Core static library
add_library(inav_core STATIC
    include/inav_filter.hpp
    include/inav_onnx.hpp
)

# Android JNI shared library (if building with Android NDK)
if(ANDROID)
    add_library(inav_jni SHARED
        ../android/jni/inav_jni.cpp
    )
    target_link_libraries(inav_jni inav_core log)
endif()

# Standalone C++ verification executable
add_executable(inav_demo src/main.cpp)
target_link_libraries(inav_demo inav_core)
