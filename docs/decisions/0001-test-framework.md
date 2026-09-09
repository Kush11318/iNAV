# ADR 0001: Choice of Unit Testing Framework for C++ Core Engine

## Status
Accepted

## Context
The core navigation engine (`core/`) requires an automated unit testing suite that can run cleanly across multiple environments:
- Local developer machines (Windows with CMake and Ninja/MinGW/Clang/MSVC)
- Continuous Integration runners (Linux GitHub Actions)
- Embedded Linux and edge toolchains
- Android NDK integration in Phase 3

The architecture specification allows choosing between **Catch2** and **GoogleTest**.

## Decision
We choose **GoogleTest (v1.14.0)** as the official unit testing framework, integrated via CMake `FetchContent`.

## Rationale
1. **Universal Compiler Compatibility**: GoogleTest is mature, battle-tested, and compiles reliably across GCC (including older MinGW versions), Clang, Apple Clang, and MSVC without deprecation or standard library version mismatches.
2. **First-Class Android NDK Support**: GoogleTest is the native, official testing framework used by the Android NDK and Android Open Source Project (AOSP). Using GoogleTest in `core/` guarantees seamless testability in Phase 3 under Android toolchains.
3. **Aerospace and Robotics Standard**: Space agencies (ISRO, NASA, ESA) and ROS/ROS2 ecosystems predominantly standardize on GoogleTest for navigation and sensor fusion stacks.
4. **Automated Discovery**: With CMake's native `include(GoogleTest)` and `gtest_discover_tests(idr_tests)`, test cases are automatically discovered and registered with `CTest` without manual maintenance.

## Consequences
- Clean assertions using `EXPECT_TRUE`, `EXPECT_FALSE`, `EXPECT_DOUBLE_EQ`.
- Automated test discovery in `ctest`.
- Seamless CI execution on both Linux and Windows runners.
