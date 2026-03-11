cmake_minimum_required(VERSION 3.10)
project(AlphaNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(AlphaNet
    main.cpp
    mathlinalg.cpp
    activations.cpp
    layers.cpp
    loss.cpp
    neuralnetwork.cpp
    data_loader.cpp
)

target_include_directories(AlphaNet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
