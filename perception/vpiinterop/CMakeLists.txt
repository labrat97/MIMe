cmake_minimum_required(VERSION 3.4...3.18)
project(vpiinterop LANGUAGES CXX)

find_package(vpi REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

pybind11_add_module(vpiinterop denseFlow.cpp)

target_link_libraries(vpiinterop vpi torch cuda "${TORCH_LIBRARIES}")
set_property(TARGET vpiinterop PROPERTY CXX_STANDARD 14)
