cmake_minimum_required(VERSION 3.4...3.18)
project(vpiinterop LANGUAGES CXX)

find_package(vpi REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(pybind11 REQUIRED)

pybind11_add_module(vpiinterop denseFlow.cpp)

target_link_libraries(vpiinterop vpi torch "${TORCH_LIBRARIES}")
set_property(TARGET vpiinterop PROPERTY CXX_STANDARD 14)
