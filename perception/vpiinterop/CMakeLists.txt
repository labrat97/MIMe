cmake_minimum_required(VERSION 3.4...3.18)
project(vpiinterop LANGUAGES CXX)

find_package(vpi REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

pybind11_add_module(vpiinterop denseFlow.cpp)

include_directories("${OpenCV_INCLUDE_DIRS}")
target_link_libraries(vpiinterop vpi torch cuda "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET vpiinterop PROPERTY CXX_STANDARD 14)
