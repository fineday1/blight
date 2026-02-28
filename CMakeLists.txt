cmake_minimum_required(VERSION 3.14)
project(RadarEngineCpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find python and pybind11
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS "${pybind11_DIR}")

pybind11_add_module(radar_engine_cpp src/bindings.cpp src/octree_carver.cpp)

target_include_directories(radar_engine_cpp PRIVATE src ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(radar_engine_cpp PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
