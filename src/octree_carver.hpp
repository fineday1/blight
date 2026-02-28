#pragma once

#include <vector>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace py = pybind11;

class OctreeCarver {
public:
    OctreeCarver(std::array<double, 2> x_limits, 
                 std::array<double, 2> y_limits, 
                 std::array<double, 2> z_limits, 
                 int max_depth = 5, 
                 int min_consensus = 2);

    py::array_t<double> solve(const std::vector<py::array_t<uint8_t>>& masks_py,
                              const std::vector<py::array_t<double>>& cameras_py);

    void test_projection(const std::vector<py::array_t<double>>& cameras_py);

    bool is_voxel_active(const std::array<double, 2>& xr, 
                         const std::array<double, 2>& yr, 
                         const std::array<double, 2>& zr, 
                         const std::vector<py::array_t<uint8_t>>& masks_py, 
                         const std::vector<py::array_t<double>>& cameras_py,
                         int depth);

private:
    std::array<double, 2> x_limits;
    std::array<double, 2> y_limits;
    std::array<double, 2> z_limits;
    int max_depth;
    int min_consensus;

    Eigen::Matrix<double, 4, 27> get_samples(const std::array<double, 2>& xr, 
                                             const std::array<double, 2>& yr, 
                                             const std::array<double, 2>& zr);

    bool is_voxel_active(const std::array<double, 2>& xr, 
                         const std::array<double, 2>& yr, 
                         const std::array<double, 2>& zr, 
                         const std::vector<cv::Mat>& masks, 
                         const std::vector<Eigen::Matrix<double, 3, 4>>& cameras,
                         int depth);

    void recursive_carve(const std::array<double, 2>& xr, 
                         const std::array<double, 2>& yr, 
                         const std::array<double, 2>& zr, 
                         int depth, 
                         const std::vector<cv::Mat>& masks, 
                         const std::vector<Eigen::Matrix<double, 3, 4>>& cameras,
                         std::vector<std::array<double, 3>>& points);
};
