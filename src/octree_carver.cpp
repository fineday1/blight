#include "octree_carver.hpp"

OctreeCarver::OctreeCarver(std::array<double, 2> x_limits, 
                           std::array<double, 2> y_limits, 
                           std::array<double, 2> z_limits, 
                           int max_depth, 
                           int min_consensus)
    : x_limits(x_limits), y_limits(y_limits), z_limits(z_limits), 
      max_depth(max_depth), min_consensus(min_consensus) {}

Eigen::Matrix<double, 4, 27> OctreeCarver::get_samples(const std::array<double, 2>& xr, 
                                                       const std::array<double, 2>& yr, 
                                                       const std::array<double, 2>& zr) {
    Eigen::Matrix<double, 4, 27> pts;
    int idx = 0;
    double x_step = (xr[1] - xr[0]) / 2.0;
    double y_step = (yr[1] - yr[0]) / 2.0;
    double z_step = (zr[1] - zr[0]) / 2.0;
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                pts(0, idx) = xr[0] + i * x_step;
                pts(1, idx) = yr[0] + j * y_step;
                pts(2, idx) = zr[0] + k * z_step;
                pts(3, idx) = 1.0;
                idx++;
            }
        }
    }
    return pts;
}

bool OctreeCarver::is_voxel_active(const std::array<double, 2>& xr, 
                                   const std::array<double, 2>& yr, 
                                   const std::array<double, 2>& zr, 
                                   const std::vector<cv::Mat>& masks, 
                                   const std::vector<Eigen::Matrix<double, 3, 4>>& cameras,
                                   int depth) {
    // 8 corners
    Eigen::Matrix<double, 4, 8> corners;
    corners << xr[0], xr[0], xr[0], xr[0], xr[1], xr[1], xr[1], xr[1],
               yr[0], yr[0], yr[1], yr[1], yr[0], yr[0], yr[1], yr[1],
               zr[0], zr[1], zr[0], zr[1], zr[0], zr[1], zr[0], zr[1],
               1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0;
               
    int consensus = 0;
    
    for (size_t c = 0; c < masks.size(); ++c) {
        const auto& P = cameras[c];
        const auto& mask = masks[c];
        int h = mask.rows;
        int w = mask.cols;
        
        Eigen::Matrix<double, 3, 8> uv_hom = P * corners;
        
        int min_u = w, max_u = -1;
        int min_v = h, max_v = -1;
        bool all_behind = true;
        
        for (int i = 0; i < 8; ++i) {
            double z_val = uv_hom(2, i);
            if (z_val > 0) {
                all_behind = false;
                int u = static_cast<int>(uv_hom(0, i) / z_val);
                int v = static_cast<int>(uv_hom(1, i) / z_val);
                if (u < min_u) min_u = u;
                if (u > max_u) max_u = u;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
        }
        
        if (all_behind) continue;
        
        // Clamp to image
        min_u = std::max(0, std::min(min_u, w - 1));
        max_u = std::max(0, std::min(max_u, w - 1));
        min_v = std::max(0, std::min(min_v, h - 1));
        max_v = std::max(0, std::min(max_v, h - 1));
        
        if (min_u > max_u || min_v > max_v) continue;
        
        // Check if any pixel in the ROI is > 0
        cv::Mat roi = mask(cv::Rect(min_u, min_v, max_u - min_u + 1, max_v - min_v + 1));
        
        // A fast way to check if there are non-zero pixels
        // Since we are looking for ANY non-zero, cv::checkRange or cv::countNonZero works.
        // For small ROIs, iterating might be faster, but let's just use OpenCV's built-in.
        if (cv::countNonZero(roi) > 0) {
            consensus++;
        }
    }
    
    int target_consensus = (depth > 0) ? min_consensus : 1;
    return consensus >= target_consensus;
}

bool OctreeCarver::is_voxel_active(const std::array<double, 2>& xr, 
                         const std::array<double, 2>& yr, 
                         const std::array<double, 2>& zr, 
                         const std::vector<py::array_t<uint8_t>>& masks_py, 
                         const std::vector<py::array_t<double>>& cameras_py,
                         int depth) {
    std::vector<cv::Mat> masks;
    for (const auto& mask_py : masks_py) {
        auto req = mask_py.request();
        masks.emplace_back(req.shape[0], req.shape[1], CV_8UC1, req.ptr, req.strides[0]);
    }
    
    std::vector<Eigen::Matrix<double, 3, 4>> cameras;
    for (const auto& cam_py : cameras_py) {
        auto req = cam_py.request();
        Eigen::Matrix<double, 3, 4> P;
        double* ptr = static_cast<double*>(req.ptr);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                P(r, c) = ptr[r * 4 + c];
            }
        }
        cameras.push_back(P);
    }
    return is_voxel_active(xr, yr, zr, masks, cameras, depth);
}

void OctreeCarver::recursive_carve(const std::array<double, 2>& xr, 
                                   const std::array<double, 2>& yr, 
                                   const std::array<double, 2>& zr, 
                                   int depth, 
                                   const std::vector<cv::Mat>& masks, 
                                   const std::vector<Eigen::Matrix<double, 3, 4>>& cameras,
                                   std::vector<std::array<double, 3>>& points) {
    
    if (!is_voxel_active(xr, yr, zr, masks, cameras, depth)) {
        return;
    }
    
    if (depth >= max_depth) {
        points.push_back({(xr[0] + xr[1]) / 2.0, (yr[0] + yr[1]) / 2.0, (zr[0] + zr[1]) / 2.0});
        return;
    }
    
    double x_m = (xr[0] + xr[1]) / 2.0;
    double y_m = (yr[0] + yr[1]) / 2.0;
    double z_m = (zr[0] + zr[1]) / 2.0;
    
    std::array<std::array<double, 2>, 2> x_splits = {{{xr[0], x_m}, {x_m, xr[1]}}};
    std::array<std::array<double, 2>, 2> y_splits = {{{yr[0], y_m}, {y_m, yr[1]}}};
    std::array<std::array<double, 2>, 2> z_splits = {{{zr[0], z_m}, {z_m, zr[1]}}};
    
    for (const auto& dx : x_splits) {
        for (const auto& dy : y_splits) {
            for (const auto& dz : z_splits) {
                recursive_carve(dx, dy, dz, depth + 1, masks, cameras, points);
            }
        }
    }
}

py::array_t<double> OctreeCarver::solve(const std::vector<py::array_t<uint8_t>>& masks_py,
                                        const std::vector<py::array_t<double>>& cameras_py) {
    
    std::vector<cv::Mat> masks;
    for (const auto& mask_py : masks_py) {
        auto req = mask_py.request();
        // Assuming 2D uint8 numpy arrays
        masks.emplace_back(req.shape[0], req.shape[1], CV_8UC1, req.ptr, req.strides[0]);
    }
    
    std::vector<Eigen::Matrix<double, 3, 4>> cameras;
    for (const auto& cam_py : cameras_py) {
        auto req = cam_py.request();
        Eigen::Matrix<double, 3, 4> P;
        double* ptr = static_cast<double*>(req.ptr);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                P(r, c) = ptr[r * 4 + c];
            }
        }
        cameras.push_back(P);
    }
    
    std::vector<std::array<double, 3>> points;
    recursive_carve(x_limits, y_limits, z_limits, 0, masks, cameras, points);
    
    if (points.empty()) {
        return py::array_t<double>(py::array::ShapeContainer({0, 3}));
    }
    
    py::array_t<double> result({static_cast<ssize_t>(points.size()), static_cast<ssize_t>(3)});
    auto req = result.request();
    double* ptr = static_cast<double*>(req.ptr);
    
    for (size_t i = 0; i < points.size(); ++i) {
        ptr[i * 3 + 0] = points[i][0];
        ptr[i * 3 + 1] = points[i][1];
        ptr[i * 3 + 2] = points[i][2];
    }
    
    return result;
}

void OctreeCarver::test_projection(const std::vector<py::array_t<double>>& cameras_py) {
    std::vector<Eigen::Matrix<double, 3, 4>> cameras;
    for (const auto& cam_py : cameras_py) {
        auto req = cam_py.request();
        Eigen::Matrix<double, 3, 4> P;
        double* ptr = static_cast<double*>(req.ptr);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                P(r, c) = ptr[r * 4 + c];
            }
        }
        cameras.push_back(P);
    }

    Eigen::Matrix<double, 4, 1> pt;
    pt << 4.318187, 1.75069, 5.477612, 1.0;
    Eigen::Matrix<double, 3, 1> uv_hom = cameras[0] * pt;
    int u = static_cast<int>(uv_hom(0) / uv_hom(2));
    int v = static_cast<int>(uv_hom(1) / uv_hom(2));
    std::cout << "C++ projection: " << u << " " << v << std::endl;
}
