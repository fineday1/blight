#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "octree_carver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(radar_engine_cpp, m) {
    m.doc() = "C++ High Performance Radar Engine";

    py::class_<OctreeCarver>(m, "OctreeCarver")
        .def(py::init<std::array<double, 2>, std::array<double, 2>, std::array<double, 2>, int, int>(),
             py::arg("x_limits"), py::arg("y_limits"), py::arg("z_limits"), 
             py::arg("max_depth") = 5, py::arg("min_consensus") = 2)
        .def("solve", &OctreeCarver::solve, 
             py::arg("masks"), py::arg("cameras"),
             "Solve voxel carving using octree")
        .def("test_projection", &OctreeCarver::test_projection,
             py::arg("cameras"), "Test projection logic");
}
