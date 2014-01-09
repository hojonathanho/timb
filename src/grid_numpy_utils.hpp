#pragma once

#include "grid.hpp"
#include "numpy_utils.hpp"

#include <boost/python.hpp>
namespace py = boost::python;

inline void from_numpy(py::object py, DoubleField& out) {
  MatrixXd data;
  util::fromNdarray(py, data); // TODO: extra memory copy here?
  from_eigen(data, out);
}

inline py::object to_numpy(const DoubleField& f) {
  return util::toNdarray2(f.data(), f.grid_params().nx, f.grid_params().ny);
}
