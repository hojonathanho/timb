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
