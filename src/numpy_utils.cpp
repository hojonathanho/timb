#include "numpy_utils.hpp"

namespace util {

template<> const char* type_traits<float>::npname = "float32";
template<> const char* type_traits<int>::npname = "int32";
template<> const char* type_traits<double>::npname = "float64";
template<> const char* type_traits<unsigned char>::npname = "uint8";

py::object np_mod;

void PythonInit() {
  np_mod = py::import("numpy");
}

} // namespace util
