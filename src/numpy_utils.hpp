#pragma once

#include <boost/python.hpp>
#include <boost/format.hpp>
#include <Eigen/Core>
#include <vector>

namespace util {

namespace py = boost::python;
extern py::object np_mod;

void PythonInit();

inline py::list toPyList(const std::vector<int>& x) {
  py::list out;
  for (int i=0; i < x.size(); ++i) out.append(x[i]);
  return out;
}

template<typename T>
std::vector<T> toVec(py::list l) {
  std::vector<T> out;
  const int len = py::len(l);
  out.resize(len);
  for (int i = 0; i < len; ++i) out[i] = py::extract<T>(l[i]);
  return out;
}

template<typename T>
struct type_traits {
  static const char* npname;
};

template <typename T>
T* getPointer(const py::object& arr) {
  long int i = py::extract<long int>(arr.attr("ctypes").attr("data"));
  T* p = (T*)i;
  return p;
}

template<typename T>
py::object toNdarray1(const T* data, size_t dim0) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0), type_traits<T>::npname);
  T* p = getPointer<T>(out);
  memcpy(p, data, dim0*sizeof(T));
  return out;
}
template<typename T>
py::object toNdarray2(const T* data, size_t dim0, size_t dim1) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1), type_traits<T>::npname);
  T* pout = getPointer<T>(out);
  memcpy(pout, data, dim0*dim1*sizeof(T));
  return out;
}
template<typename T>
py::object toNdarray3(const T* data, size_t dim0, size_t dim1, size_t dim2) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1, dim2), type_traits<T>::npname);
  T* pout = getPointer<T>(out);
  memcpy(pout, data, dim0*dim1*dim2*sizeof(T));
  return out;
}

template<typename Derived>
py::object toNdarray(const Eigen::DenseBase<Derived>& m) {
  typedef typename Eigen::DenseBase<Derived>::Scalar T;
  py::object out = np_mod.attr("empty")(py::make_tuple(m.rows(), m.cols()), type_traits<T>::npname);
  T* pout = getPointer<T>(out);
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      pout[m.cols()*i + j] = m(i,j);
    }
  }
  return out.attr("squeeze")();
}

// ensure C-order and data type, possibly making a new ndarray
template<typename T>
py::object ensureFormat(py::object a) {
  return np_mod.attr("ascontiguousarray")(a, type_traits<T>::npname);
}

template<typename Derived>
void fromNdarray(py::object a, Eigen::DenseBase<Derived>& out) {
  typedef typename Eigen::DenseBase<Derived>::Scalar T;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > RowMajorMapT;
  a = np_mod.attr("atleast_2d")(a);
  py::object shape = a.attr("shape");
  if (py::len(shape) != 2) {
    throw std::runtime_error((boost::format("expected 2-d array, got %d-d instead") % py::len(shape)).str());
  }
  a = ensureFormat<T>(a);
  out = RowMajorMapT(getPointer<T>(a), py::extract<int>(shape[0]), py::extract<int>(shape[1]));
}

} // namespace util
