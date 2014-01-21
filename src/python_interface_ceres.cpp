#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/slice.hpp>
#include <sstream>
#include "common.hpp"

#include "tracking_ceres.hpp"
#include "tracking_utils.hpp"
#include "logging.hpp"
#include "numpy_utils.hpp"
#include "grid.hpp"
#include "grid_numpy_utils.hpp"

namespace py = boost::python;

using namespace timb_ceres;

struct PyOptResult {
  PyOptResult(const DoubleField& phi, const DoubleField& u_x, const DoubleField& u_y) {
    py_phi = to_numpy(phi);
    py_u_x = to_numpy(u_x);
    py_u_y = to_numpy(u_y);
  }
  py::object py_phi, py_u_x, py_u_y;
};

struct PyCeresTrackingProblem : public CeresTrackingProblem {
  PyCeresTrackingProblem(const GridParams& gp) : CeresTrackingProblem(gp) { }
  PyOptResult py_optimize() {
    optimize();
    return PyOptResult(m_phi, m_u_x, m_u_y);
  }
};
typedef boost::shared_ptr<PyCeresTrackingProblem> PyCeresTrackingProblemPtr;

BOOST_PYTHON_MODULE(ctimb_ceres) {
  util::LoggingInit();
  util::PythonInit();

  py::class_<GridParams>("GridParams", py::init<double, double, double, double, int, int>())
    .def_readonly("xmin", &GridParams::xmin)
    .def_readonly("xmax", &GridParams::xmax)
    .def_readonly("ymin", &GridParams::ymin)
    .def_readonly("ymax", &GridParams::ymax)
    .def_readonly("nx", &GridParams::nx)
    .def_readonly("ny", &GridParams::ny)
    .def_readonly("eps_x", &GridParams::eps_x)
    .def_readonly("eps_y", &GridParams::eps_y)
    ;

  py::class_<PyCeresTrackingProblem, PyCeresTrackingProblemPtr>("TrackingProblem", py::init<const GridParams&>())
    .def_readwrite("flow_norm_coeff", &PyCeresTrackingProblem::m_flow_norm_coeff)
    .def_readwrite("flow_rigidity_coeff", &PyCeresTrackingProblem::m_flow_rigidity_coeff)
    .def_readwrite("observation_coeff", &PyCeresTrackingProblem::m_observation_coeff)
    .def_readwrite("agreement_coeff", &PyCeresTrackingProblem::m_agreement_coeff)
    .def("optimize", &PyCeresTrackingProblem::py_optimize)
    ;

  py::class_<PyOptResult>("OptResult", py::no_init)
    .def_readonly("phi", &PyOptResult::py_phi)
    .def_readonly("u_x", &PyOptResult::py_u_x)
    .def_readonly("u_y", &PyOptResult::py_u_y)
    ;
}
