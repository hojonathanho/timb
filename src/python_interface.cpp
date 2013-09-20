#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "tracking_problem.hpp"
#include "logging.hpp"
#include "numpy_utils.hpp"
namespace py = boost::python;

struct PyTrackingProblemResult {
  py::object phi, u;
  OptResultPtr opt_result;

  PyTrackingProblemResult(const TrackingProblemResult& res) {
    phi = util::toNdarray(to_eigen(res.phi));
    py::object u_x = util::toNdarray(to_eigen(res.u_x));
    py::object u_y = util::toNdarray(to_eigen(res.u_y));
    u = util::np_mod.attr("dstack")(py::make_tuple(u_x, u_y));
    opt_result = res.opt_result;
  }
};
typedef boost::shared_ptr<PyTrackingProblemResult> PyTrackingProblemResultPtr;


class PyTrackingProblem : public TrackingProblem {
public:
  PyTrackingProblem(double xmin_, double xmax_, double ymin_, double ymax_, int nx_, int ny_) : TrackingProblem(xmin_, xmax_, ymin_, ymax_, nx_, ny_) { }

  void py_set_observation_points(py::object py_pts) {
    MatrixX2d pts;
    util::fromNdarray(py_pts, pts);
    set_observation_points(pts);
  }

  void py_set_prev_phi(py::object py_prev_phi) {
    // TODO: there is an extra memory copy here... ?
    MatrixXd prev_phi_data;
    util::fromNdarray(py_prev_phi, prev_phi_data);
    DoubleField prev_phi(m_ctx->grid_params);
    from_eigen(prev_phi_data, prev_phi);
    set_prev_phi(prev_phi);
  }

  PyTrackingProblemResultPtr py_optimize() {
    return PyTrackingProblemResultPtr(new PyTrackingProblemResult(*optimize()));
  }
};
typedef boost::shared_ptr<PyTrackingProblem> PyTrackingProblemPtr;

BOOST_PYTHON_MODULE(ctimbpy) {
  util::LoggingInit();
  util::PythonInit();

  // py::register_exception_translator<std::exception>(&bs::TranslateStdException);

  py::class_<std::vector<double> >("PyVec")
    .def(py::vector_indexing_suite<std::vector<double> >());

  py::class_<OptResult, OptResultPtr>("OptResult", py::no_init)
    .def_readonly("status", &OptResult::status)
    .def_readonly("cost", &OptResult::cost)
    .def_readonly("cost_over_iters", &OptResult::cost_over_iters)
    ;

  py::class_<TrackingProblemCoeffs, TrackingProblemCoeffsPtr>("TrackingProblemCoeffs")
    .def_readwrite("flow_norm", &TrackingProblemCoeffs::flow_norm)
    .def_readwrite("flow_rigidity", &TrackingProblemCoeffs::flow_rigidity)
    .def_readwrite("observation", &TrackingProblemCoeffs::observation)
    .def_readwrite("phi_agreement", &TrackingProblemCoeffs::phi_agreement)
    ;

  py::class_<PyTrackingProblemResult, PyTrackingProblemResultPtr>("TrackingProblemResult", py::no_init)
    .def_readwrite("phi", &PyTrackingProblemResult::phi)
    .def_readwrite("u", &PyTrackingProblemResult::u)
    .def_readonly("opt_result", &PyTrackingProblemResult::opt_result)
    ;

  py::class_<PyTrackingProblem, PyTrackingProblemPtr>("TrackingProblem", py::init<double, double, double, double, int, int>())
    .def("set_observation_points", &PyTrackingProblem::py_set_observation_points)
    .def("set_prev_phi", &PyTrackingProblem::py_set_prev_phi)
    .def_readwrite("coeffs", &PyTrackingProblem::m_coeffs)
    .def("optimize", &PyTrackingProblem::py_optimize)
    ;
}
