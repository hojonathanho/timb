#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/slice.hpp>
#include <sstream>
#include "tracking_problem.hpp"
#include "logging.hpp"
#include "numpy_utils.hpp"
namespace py = boost::python;

struct PyTrackingProblemResult {
  py::object phi, u, next_phi, next_omega;
  OptResultPtr opt_result;

  PyTrackingProblemResult(const TrackingProblemResult& res) {
    phi = util::toNdarray(to_eigen(res.phi));

    py::object u_x = util::toNdarray(to_eigen(res.u_x));
    py::object u_y = util::toNdarray(to_eigen(res.u_y));
    u = util::np_mod.attr("dstack")(py::make_tuple(u_x, u_y));

    next_phi = util::toNdarray(to_eigen(res.next_phi));

    next_omega = util::toNdarray(to_eigen(res.next_omega));

    opt_result = res.opt_result;
  }
};
typedef boost::shared_ptr<PyTrackingProblemResult> PyTrackingProblemResultPtr;

void to_scalar_field(py::object py, DoubleField& out) {
  MatrixXd data;
  util::fromNdarray(py, data); // TODO: extra memory copy here?
  from_eigen(data, out);
}

class PyTrackingProblem : public TrackingProblem {
public:
  PyTrackingProblem(double xmin_, double xmax_, double ymin_, double ymax_, int nx_, int ny_) : TrackingProblem(xmin_, xmax_, ymin_, ymax_, nx_, ny_) { }

  void py_set_obs(py::object py_vals, py::object py_weights, py::object py_mask) {
    DoubleField vals(m_ctx->grid_params), weights(m_ctx->grid_params), mask(m_ctx->grid_params);
    to_scalar_field(py_vals, vals);
    to_scalar_field(py_weights, weights);
    to_scalar_field(py_mask, mask);
    set_obs(vals, weights, mask);
  }

  void py_set_prior(py::object py_mean, py::object py_omega) {
    DoubleField mean(m_ctx->grid_params), omega(m_ctx->grid_params);
    to_scalar_field(py_mean, mean);
    to_scalar_field(py_omega, omega);
    set_prior(mean, omega);
  }

  void py_set_init_u(py::object py_u) {
    DoubleField u_x(m_ctx->grid_params), u_y(m_ctx->grid_params);
    to_scalar_field(py_u[py::make_tuple(py::slice(),py::slice(),0)], u_x);
    to_scalar_field(py_u[py::make_tuple(py::slice(),py::slice(),1)], u_y);
    set_init_u(u_x, u_y);
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
    .def_readwrite("prior", &TrackingProblemCoeffs::prior)
    ;

  py::class_<PyTrackingProblemResult, PyTrackingProblemResultPtr>("TrackingProblemResult", py::no_init)
    .def_readwrite("phi", &PyTrackingProblemResult::phi)
    .def_readwrite("u", &PyTrackingProblemResult::u)
    .def_readwrite("next_phi", &PyTrackingProblemResult::next_phi)
    .def_readwrite("next_omega", &PyTrackingProblemResult::next_omega)
    .def_readonly("opt_result", &PyTrackingProblemResult::opt_result)
    ;

  py::class_<PyTrackingProblem, PyTrackingProblemPtr>("TrackingProblem", py::init<double, double, double, double, int, int>())
    .def("set_obs", &PyTrackingProblem::py_set_obs)
    .def("set_prior", &PyTrackingProblem::py_set_prior)
    .def("set_init_u", &PyTrackingProblem::py_set_init_u)
    .def_readwrite("coeffs", &PyTrackingProblem::m_coeffs)
    .def("optimize", &PyTrackingProblem::py_optimize)
    ;
}
