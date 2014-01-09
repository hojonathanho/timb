#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/slice.hpp>
#include <sstream>
#include "common.hpp"

// #include "tracking_problem.hpp"
#include "logging.hpp"
#include "numpy_utils.hpp"
#include "optimizer.hpp"
namespace py = boost::python;


/*
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
*/
// void to_scalar_field(py::object py, DoubleField& out) {
//   MatrixXd data;
//   util::fromNdarray(py, data); // TODO: extra memory copy here?
//   from_eigen(data, out);
// }
/*
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
*/

template<typename T>
vector<T> from_py_list(const py::list& lst) {
  vector<T> out;
  const int n = py::len(lst);
  out.reserve(n);
  for (int i = 0; i < n; ++i) {
    out.push_back(py::extract<T>(lst[i]));
  }
  return out;
}

template<typename T>
py::list to_py_list(const vector<T>& vec) {
  py::list lst;
  for (const T& x : vec) {
    lst.append(x);
  }
  return lst;
}

struct PyOptResult : public OptResult {
  py::object py_x;
  py::object py_cost_vals;
  py::object py_cost_over_iters;
  PyOptResult(OptResultPtr o) {
    status = o->status;
    py_x = util::toNdarray1(o->x.data(), o->x.size());
    cost = o->cost;
    py_cost_vals = util::toNdarray1(o->cost_vals.data(), o->cost_vals.size());
    py_cost_over_iters = to_py_list(o->cost_over_iters);
    n_func_evals = o->n_func_evals;
    n_jacobian_evals = o->n_jacobian_evals;
    n_qp_solves = o->n_qp_solves;
    n_iters = o->n_iters;
  }
};
typedef boost::shared_ptr<PyOptResult> PyOptResultPtr;

struct PyOptimizer : public Optimizer {
  py::list py_add_vars(py::list py_names) {
    vector<Var> out;
    add_vars(from_py_list<string>(py_names), out);
    return to_py_list(out);
  }
  void py_add_cost_1(CostFuncPtr cost) { add_cost(cost); }
  void py_add_cost_2(CostFuncPtr cost, double coeff=1.) { add_cost(cost, coeff); }
  PyOptResultPtr py_optimize(py::object py_start_x) {
    VectorXd start_x;
    util::from1darray(py_start_x, start_x);
    OptResultPtr result = optimize(start_x);
    return PyOptResultPtr(new PyOptResult(result));
  }
};
typedef boost::shared_ptr<PyOptimizer> PyOptimizerPtr;

struct SimpleCost : public CostFunc {
  Var m_var; double m_c; string m_name;
  SimpleCost(const Var& var, double c, const string& name) : m_var(var), m_c(c), m_name(name) { }
  string name() const { return m_name; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return true; }
  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out(0) = m_var.value(x) - m_c;
  }
  void linearize(const VectorXd&, CostFuncLinearization& lin) {
    lin.set_by_expr(0, m_var - m_c);
  }
};
typedef boost::shared_ptr<SimpleCost> SimpleCostPtr;



BOOST_PYTHON_MODULE(ctimbpy) {
  util::LoggingInit();
  util::PythonInit();

  // py::register_exception_translator<std::exception>(&bs::TranslateStdException);

  py::class_<std::vector<double> >("PyVec")
    .def(py::vector_indexing_suite<std::vector<double> >());


  py::class_<Var>("Var")
    .add_property("name", &Var::name)
    .def("__repr__", &Var::name)
    ;

  py::enum_<OptStatus>("OptStatus")
    .value("OPT_INCOMPLETE", OPT_INCOMPLETE)
    .value("OPT_CONVERGED", OPT_CONVERGED)
    .value("OPT_ITER_LIMIT", OPT_ITER_LIMIT)
    ;

  py::class_<PyOptResult, PyOptResultPtr>("OptResult", py::no_init)
    .def_readonly("status", &PyOptResult::status)
    .def_readonly("x", &PyOptResult::py_x)
    .def_readonly("cost", &PyOptResult::cost)
    .def_readonly("cost_over_iters", &PyOptResult::cost_over_iters)
    ;

  py::class_<OptParams>("OptParams", py::no_init)
    .def_readwrite("init_trust_region_size", &OptParams::init_trust_region_size)
    .def_readwrite("min_trust_region_size", &OptParams::min_trust_region_size)
    .def_readwrite("grad_convergence_tol", &OptParams::grad_convergence_tol)
    .def_readwrite("approx_improve_rel_tol", &OptParams::approx_improve_rel_tol)
    .def_readwrite("max_iter", &OptParams::max_iter)
    .def_readwrite("check_linearizations", &OptParams::check_linearizations)
    ;

  py::class_<PyOptimizer, PyOptimizerPtr>("Optimizer")
    .def("params", &PyOptimizer::params, py::return_internal_reference<>())
    .def("add_vars", &PyOptimizer::py_add_vars)
    .def("add_cost", &PyOptimizer::py_add_cost_1)
    .def("add_cost", &PyOptimizer::py_add_cost_2)
    .def("set_cost_coeff", &PyOptimizer::set_cost_coeff)
    // TODO: callback
    .def("num_vars", &PyOptimizer::num_vars)
    .def("optimize", &PyOptimizer::py_optimize)
    ;

  py::class_<CostFunc, CostFuncPtr, boost::noncopyable>("CostFunc", py::no_init)
    // Don't expose anything. Only subclasses should be used from Python
    ;

  py::class_<SimpleCost, SimpleCostPtr, py::bases<CostFunc> >("SimpleCost", py::init<const Var&, double, const string&>())
    ;

/*
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
*/
}
