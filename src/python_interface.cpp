#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/slice.hpp>
#include <sstream>
#include "common.hpp"

#include "tracking_problem.hpp"
#include "tracking_problem_rigid.hpp"
#include "tracking_utils.hpp"
#include "logging.hpp"
#include "numpy_utils.hpp"
#include "grid_numpy_utils.hpp"
#include "optimizer.hpp"

namespace py = boost::python;

// struct PyOptResult : public OptResult {
//   py::object py_x;
//   py::list py_x_over_iters;
//   py::object py_cost_detail;
//   py::object py_cost_over_iters;
//   PyOptResult(OptResultPtr o) {
//     status = o->status;
//     py_x = util::toNdarray1(o->x.data(), o->x.size());
//     for (const VectorXd& x : o->x_over_iters) {
//       py_x_over_iters.append(util::toNdarray1(x.data(), x.size()));
//     }
//     cost = o->cost;
//     py_cost_detail = util::toNdarray1(o->cost_detail.data(), o->cost_detail.size());
//     py_cost_over_iters = util::toPyList(o->cost_over_iters);
//     n_func_evals = o->n_func_evals;
//     n_jacobian_evals = o->n_jacobian_evals;
//     n_qp_solves = o->n_qp_solves;
//     n_iters = o->n_iters;
//   }
// };
// typedef boost::shared_ptr<PyOptResult> PyOptResultPtr;

py::dict optresult_to_dict(OptResultPtr o) {
  py::dict d;
  d["status"] = o->status;
  d["x"] = util::toNdarray1(o->x.data(), o->x.size());
  py::list x_over_iters;
  for (const VectorXd& x : o->x_over_iters) {
    x_over_iters.append(util::toNdarray1(x.data(), x.size()));
  }
  d["x_over_iters"] = x_over_iters;
  d["cost"] = o->cost;
  d["cost_detail"] = util::toNdarray1(o->cost_detail.data(), o->cost_detail.size());
  d["cost_over_iters"] = util::toPyList(o->cost_over_iters);
  d["n_func_evals"] = o->n_func_evals;
  d["n_jacobian_evals"] = o->n_jacobian_evals;
  d["n_qp_solves"] = o->n_qp_solves;
  d["n_iters"] = o->n_iters;
  return d;
}

struct GridParams_pickle_suite : py::pickle_suite {
  static py::tuple getinitargs(const GridParams& gp) {
    return py::make_tuple(gp.xmin, gp.xmax, gp.ymin, gp.ymax, gp.nx, gp.ny);
  }
};

struct PyOptimizer : public Optimizer {
  py::list py_add_vars(py::list py_names) {
    vector<Var> out;
    add_vars(util::toVec<string>(py_names), out);
    return util::toPyList(out);
  }
  void py_add_cost_1(CostFuncPtr cost) { add_cost(cost); }
  void py_add_cost_2(CostFuncPtr cost, double coeff) { add_cost(cost, coeff); }
  py::dict py_optimize(py::object py_start_x) {
    VectorXd start_x;
    util::from1darray(py_start_x, start_x);
    OptResultPtr result = optimize(start_x);
    // return PyOptResultPtr(new PyOptResult(result));
    return optresult_to_dict(result);
  }
};
typedef boost::shared_ptr<PyOptimizer> PyOptimizerPtr;


static VarField py_make_var_field(PyOptimizerPtr py_opt, const string& prefix, const GridParams& gp) {
  VarField out(gp);
  make_field_vars(prefix, *py_opt, out);
  return out;
}

static py::object py_apply_flow(const GridParams& gp, py::object py_phi, py::object py_u_x, py::object py_u_y) {
  DoubleField phi(gp), u_x(gp), u_y(gp), flowed_phi(gp);
  from_numpy(py_phi, phi);
  from_numpy(py_u_x, u_x);
  from_numpy(py_u_y, u_y);
  apply_flow(phi, u_x, u_y, flowed_phi);
  return to_numpy(flowed_phi);
}

// static py::object py_march_from_zero_crossing(py::object py_phi, bool propagate_sign=true, py::object py_ignore_mask=py::object()) {
//   MatrixXd phi;
//   util::fromNdarray(py_phi, phi);

//   boost::scoped_ptr<MatrixXi> pignore_mask;
//   if (py_ignore_mask != py::object()) {
//     pignore_mask.reset(new MatrixXi);
//     util::fromNdarray(py_ignore_mask, *pignore_mask);
//   }

//   MatrixXd out;
//   march_from_zero_crossing(phi, propagate_sign, pignore_mask.get(), out);

//   return util::toNdarray(out);
// }
// BOOST_PYTHON_FUNCTION_OVERLOADS(py_march_from_zero_crossing_overloads, py_march_from_zero_crossing, 1, 3)

// static py::object py_compute_flowed_precision(const GridParams& gp, py::object py_precision_diag, py::object py_u_x, py::object py_u_y, bool direct) {
//   DoubleField u_x(gp), u_y(gp);
//   from_numpy(py_u_x, u_x);
//   from_numpy(py_u_y, u_y);
//   VectorXd precision_diag;
//   util::from1darray(py_precision_diag, precision_diag);
//   VectorXd out;
//   if (direct) {
//     compute_flowed_precision_direct(precision_diag, u_x, u_y, out);
//   } else {
//     compute_flowed_precision(precision_diag, u_x, u_y, out);
//   }
//   return util::toNdarray1(out.data(), out.size());
// }

static string py_print_gridparams(const GridParams* gp) {
  return (boost::format("GridParams: x: [%f, %f], nx: %d, y: [%f, %f], ny: %d")
    % gp->xmin % gp->xmax % gp->nx % gp->ymin % gp->ymax % gp->ny).str();
}

DoubleField py_make_double_field(const GridParams& gp, py::object py_gdata) {
  DoubleField dbf(gp);
  from_numpy(py_gdata, dbf);
  return dbf;
}

py::object py_apply_rigid_transform(const DoubleField &phi, double dx, double dy, double dth) {
  return to_numpy(apply_rigid_transform(phi, dx, dy, dth));
}

struct ExampleCost : public CostFunc {
  Var m_var; double m_c; string m_name;
  ExampleCost(const Var& var, double c, const string& name) : m_var(var), m_c(c), m_name(name) { }
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
typedef boost::shared_ptr<ExampleCost> ExampleCostPtr;


BOOST_PYTHON_MODULE(ctimb) {
  util::LoggingInit();
  util::PythonInit();

  py::class_<Var>("Var")
    .add_property("name", &Var::name)
    .def("__repr__", &Var::name)
    ;

  py::enum_<OptStatus>("OptStatus")
    .value("OPT_INCOMPLETE", OPT_INCOMPLETE)
    .value("OPT_CONVERGED", OPT_CONVERGED)
    .value("OPT_ITER_LIMIT", OPT_ITER_LIMIT)
    ;

  // py::class_<PyOptResult, PyOptResultPtr>("OptResult", py::no_init)
  //   .def_readonly("status", &PyOptResult::status)
  //   .def_readonly("x", &PyOptResult::py_x)
  //   .def_readonly("cost", &PyOptResult::cost)
  //   .def_readonly("cost_detail", &PyOptResult::py_cost_detail)
  //   .def_readonly("cost_over_iters", &PyOptResult::py_cost_over_iters)
  //   .def_readonly("x_over_iters", &PyOptResult::py_x_over_iters)
  //   ;

  py::class_<OptParams>("OptParams", py::no_init)
    .def_readwrite("init_trust_region_size", &OptParams::init_trust_region_size)
    .def_readwrite("min_trust_region_size", &OptParams::min_trust_region_size)
    .def_readwrite("grad_convergence_tol", &OptParams::grad_convergence_tol)
    .def_readwrite("approx_improve_rel_tol", &OptParams::approx_improve_rel_tol)
    .def_readwrite("max_iter", &OptParams::max_iter)
    .def_readwrite("check_linearizations", &OptParams::check_linearizations)
    .def_readwrite("keep_results_over_iterations", &OptParams::keep_results_over_iterations)
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

  py::class_<GridParams>("GridParams", py::init<double, double, double, double, int, int>())
    .def_readonly("xmin", &GridParams::xmin)
    .def_readonly("xmax", &GridParams::xmax)
    .def_readonly("ymin", &GridParams::ymin)
    .def_readonly("ymax", &GridParams::ymax)
    .def_readonly("nx", &GridParams::nx)
    .def_readonly("ny", &GridParams::ny)
    .def_readonly("eps_x", &GridParams::eps_x)
    .def_readonly("eps_y", &GridParams::eps_y)
    .def("__repr__", &py_print_gridparams)
    .def_pickle(GridParams_pickle_suite())
    ;

  py::class_<VarField>("VarField", py::no_init);
  py::class_<DoubleField>("DoubleField", py::no_init);


  py::class_<ExampleCost, ExampleCostPtr, py::bases<CostFunc> >("ExampleCost", py::init<const Var&, double, const string&>());
  py::class_<FlowNormCost, FlowNormCostPtr, py::bases<CostFunc> >("FlowNormCost", py::init<const VarField&, const VarField&>());
  py::class_<FlowRigidityCost, FlowRigidityCostPtr, py::bases<CostFunc> >("FlowRigidityCost", py::init<const VarField&, const VarField&>());
  py::class_<GradientCost, GradientCostPtr, py::bases<CostFunc> >("GradientCost", py::init<const VarField&>());
  py::class_<LaplacianCost, LaplacianCostPtr, py::bases<CostFunc> >("LaplacianCost", py::init<const VarField&>());
  py::class_<TPSCost, TPSCostPtr, py::bases<CostFunc> >("TPSCost", py::init<const VarField&>());
  py::class_<ObservationCost, ObservationCostPtr, py::bases<CostFunc> >("ObservationCost", py::init<const VarField&>())
    .def("set_observation", &ObservationCost::py_set_observation)
    ;
  py::class_<ObservationZeroCrossingCost, ObservationZeroCrossingCostPtr, py::bases<CostFunc> >("ObservationZeroCrossingCost", py::init<const VarField&>())
    .def("set_zero_points", &ObservationZeroCrossingCost::py_set_zero_points)
    ;
  py::class_<AgreementCost, AgreementCostPtr, py::bases<CostFunc> >("AgreementCost", py::init<const VarField&, const VarField&, const VarField&>())
    .def("set_prev_phi_and_weights", &AgreementCost::py_set_prev_phi_and_weights)
    ;

  py::class_<RigidObservationZeroCrossingCost, RigidObservationZeroCrossingCostPtr, py::bases<CostFunc> >("RigidObservationZeroCrossingCost",
      py::init<const DoubleField, const DoubleField, const Var&, const Var&, const Var&>())
      .def("set_zero_points", &RigidObservationZeroCrossingCost::py_set_zero_points)
    ;
  py::class_<DisplacementCost, DisplacementCostPtr, py::bases<CostFunc> >("DisplacementCost",
      py::init<const Var&, const Var&, const Var&>());


  py::def("make_double_field", &py_make_double_field);
  py::def("apply_rigid_transform", &py_apply_rigid_transform);


  py::def("make_var_field", &py_make_var_field);
  py::def("apply_flow", &py_apply_flow);
  // py::def("march_from_zero_crossing", py_march_from_zero_crossing, py_march_from_zero_crossing_overloads(py::args("phi", "propagate_sign", "ignore_mask"), "docstring"));
  // py::def("compute_flowed_precision", &py_compute_flowed_precision);
}
