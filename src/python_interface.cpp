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
    return optresult_to_dict(optimize(start_x));
  }

  void py_add_intermediate_callback(py::object fn) {
    add_intermediate_callback(
      [fn](const VectorXd& prev_x, const VectorXd& x,
          double true_old_cost, double true_improvement,
          double model_improvement, double ratio) {
        fn(util::toNdarray(prev_x), util::toNdarray(x), true_old_cost, true_improvement, model_improvement, ratio);
      }
    );
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
static py::object py_apply_flow_to_weights(const GridParams& gp, py::object py_phi, py::object py_u_x, py::object py_u_y) {
  DoubleField phi(gp), u_x(gp), u_y(gp), flowed_phi(gp);
  from_numpy(py_phi, phi);
  from_numpy(py_u_x, u_x);
  from_numpy(py_u_y, u_y);
  apply_flow_to_weights(phi, u_x, u_y, flowed_phi);
  return to_numpy(flowed_phi);
}

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

double py_timb_problem_eval_objective(
  const GridParams& gp,

  py::object py_phi,
  py::object py_u,
  py::object py_v,

  // observation values and weights
  py::object py_z,
  py::object py_w_z,

  // prev phi
  py::object py_prev_phi,
  // flowed weights
  py::object py_wtilde,

  double alpha, // strain cost coeff
  double beta // norm cost coeff
) {
  DoubleField phi(gp), u(gp), v(gp), z(gp), w_z(gp), prev_phi(gp), wtilde(gp);
  from_numpy(py_phi, phi);
  from_numpy(py_u, u);
  from_numpy(py_v, v);
  from_numpy(py_z, z);
  from_numpy(py_w_z, w_z);
  from_numpy(py_prev_phi, prev_phi);
  from_numpy(py_wtilde, wtilde);
  return timb_problem_eval_objective(phi, u, v, z, w_z, prev_phi, wtilde, alpha, beta);
}

double py_timb_problem_eval_model_objective(
  const GridParams& gp,

  py::object py_phi,
  py::object py_u,
  py::object py_v,

  // observation values and weights
  py::object py_z,
  py::object py_w_z,

  // linearized flow
  py::object py_mu_0,
  py::object py_mu_u,
  py::object py_mu_v,
  // flowed weights
  py::object py_wtilde,

  double alpha, // strain cost coeff
  double beta // norm cost coeff
) {
  DoubleField phi(gp), u(gp), v(gp), z(gp), w_z(gp), mu_0(gp), mu_u(gp), mu_v(gp), wtilde(gp);
  from_numpy(py_phi, phi);
  from_numpy(py_u, u);
  from_numpy(py_v, v);
  from_numpy(py_z, z);
  from_numpy(py_w_z, w_z);
  from_numpy(py_mu_0, mu_0);
  from_numpy(py_mu_u, mu_u);
  from_numpy(py_mu_v, mu_v);
  from_numpy(py_wtilde, wtilde);
  return timb_problem_eval_model_objective(phi, u, v, z, w_z, mu_0, mu_u, mu_v, wtilde, alpha, beta);
}

py::object py_timb_linearize_flowed_prev_phi(const GridParams& gp, py::object py_prev_phi, py::object py_u, py::object py_v) {
  DoubleField prev_phi(gp), u(gp), v(gp);
  from_numpy(py_prev_phi, prev_phi);
  from_numpy(py_u, u);
  from_numpy(py_v, v);

  DoubleField out_0(gp), out_u(gp), out_v(gp);
  timb_linearize_flowed_prev_phi(prev_phi, u, v, out_0, out_u, out_v);
  return py::make_tuple(to_numpy(out_0), to_numpy(out_u), to_numpy(out_v));
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

  ////////// Levenberg-Marquardt optimizer //////////

  py::class_<Var>("Var")
    .add_property("name", &Var::name)
    .def("__repr__", &Var::name)
    ;

  py::enum_<OptStatus>("OptStatus")
    .value("OPT_INCOMPLETE", OPT_INCOMPLETE)
    .value("OPT_CONVERGED", OPT_CONVERGED)
    .value("OPT_ITER_LIMIT", OPT_ITER_LIMIT)
    ;

  py::class_<OptParams>("OptParams", py::no_init)
    .def_readwrite("init_damping", &OptParams::init_damping)
    .def_readwrite("init_damping_increase_factor", &OptParams::init_damping_increase_factor)
    .def_readwrite("min_scaling", &OptParams::min_scaling)
    .def_readwrite("grad_convergence_tol", &OptParams::grad_convergence_tol)
    .def_readwrite("approx_improve_rel_tol", &OptParams::approx_improve_rel_tol)
    .def_readwrite("max_iter", &OptParams::max_iter)
    .def_readwrite("enable_var_scaling", &OptParams::enable_var_scaling)
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
    .def("add_intermediate_callback", &PyOptimizer::py_add_intermediate_callback)
    .def("num_vars", &PyOptimizer::num_vars)
    .def("optimize", &PyOptimizer::py_optimize)
    ;

  py::class_<CostFunc, CostFuncPtr, boost::noncopyable>("CostFunc", py::no_init)
    // Don't expose anything. Only subclasses should be used from Python
    ;


  ////////// Structures and costs for tracking optimization problem //////////

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

  ////////// Utilities -- tracking problem //////////

  py::def("make_var_field", &py_make_var_field);
  py::def("apply_flow", &py_apply_flow);
  py::def("apply_flow_to_weights", &py_apply_flow_to_weights);

  ////////// Utilities -- rigid problem (KinectFusion) //////////

  py::def("make_double_field", &py_make_double_field);
  py::def("apply_rigid_transform", &py_apply_rigid_transform);


  ////////// Tracking optimization problem hardcoded implementation //////////
  py::def("timb_problem_eval_objective", &py_timb_problem_eval_objective);
  py::def("timb_problem_eval_model_objective", &py_timb_problem_eval_model_objective);
  py::def("timb_linearize_flowed_prev_phi", &py_timb_linearize_flowed_prev_phi);
}
