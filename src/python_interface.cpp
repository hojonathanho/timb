#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/slice.hpp>
#include <sstream>
#include "common.hpp"

#include "tracking_problem.hpp"
#include "logging.hpp"
#include "numpy_utils.hpp"
#include "optimizer.hpp"
namespace py = boost::python;

struct PyOptResult : public OptResult {
  py::object py_x;
  py::list py_x_over_iters;
  py::object py_cost_detail;
  py::object py_cost_over_iters;
  PyOptResult(OptResultPtr o) {
    status = o->status;
    py_x = util::toNdarray1(o->x.data(), o->x.size());
    for (const VectorXd& x : o->x_over_iters) {
      py_x_over_iters.append(util::toNdarray1(x.data(), x.size()));
    }
    cost = o->cost;
    py_cost_detail = util::toNdarray1(o->cost_detail.data(), o->cost_detail.size());
    py_cost_over_iters = util::toPyList(o->cost_over_iters);
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
    add_vars(util::toVec<string>(py_names), out);
    return util::toPyList(out);
  }
  void py_add_cost_1(CostFuncPtr cost) { add_cost(cost); }
  void py_add_cost_2(CostFuncPtr cost, double coeff) { add_cost(cost, coeff); }
  PyOptResultPtr py_optimize(py::object py_start_x) {
    VectorXd start_x;
    util::from1darray(py_start_x, start_x);
    OptResultPtr result = optimize(start_x);
    return PyOptResultPtr(new PyOptResult(result));
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

#include <boost/multi_array.hpp>
#include <boost/heap/fibonacci_heap.hpp>
static py::object py_march_from_zero_crossing(py::object py_phi, py::object py_ignore_mask) {
  MatrixXd phi;
  util::fromNdarray(py_phi, phi);

  bool use_ignore_mask = py_ignore_mask != py::object();
  Eigen::MatrixXi ignore_mask;
  if (use_ignore_mask) {
    util::fromNdarray(py_ignore_mask, ignore_mask);
  }

  MatrixXd out(phi.rows(), phi.cols());
  out.fill(std::numeric_limits<double>::max());

  struct PointAndDist {
    int i, j;
    double d;
  };
  struct PointAndDistCmp {
    bool operator()(const PointAndDist& a, const PointAndDist& b) const {
      return a.d > b.d;
    }
  };
  typedef boost::heap::fibonacci_heap<PointAndDist, boost::heap::compare<PointAndDistCmp> > Heap;
  Heap heap;

  boost::multi_array<Heap::handle_type, 2> handles(boost::extents[phi.rows()][phi.cols()]);

#define IN_RANGE(I,J) (0 <= (I) && (I) < phi.rows() && 0 <= (J) && (J) < phi.cols())
  // find zero crossing of phi
  for (int i = 0; i < phi.rows(); ++i) {
    for (int j = 0; j < phi.cols(); ++j) {
      if (use_ignore_mask && ignore_mask(i,j)) {
        continue;
      }

      if (phi(i,j) == 0.) {
        out(i,j) = 0.;
      } else {
        std::pair<int,int> neighbors[4] = {
          std::make_pair(i-1, j),
          std::make_pair(i+1, j),
          std::make_pair(i, j-1),
          std::make_pair(i, j+1)
        };
        for (const auto& nbd : neighbors) {
          if (!IN_RANGE(nbd.first, nbd.second)) continue;
          if (use_ignore_mask && ignore_mask(nbd.first, nbd.second)) continue;
          if (phi(nbd.first,nbd.second)*phi(i,j) >= 0) continue;
          double dist_to_zero = phi(i,j) / (phi(i,j) - phi(nbd.first,nbd.second));
          out(i,j) = std::min(out(i,j), dist_to_zero);
        }
      }
    }
  }

  for (int i = 0; i < phi.rows(); ++i) {
    for (int j = 0; j < phi.cols(); ++j) {
      handles[i][j] = heap.push({i, j, out(i,j)});
    }
  }

  while (!heap.empty()) {
    PointAndDist top = heap.top(); heap.pop();
    const int i = top.i, j = top.j;
    std::pair<int,int> neighbors[4] = {
      std::make_pair(i-1, j),
      std::make_pair(i+1, j),
      std::make_pair(i, j-1),
      std::make_pair(i, j+1)
    };
    for (const auto& nbd : neighbors) {
      if (!IN_RANGE(nbd.first, nbd.second)) continue;
      double new_d = out(i,j) + 1.;
      if (new_d < out(nbd.first,nbd.second)) {
        out(nbd.first,nbd.second) = new_d;
        heap.update(handles[nbd.first][nbd.second], {nbd.first,nbd.second,new_d});
      }
    }
  }

#undef IN_RANGE
  return util::toNdarray(out);
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


BOOST_PYTHON_MODULE(ctimbpy) {
  util::LoggingInit();
  util::PythonInit();

  py::class_<Var>("Var")
    .add_property("name", &Var::name)
    .def("__repr__", &Var::name)
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
    .def_readonly("cost_detail", &PyOptResult::py_cost_detail)
    .def_readonly("cost_over_iters", &PyOptResult::py_cost_over_iters)
    .def_readonly("x_over_iters", &PyOptResult::py_x_over_iters)
    ;

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

  py::class_<VarField>("VarField", py::no_init);
  py::def("make_var_field", &py_make_var_field);
  py::def("apply_flow", &py_apply_flow);
  py::def("march_from_zero_crossing", &py_march_from_zero_crossing);

  py::class_<ExampleCost, ExampleCostPtr, py::bases<CostFunc> >("ExampleCost", py::init<const Var&, double, const string&>());
  py::class_<FlowNormCost, FlowNormCostPtr, py::bases<CostFunc> >("FlowNormCost", py::init<const VarField&, const VarField&>());
  py::class_<FlowRigidityCost, FlowRigidityCostPtr, py::bases<CostFunc> >("FlowRigidityCost", py::init<const VarField&, const VarField&>());
  py::class_<ObservationCost, ObservationCostPtr, py::bases<CostFunc> >("ObservationCost", py::init<const VarField&>())
    .def("set_observation", &ObservationCost::py_set_observation)
    ;
  py::class_<ObservationZeroCrossingCost, ObservationZeroCrossingCostPtr, py::bases<CostFunc> >("ObservationZeroCrossingCost", py::init<const VarField&>())
    .def("set_zero_points", &ObservationZeroCrossingCost::py_set_zero_points)
    ;
  py::class_<AgreementCost, AgreementCostPtr, py::bases<CostFunc> >("AgreementCost", py::init<const VarField&, const VarField&, const VarField&>())
    .def("set_prev_phi_and_weights", &AgreementCost::py_set_prev_phi_and_weights)
    ;
}
