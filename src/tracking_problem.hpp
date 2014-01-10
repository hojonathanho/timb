#pragma once

#include "common.hpp"
#include "expr.hpp"
#include "numpy_utils.hpp"
#include "optimizer.hpp"
#include "grid.hpp"
#include "grid_numpy_utils.hpp"
#include <boost/bind.hpp>

void make_field_vars(const string& prefix, Optimizer& opt, VarField& f) {
  vector<string> names;
  names.reserve(f.grid_params().nx * f.grid_params().ny);
  for (int i = 0; i < f.grid_params().nx; ++i) {
    for (int j = 0; j < f.grid_params().ny; ++j) {
      names.push_back((boost::format("%s_%d_%d") % prefix % i % j).str());
    }
  }
  vector<Var> vars;
  opt.add_vars(names, vars);
  int k = 0;
  for (int i = 0; i < f.grid_params().nx; ++i) {
    for (int j = 0; j < f.grid_params().ny; ++j) {
      f(i,j) = vars[k++];
    }
  }
}


struct FlowRigidityCost : public CostFunc {
  const GridParams m_gp;
  const VarField m_u_x, m_u_y;
  DoubleField m_u_x_vals, m_u_y_vals;
  AffExprField m_u_x_x, m_u_x_y, m_u_y_x, m_u_y_y; // derivatives
  DoubleField m_u_x_x_vals, m_u_x_y_vals, m_u_y_x_vals, m_u_y_y_vals; // derivatives

  FlowRigidityCost(const VarField& u_x, const VarField& u_y) :
      m_gp(u_x.grid_params()),
      m_u_x(u_x), m_u_y(u_y),
      m_u_x_vals(m_gp), m_u_y_vals(m_gp),
      m_u_x_x(m_gp), m_u_x_y(m_gp), m_u_y_x(m_gp), m_u_y_y(m_gp),
      m_u_x_x_vals(m_gp), m_u_x_y_vals(m_gp), m_u_y_x_vals(m_gp), m_u_y_y_vals(m_gp) {
    assert(u_y.grid_params() == m_gp);
    gradient(u_x, m_u_x_x, m_u_x_y);
    gradient(u_y, m_u_y_x, m_u_y_y);
  }

  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    extract_field_values(x, m_u_x, m_u_x_vals);
    extract_field_values(x, m_u_y, m_u_y_vals);
    gradient(m_u_x_vals, m_u_x_x_vals, m_u_x_y_vals);
    gradient(m_u_y_vals, m_u_y_x_vals, m_u_y_y_vals);

    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        out(k++) = 2.*m_u_x_x_vals(i,j);
        out(k++) = SQRT2*(m_u_x_y_vals(i,j) + m_u_y_x_vals(i,j));
        out(k++) = 2.*m_u_y_y_vals(i,j);
      }
    }
  }

  void linearize(const VectorXd& x, CostFuncLinearization& lin) {
    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        lin.set_by_expr(k++, 2.*m_u_x_x(i,j));
        lin.set_by_expr(k++, SQRT2*(m_u_x_y(i,j) + m_u_y_x(i,j)));
        lin.set_by_expr(k++, 2.*m_u_y_y(i,j));
      }
    }
  }

  string name() const { return "flow_rigidity"; }
  int num_residuals() const { return 3 * m_gp.nx * m_gp.ny; }
  bool is_linear() const { return true; }
};
typedef boost::shared_ptr<FlowRigidityCost> FlowRigidityCostPtr;

struct FlowNormCost : public CostFunc {
  const GridParams m_gp;
  const VarField m_u_x, m_u_y;

  DoubleField m_u_x_vals, m_u_y_vals;

  FlowNormCost(const VarField& u_x, const VarField& u_y) :
      m_gp(u_x.grid_params()),
      m_u_x(u_x), m_u_y(u_y),
      m_u_x_vals(m_gp), m_u_y_vals(m_gp) {
    assert(u_y.grid_params() == m_gp);
  }

  string name() const { return "flow_norm"; }
  int num_residuals() const { return 2 * m_gp.nx * m_gp.ny; }
  bool is_linear() const { return true; }

  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    extract_field_values(x, m_u_x, m_u_x_vals);
    extract_field_values(x, m_u_y, m_u_y_vals);

    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        out(k++) = m_u_x_vals(i,j);
        out(k++) = m_u_y_vals(i,j);
      }
    }
    assert(k == num_residuals());
  }

  void linearize(const VectorXd& x, CostFuncLinearization& lin) {
    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        lin.set_by_expr(k++, AffExpr(m_u_x(i,j)));
        lin.set_by_expr(k++, AffExpr(m_u_y(i,j)));
      }
    }
    assert(k == num_residuals());
  }
};
typedef boost::shared_ptr<FlowNormCost> FlowNormCostPtr;

struct ObservationCost : public CostFunc {
  const GridParams m_gp;
  const VarField m_phi; // next timestep TSDF variables
  DoubleField m_phi_vals; // current iterate values of TSDF
  DoubleField m_vals, m_weights; // observation values and weights

  ObservationCost(const VarField& phi)
    : m_gp(phi.grid_params()),
      m_phi(phi), m_phi_vals(m_gp),
      m_vals(m_gp), m_weights(m_gp)
  { }

  void set_observation(const DoubleField& vals, const DoubleField& weights) {
    m_vals = vals;
    m_weights = weights;
  }

  void py_set_observation(py::object py_vals, py::object py_weights) {
    from_numpy(py_vals, m_vals);
    from_numpy(py_weights, m_weights);
  }

  string name() const { return "observation"; }
  int num_residuals() const { return m_gp.nx * m_gp.ny; }
  bool is_linear() const { return true; }

  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    extract_field_values(x, m_phi, m_phi_vals);

    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        out(k++) = sqrt(m_weights(i,j)) * (m_phi_vals(i,j) - m_vals(i,j));
      }
    }
    assert(k == num_residuals());
  }

  void linearize(const VectorXd&, CostFuncLinearization& lin) {
    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        lin.set_by_expr(k++, sqrt(m_weights(i,j)) * (m_phi(i,j) - m_vals(i,j)));
      }
    }
    assert(k == num_residuals());
  }
};
typedef boost::shared_ptr<ObservationCost> ObservationCostPtr;

struct AgreementCost : public CostFunc {
  const GridParams m_gp;
  VarField m_phi, m_u_x, m_u_y;

  DoubleField m_prev_phi_vals, m_weights; // user data

  DoubleField m_phi_vals, m_u_x_vals, m_u_y_vals; // temporary memory

  AgreementCost(const VarField& phi, const VarField& u_x, const VarField& u_y)
    : m_gp(phi.grid_params()),
      m_phi(phi), m_u_x(u_x), m_u_y(u_y),
      m_prev_phi_vals(m_gp), m_weights(m_gp),
      m_phi_vals(m_gp), m_u_x_vals(m_gp), m_u_y_vals(m_gp)
  { }

  void set_prev_phi_and_weights(const DoubleField& prev_phi, const DoubleField& weights) {
    m_prev_phi_vals = prev_phi;
    m_weights = weights;
  }

  void py_set_prev_phi_and_weights(py::object py_prev_phi, py::object py_weights) {
    from_numpy(py_prev_phi, m_prev_phi_vals);
    from_numpy(py_weights, m_weights);
  }

  string name() const { return "agreement"; }
  int num_residuals() const { return m_gp.nx * m_gp.ny; }
  bool is_linear() const { return false; }

  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    extract_field_values(x, m_phi, m_phi_vals);
    extract_field_values(x, m_u_x, m_u_x_vals);
    extract_field_values(x, m_u_y, m_u_y_vals);

    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        auto xy = m_gp.to_xy(i, j);
        double flowed_x = xy.first - m_u_x_vals(i,j), flowed_y = xy.second - m_u_y_vals(i,j);
        double flowed_prev_phi_val = m_prev_phi_vals.eval_xy(flowed_x, flowed_y);
        out(k++) = sqrt(m_weights(i,j)) * (m_phi_vals(i,j) - flowed_prev_phi_val);
      }
    }
    assert(k == num_residuals());
  }

  void linearize(const VectorXd& x, CostFuncLinearization& lin) {
    extract_field_values(x, m_phi, m_phi_vals);
    extract_field_values(x, m_u_x, m_u_x_vals);
    extract_field_values(x, m_u_y, m_u_y_vals);

    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        auto xy = m_gp.to_xy(i, j);
        double flowed_x = xy.first - m_u_x_vals(i,j), flowed_y = xy.second - m_u_y_vals(i,j);
        double flowed_prev_phi_val = m_prev_phi_vals.eval_xy(flowed_x, flowed_y);
        auto prev_phi_grad = m_prev_phi_vals.grad_xy(flowed_x, flowed_y);
        AffExpr phi_expr = m_phi(i,j) + prev_phi_grad.x*(m_u_x(i,j) - m_u_x_vals(i,j)) + prev_phi_grad.y*(m_u_y(i,j) - m_u_y_vals(i,j));
        // TODO: negate gradient?
        lin.set_by_expr(k++, sqrt(m_weights(i,j)) * (phi_expr - flowed_prev_phi_val));
      }
    }
    assert(k == num_residuals());
  }
};
typedef boost::shared_ptr<AgreementCost> AgreementCostPtr;

#if 0
struct ObservationCost : public CostFunc {
  const VarField& m_phi, m_u_x, m_u_y;

  // cost parameters
  DoubleField m_vals, m_weights, m_mask;

  // temporary memory for eval() and quadratic()
  DoubleField m_tmp_phi_vals, m_tmp_u_x_vals, m_tmp_u_y_vals;
  AffExprField m_tmp_curr_phi;

  ObservationCost(const VarField& phi, const VarField& u_x, const VarField& u_y)
    : m_phi(phi), m_u_x(u_x), m_u_y(u_y),
      m_vals(phi.grid_params()), m_weights(phi.grid_params()), m_mask(phi.grid_params()),
      m_tmp_phi_vals(phi.grid_params()),
      m_tmp_u_x_vals(phi.grid_params()), m_tmp_u_y_vals(phi.grid_params()),
      m_tmp_curr_phi(phi.grid_params()),
      CostFunc("obs") { }

  void set_from_vals_and_mask(const DoubleField& vals, const DoubleField& weights, const DoubleField& mask) {
    m_vals = vals;
    m_weights = weights;
    m_mask = mask;
  }

  virtual double eval(const VectorXd& x) {
    const GridParams& gp = m_phi.grid_params();
    extract_values(x, m_phi, m_tmp_phi_vals);
    extract_values(x, m_u_x, m_tmp_u_x_vals);
    extract_values(x, m_u_y, m_tmp_u_y_vals);

    // if mask[i] is true, then put a cost on (new_phi[i] - vals[i])^2
    // where new_phi[i] = flowed phi by u
    // TODO: use apply_flow
    double cost = 0.;
    for (int i = 0; i < gp.nx; ++i) {
      for (int j = 0; j < gp.ny; ++j) {
        if (m_mask(i,j) == 0) continue;
        assert(m_weights(i,j) >= 0);
        auto xy = gp.to_xy(i, j);
        double flowed_prev_phi_val = m_tmp_phi_vals.eval_xy(xy.first - m_tmp_u_x_vals(i,j), xy.second - m_tmp_u_y_vals(i,j));
        cost += m_weights(i,j) * square(flowed_prev_phi_val - m_vals(i,j));
      }
    }
    return cost;
  }

  virtual QuadFunctionPtr quadratic(const VectorXd& x) {
    const GridParams& gp = m_phi.grid_params();
    extract_values(x, m_phi, m_tmp_phi_vals);
    extract_values(x, m_u_x, m_tmp_u_x_vals);
    extract_values(x, m_u_y, m_tmp_u_y_vals);

    // m_tmp_curr_phi is a scalar field of expressions
    // that represents the current TSDF with the flow field held fixed
    apply_flow(m_phi, m_tmp_u_x_vals, m_tmp_u_y_vals, m_tmp_curr_phi);

    QuadExpr expr;
    // add on contributions from linearizing wrt u
    for (int i = 0; i < gp.nx; ++i) {
      for (int j = 0; j < gp.ny; ++j) {
        if (m_mask(i,j) == 0) continue;
        if (m_weights(i,j) < 1e-10) continue;
        auto xy = gp.to_xy(i, j);
        double flowed_x = xy.first - m_tmp_u_x_vals(i,j), flowed_y = xy.second - m_tmp_u_y_vals(i,j);
        auto prev_phi_grad = m_tmp_phi_vals.grad_xy(flowed_x, flowed_y);
        AffExpr val = cleanupAff(
          m_tmp_curr_phi(i,j)
          - prev_phi_grad.x*(m_u_x(i,j) - m_tmp_u_x_vals(i,j))
          - prev_phi_grad.y*(m_u_y(i,j) - m_tmp_u_y_vals(i,j))
        );
        exprInc(expr, m_weights(i,j) * exprSquare(val - m_vals(i,j)));
      }
    }

    return QuadFunctionPtr(new QuadFunction(expr));
  }
};
#endif

#if 0
struct PriorCost : public QuadraticCostFunc {
  const VarField& m_phi;
  PriorCost(const VarField& phi) : m_phi(phi), QuadraticCostFunc("prior") { }

  void set_prior(const DoubleField& mean, const DoubleField& omega) {
    const GridParams& p = m_phi.grid_params();
    assert(mean.grid_params() == p && omega.grid_params() == p);

    QuadExpr expr;
    for (int i = 0; i < p.nx; ++i) {
      for (int j = 0; j < p.ny; ++j) {
        exprInc(expr, omega(i,j)*exprSquare(m_phi(i,j) - mean(i,j)));
      }
    }
    init_from_expr(expr);
  }
};


struct TrackingProblemContext {
  const GridParams grid_params;
  VarField phi, u_x, u_y;
  TrackingProblemContext(const GridParams& grid_params_) : grid_params(grid_params_), phi(grid_params_), u_x(grid_params_), u_y(grid_params_) { }
};
typedef boost::shared_ptr<TrackingProblemContext> TrackingProblemContextPtr;

struct TrackingProblemCoeffs {
  double flow_norm;
  double flow_rigidity;
  double observation;
  double prior;
};
typedef boost::shared_ptr<TrackingProblemCoeffs> TrackingProblemCoeffsPtr;

struct TrackingProblemResult {
  DoubleField phi;
  DoubleField u_x, u_y;
  DoubleField next_phi, next_omega;
  OptResultPtr opt_result;
  TrackingProblemResult(const GridParams& gp) : phi(gp), u_x(gp), u_y(gp), next_phi(gp), next_omega(gp) { }
};
typedef boost::shared_ptr<TrackingProblemResult> TrackingProblemResultPtr;

class TrackingProblem {
public:
  TrackingProblem(TrackingProblemContextPtr ctx) { init(ctx); }
  TrackingProblem(double xmin_, double xmax_, double ymin_, double ymax_, int nx_, int ny_) {
    GridParams gp(xmin_, xmax_, ymin_, ymax_, nx_, ny_);
    TrackingProblemContextPtr ctx(new TrackingProblemContext(gp));
    init(ctx);
  }

  TrackingProblemCoeffsPtr m_coeffs; // cost coefficients

  void set_obs(const DoubleField& vals, const DoubleField& weights, const DoubleField& mask) {
    assert(vals.grid_params() == m_ctx->grid_params && weights.grid_params() == m_ctx->grid_params && mask.grid_params() == m_ctx->grid_params);
    ((ObservationCost&) *m_observation_cost).set_from_vals_and_mask(vals, weights, mask);
  }

  void set_prior(const DoubleField& phi_mean, const DoubleField& omega) {
    assert(phi_mean.grid_params() == m_ctx->grid_params && omega.grid_params() == m_ctx->grid_params);
    m_phi_mean.reset(new DoubleField(phi_mean));
    m_omega.reset(new DoubleField(omega));
    ((PriorCost&) *m_prior_cost).set_prior(phi_mean, omega);
  }

  void set_init_u(const DoubleField& u_x, const DoubleField& u_y) {
    assert(u_x.grid_params() == m_ctx->grid_params && u_y.grid_params() == m_ctx->grid_params);
    m_init_u_x.reset(new DoubleField(u_x));
    m_init_u_y.reset(new DoubleField(u_y));
  }

  TrackingProblemResultPtr optimize() {
    set_coeffs();

    VectorXd init_x(VectorXd::Zero(m_opt->num_vars())); // TODO: allocate once only
    int k = 0;
    // initial value for phi is the mean
    for (int i = 0; i < m_ctx->grid_params.nx; ++i) {
      for (int j = 0; j < m_ctx->grid_params.ny; ++j) {
        init_x(k++) = (*m_phi_mean)(i,j);
      }
    }
    // initialize with zero flow field if not provided
    if (m_init_u_x && m_init_u_y) {
      for (int i = 0; i < m_ctx->grid_params.nx; ++i) {
        for (int j = 0; j < m_ctx->grid_params.ny; ++j) {
          init_x(k++) = (*m_init_u_x)(i,j);
        }
      }
      for (int i = 0; i < m_ctx->grid_params.nx; ++i) {
        for (int j = 0; j < m_ctx->grid_params.ny; ++j) {
          init_x(k++) = (*m_init_u_y)(i,j);
        }
      }
    }

    OptResultPtr result = m_opt->optimize(init_x);

    // extract result
    TrackingProblemResultPtr out(new TrackingProblemResult(m_ctx->grid_params));
    out->opt_result = result;
    k = 0;
    for (int i = 0; i < m_ctx->grid_params.nx; ++i) {
      for (int j = 0; j < m_ctx->grid_params.ny; ++j) {
        out->phi(i,j) = result->x(k++);
      }
    }
    for (int i = 0; i < m_ctx->grid_params.nx; ++i) {
      for (int j = 0; j < m_ctx->grid_params.ny; ++j) {
        out->u_x(i,j) = result->x(k++);
      }
    }
    for (int i = 0; i < m_ctx->grid_params.nx; ++i) {
      for (int j = 0; j < m_ctx->grid_params.ny; ++j) {
        out->u_y(i,j) = result->x(k++);
      }
    }
    // compute new phi and omega from result
    apply_flow(out->phi, out->u_x, out->u_y, out->next_phi);
    apply_flow(*m_omega, out->u_x, out->u_y, out->next_omega);

    return out;
  }

protected:
  OptimizerPtr m_opt;
  TrackingProblemContextPtr m_ctx; // stores variables and grid parameters
  // cost objects
  CostFuncPtr m_flow_norm_cost;
  CostFuncPtr m_flow_rigidity_cost;
  CostFuncPtr m_observation_cost;
  CostFuncPtr m_prior_cost;

  // cost parameters and initial guess for optimization
  DoubleFieldPtr m_phi_mean, m_omega;
  DoubleFieldPtr m_init_u_x, m_init_u_y;

  void init(TrackingProblemContextPtr ctx) {
    // default cost coefficients
    m_coeffs.reset(new TrackingProblemCoeffs);
    m_coeffs->flow_norm = 1e-2;
    m_coeffs->flow_rigidity = 1;
    m_coeffs->observation = 1.;
    m_coeffs->prior = 1.;

    // set up optimization problem and variables
    m_ctx = ctx;
    m_opt.reset(new Optimizer);
    // ordering important here
    make_field_vars("phi", *m_opt, m_ctx->phi);
    make_field_vars("u_x", *m_opt, m_ctx->u_x);
    make_field_vars("u_y", *m_opt, m_ctx->u_y);

    // set up cost objects
    m_flow_norm_cost.reset(new FlowNormCost(m_ctx->u_x, m_ctx->u_y));
    m_flow_rigidity_cost.reset(new FlowRigidityCost(m_ctx->u_x, m_ctx->u_y));
    m_observation_cost.reset(new ObservationCost(m_ctx->phi, m_ctx->u_x, m_ctx->u_y));
    m_prior_cost.reset(new PriorCost(m_ctx->phi));
    m_opt->add_cost(m_flow_norm_cost);
    m_opt->add_cost(m_flow_rigidity_cost);
    m_opt->add_cost(m_observation_cost);
    m_opt->add_cost(m_prior_cost);
    set_coeffs();

    // m_opt->add_callback(boost::bind(&TrackingProblem::print_vals, this, _1));
  }

  // copy cost coefficients over into the optimizer
  void set_coeffs() {
    m_opt->set_cost_coeff(m_flow_norm_cost, m_coeffs->flow_norm);
    m_opt->set_cost_coeff(m_flow_rigidity_cost, m_coeffs->flow_rigidity);
    m_opt->set_cost_coeff(m_observation_cost, m_coeffs->observation);
    m_opt->set_cost_coeff(m_prior_cost, m_coeffs->prior);
  }

  void print_vals(const VectorXd& x) {
    DoubleField tmp_phi_vals(m_ctx->grid_params), tmp_u_x_vals(m_ctx->grid_params), tmp_u_y_vals(m_ctx->grid_params);
    extract_values(x, m_ctx->phi, tmp_phi_vals);
    extract_values(x, m_ctx->u_x, tmp_u_x_vals);
    extract_values(x, m_ctx->u_y, tmp_u_y_vals);
    std::cout << "=== phi ===\n" << tmp_phi_vals << '\n';
    std::cout << "=== u_x ===\n" << tmp_u_x_vals << '\n';
    std::cout << "=== u_y ===\n" << tmp_u_y_vals << '\n';
  }
};
typedef boost::shared_ptr<TrackingProblem> TrackingProblemPtr;
#endif