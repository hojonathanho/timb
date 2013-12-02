#pragma once

#include "common.hpp"
#include "expr.hpp"
#include "numpy_utils.hpp"
#include "problem.hpp"

template<typename T>
inline T clip(T x, T lo, T hi) {
  return std::max(std::min(x, hi), lo);
}

template<typename T>
inline T square(const T& x) { return x*x; }

struct GridParams {
  const double xmin, xmax, ymin, ymax;
  const double eps_x, eps_y;
  const int nx, ny;
  GridParams(double xmin_, double xmax_, double ymin_, double ymax_, int nx_, int ny_)
    : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), nx(nx_), ny(ny_),
      eps_x((xmax_ - xmin_)/(nx_ - 1.)),
      eps_y((ymax_ - ymin_)/(ny_ - 1.))
  { }
  std::pair<double, double> to_xy(double i, double j) const {
    return std::make_pair(xmin + i*eps_x, ymin + j*eps_y);
  }
  std::pair<double, double> to_ij(double x, double y) const {
    return std::make_pair((x - xmin)/eps_x, (y - ymin)/eps_y);
  }
};
bool operator==(const GridParams& a, const GridParams& b) {
  return
    a.xmin == b.xmin &&
    a.xmax == b.xmax &&
    a.ymin == b.ymin &&
    a.ymax == b.ymax &&
    a.nx == b.nx &&
    a.ny == b.ny;
}

template<typename ElemT, typename ExprT=ElemT>
class ScalarField {
public:
  explicit ScalarField(const GridParams& grid_params) :
    m_grid_params(grid_params),
    m_data(grid_params.nx * grid_params.ny, ElemT(0))
  { }
  ScalarField(const ScalarField& other) : m_grid_params(other.m_grid_params), m_data(other.m_data) { }

  void operator=(const ScalarField& other) {
    assert(m_grid_params == other.m_grid_params);
    m_data = other.m_data;
  }

  typedef boost::shared_ptr<ScalarField<ElemT, ExprT> > Ptr;
  struct ExprVec { ExprT x, y; };

  const ElemT& get(int i, int j) const {
    assert(0 <= i && i < m_grid_params.nx && 0 <= j && j < m_grid_params.ny);
    return m_data[i*m_grid_params.ny + j];
  }
  ElemT& get(int i, int j) {
    assert(0 <= i && i < m_grid_params.nx && 0 <= j && j < m_grid_params.ny);
    return m_data[i*m_grid_params.ny + j];
  }
  const ElemT& operator()(int i, int j) const { return get(i,j); }
  ElemT& operator()(int i, int j) { return get(i,j); }

  ExprT eval_ij(double i, double j) const {
    // bilinear interpolation
    int ax = (int) floor(i), ay = (int) floor(j);
    int bx = ax + 1, by = ay + 1;
    ax = clip(ax, 0, m_grid_params.nx - 1);
    bx = clip(bx, 0, m_grid_params.nx - 1);
    ay = clip(ay, 0, m_grid_params.ny - 1);
    by = clip(by, 0, m_grid_params.ny - 1);
    double dx = std::max(i - ax, 0.), dy = std::max(j - ay, 0.);
    return (1.-dy)*((1.-dx)*get(ax,ay) + dx*get(bx,ay)) + dy*((1.-dx)*get(ax,by) + dx*get(bx,by));
  }

  ExprT eval_xy(double x, double y) const {
    std::pair<double, double> p = m_grid_params.to_ij(x, y);
    return eval_ij(p.first, p.second);
  }

  ExprVec grad_ij(double i, double j) const {
    static const double delta = 1e-5;
    ExprVec g = {
      (eval_ij(i+delta, j) - eval_ij(i-delta, j)) / (2.*delta),
      (eval_ij(i, j+delta) - eval_ij(i, j-delta)) / (2.*delta)
    };
    return g;
  }

  ExprVec grad_xy(double x, double y) const {
    std::pair<double, double> p = m_grid_params.to_ij(x, y);
    return grad_ij(p.first, p.second);
  }

  ElemT* data() { return m_data.data(); }
  const ElemT* data() const { return m_data.data(); }
  const GridParams& grid_params() const { return m_grid_params; }

private:
  const GridParams m_grid_params;
  std::vector<ElemT> m_data;
};

typedef ScalarField<Var, AffExpr> VarField;
typedef ScalarField<AffExpr, AffExpr> AffExprField;
typedef ScalarField<double, double> DoubleField;
typedef boost::shared_ptr<DoubleField> DoubleFieldPtr;


template<typename T, typename S>
std::ostream& operator<<(std::ostream& o, const ScalarField<T, S>& f) {
  for (int i = 0; i < f.grid_params().nx; ++i) {
    for (int j = 0; j < f.grid_params().ny; ++j) {
      o << f(i,j) << ' ';
    }
    o << '\n';
  }
  return o;
}

void extract_values(const VectorXd& x, const VarField& vars, DoubleField& out) {
  assert(out.grid_params() == vars.grid_params());
  for (int i = 0; i < vars.grid_params().nx; ++i) {
    for (int j = 0; j < vars.grid_params().ny; ++j) {
      out(i,j) = vars(i,j).value(x);
    }
  }
}

typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > RowMajorConstDblMapT;
RowMajorConstDblMapT to_eigen(const DoubleField& f) {
  return RowMajorConstDblMapT(f.data(), f.grid_params().nx, f.grid_params().ny);
}

template<typename Derived>
void from_eigen(const Eigen::DenseBase<Derived> &in, DoubleField& out) {
  assert(out.grid_params().nx == in.rows() && out.grid_params().ny == in.cols());
  for (int i = 0; i < in.rows(); ++i) {
    for (int j = 0; j < in.cols(); ++j) {
      out(i,j) = in(i,j);
    }
  }
}

// one-sided finite differences
template<typename T, typename S>
void gradient(const ScalarField<T, S>& f, ScalarField<S, S>& g_x, ScalarField<S, S>& g_y) {
  const GridParams& p = f.grid_params();
  assert(p == g_x.grid_params() && p == g_y.grid_params());

  // derivatives in x direction
  for (int i = 0; i < p.nx - 1; ++i) {
    for (int j = 0; j < p.ny; ++j) {
      g_x(i,j) = (f(i+1,j) - f(i,j)) * (1./p.eps_x);
    }
  }
  // copy over last row
  for (int j = 0; j < p.ny; ++j) {
    g_x(p.nx-1,j) = g_x(p.nx-2,j);
  }

  // derivatives in y direction
  for (int i = 0; i < p.nx; ++i) {
    for (int j = 0; j < p.ny - 1; ++j) {
      g_y(i,j) = (f(i,j+1) - f(i,j)) * (1./p.eps_y);
    }
    // copy over last column
    g_y(i,p.ny-1) = g_y(i,p.ny-2);
  }
}

void make_field_vars(const string& prefix, Optimizer& opt, VarField& f) {
  vector<string> names;
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

template<typename ElemT, typename ExprT>
void apply_flow(const ScalarField<ElemT, ExprT>& phi, const DoubleField& u_x, const DoubleField& u_y, ScalarField<ExprT, ExprT>& out) {
  const GridParams& gp = phi.grid_params();
  assert(u_x.grid_params() == gp && u_y.grid_params() == gp && out.grid_params() == gp);

  for (int i = 0; i < gp.nx; ++i) {
    for (int j = 0; j < gp.ny; ++j) {
      auto xy = gp.to_xy(i, j);
      out(i,j) = phi.eval_xy(xy.first - u_x(i,j), xy.second - u_y(i,j));
    }
  }
}


struct FlowRigidityCost : public QuadraticCostFunc {
  FlowRigidityCost(const VarField& u_x, const VarField& u_y) : QuadraticCostFunc("rigidity") {
    const GridParams& p = u_x.grid_params();
    AffExprField u_x_x(p), u_x_y(p), u_y_x(p), u_y_y(p);
    gradient(u_x, u_x_x, u_x_y);
    gradient(u_y, u_y_x, u_y_y);
    QuadExpr expr(0);
    for (int i = 0; i < p.nx; ++i) {
      for (int j = 0; j < p.ny; ++j) {
        // ||J + J^T||^2
        exprInc(expr, exprSquare(2.*u_x_x(i,j)));
        exprInc(expr, 2*exprSquare(u_x_y(i,j) + u_y_x(i,j)));
        exprInc(expr, exprSquare(2.*u_y_y(i,j)));
      }
    }
    init_from_expr(expr);
  }
};

struct FlowNormCost : public QuadraticCostFunc {
  FlowNormCost(const VarField& u_x, const VarField& u_y) : QuadraticCostFunc("flow_norm") {
    const GridParams& p = u_x.grid_params();
    QuadExpr expr;
    for (int i = 0; i < p.nx; ++i) {
      for (int j = 0; j < p.ny; ++j) {
        exprInc(expr, exprSquare(u_x(i,j)));
        exprInc(expr, exprSquare(u_y(i,j)));
      }
    }
    init_from_expr(expr);
  }
};

struct ObservationCost : public CostFunc {
  const VarField& m_phi, m_u_x, m_u_y;

  // cost parameters
  DoubleField m_vals, m_mask;

  // temporary memory for eval() and quadratic()
  DoubleField m_tmp_phi_vals, m_tmp_u_x_vals, m_tmp_u_y_vals;
  AffExprField m_tmp_curr_phi;

  ObservationCost(const VarField& phi, const VarField& u_x, const VarField& u_y)
    : m_phi(phi), m_u_x(u_x), m_u_y(u_y),
      m_vals(phi.grid_params()), m_mask(phi.grid_params()),
      m_tmp_phi_vals(phi.grid_params()),
      m_tmp_u_x_vals(phi.grid_params()), m_tmp_u_y_vals(phi.grid_params()),
      m_tmp_curr_phi(phi.grid_params()),
      CostFunc("obs") { }

  void set_from_vals_and_mask(const DoubleField& vals, const DoubleField& mask) {
    m_vals = vals;
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
        auto xy = gp.to_xy(i, j);
        double flowed_prev_phi_val = m_tmp_phi_vals.eval_xy(xy.first - m_tmp_u_x_vals(i,j), xy.second - m_tmp_u_y_vals(i,j));
        cost += square(flowed_prev_phi_val - m_vals(i,j));
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
        auto xy = gp.to_xy(i, j);
        double flowed_x = xy.first - m_tmp_u_x_vals(i,j), flowed_y = xy.second - m_tmp_u_y_vals(i,j);
        auto prev_phi_grad = m_tmp_phi_vals.grad_xy(flowed_x, flowed_y);
        AffExpr val = m_tmp_curr_phi(i,j) - prev_phi_grad.x*(m_u_x(i,j) - m_tmp_u_x_vals(i,j)) - prev_phi_grad.y*(m_u_y(i,j) - m_tmp_u_y_vals(i,j));
        exprInc(expr, exprSquare(val - m_vals(i,j)));
      }
    }

    return QuadFunctionPtr(new QuadFunction(expr));
  }
};

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

  void set_obs(const DoubleField& vals, const DoubleField& mask) {
    assert(vals.grid_params() == m_ctx->grid_params && mask.grid_params() == m_ctx->grid_params);
    ((ObservationCost&) *m_observation_cost).set_from_vals_and_mask(vals, mask);
  }

  void set_prior(const DoubleField& phi_mean, const DoubleField& omega) {
    assert(phi_mean.grid_params() == m_ctx->grid_params && omega.grid_params() == m_ctx->grid_params);
    m_phi_mean.reset(new DoubleField(phi_mean));
    m_omega.reset(new DoubleField(omega));
    ((PriorCost&) *m_prior_cost).set_prior(phi_mean, omega);
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
    // initialize with zero flow field

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

  void init(TrackingProblemContextPtr ctx) {
    // default cost coefficients
    m_coeffs.reset(new TrackingProblemCoeffs);
    m_coeffs->flow_norm = 1e-9;
    m_coeffs->flow_rigidity = 1e-3;
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
  }

  // copy cost coefficients over into the optimizer
  void set_coeffs() {
    m_opt->set_cost_coeff(m_flow_norm_cost, m_coeffs->flow_norm);
    m_opt->set_cost_coeff(m_flow_rigidity_cost, m_coeffs->flow_rigidity);
    m_opt->set_cost_coeff(m_observation_cost, m_coeffs->observation);
    m_opt->set_cost_coeff(m_prior_cost, m_coeffs->prior);
  }
};
typedef boost::shared_ptr<TrackingProblem> TrackingProblemPtr;
