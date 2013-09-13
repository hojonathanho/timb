#include "common.hpp"


template<typename T>
inline T clip(T x, T lo, T hi) {
  return std::max(std::min(x, hi), lo);
}

struct GridParams {
  const double xmin, xmax, ymin, ymax;
  const double eps_x, eps_y;
  const int nx, ny;
  GridParams(double xmin_, double xmax_, double ymin_, double ymax_, int nx_, int ny_)
    : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), nx(nx_), ny(ny_),
      eps_x((xmax_ - xmin_)/(nx_ - 1.)),
      eps_y((ymax_ - ymin_)/(ny_ - 1.))
  { }
};

template<typename ElemT, typename ExprT=ElemT>
class ScalarField {
public:
  ScalarField(const GridParams& grid_params) :
    m_grid_params(grid_params),
    m_data(grid_params.nx * grid_params.ny, ElemT(0))
  { }

  struct Vector { ExprT x, y; };

  std::pair<double, double> to_xy(double i, double j) const {
    return std::make_pair(m_grid_params.xmin + i*m_grid_params.eps_x, m_grid_params.ymin + j*m_grid_params.eps_y);
  }

  std::pair<double, double> to_ij(double x, double y) const {
    return std::make_pair((x - m_grid_params.xmin)/m_grid_params.eps_x, (y - m_grid_params.ymin)/m_grid_params.eps_y);
  }

  ElemT& operator[](int i, int j) {
    assert(0 <= i && i < m_grid_params.nx && 0 <= j && j < m_grid_params.ny);
    return m_data[i*m_grid_params.ny + j];
  }

  ExprT eval_ij(double i, double j) const {
    // bilinear interpolation
    int ax = (int) floor(i), ay = (int) floor(j);
    int bx = ax + 1, by = ay + 1;
    ax = clip(ax, 0, m_grid_params.nx - 1);
    bx = clip(bx, 0, m_grid_params.nx - 1);
    ay = clip(ay, 0, m_grid_params.ny - 1);
    by = clip(by, 0, m_grid_params.ny - 1);
    double dx = std::max(i - ax, 0.), dy = std::max(j - ay, 0.);
    return (1.-dy)*((1.-dx)*(*this)[ax,ay] + dx*(*this)[bx,ay]) + dy*((1.-dx)*(*this)[ax,by] + dx*(*this)[bx,by]);
  }

  ExprT eval_xy(double x, double y) const {
    std::pair<double, double> p = to_ij(x, y);
    return eval_ij(p.first, p.second);
  }

  Vector grad_ij(double i, double j) const {
    static const double delta = 1e-5;
    Vector g = {
      (eval_ij(i+delta, j) - eval_ij(i-delta, j)) / (2.*delta),
      (eval_ij(i, j+delta) - eval_ij(i, j-delta)) / (2.*delta)
    };
    return g;
  }

  Vector grad_xy(double x, double y) const {
    std::pair<double, double> p = to_ij(x, y);
    return grad_ij(p.first, p.second);
  }

  ElemT* data() { return m_data.data(); }
  const GridParams& grid_params() const { return m_grid_params; }

private:
  const GridParams m_grid_params;
  std::vector<ElemT> m_data;
};

typedef ScalarField<Var, AffExpr> VarField;
typedef ScalarField<AffExpr, AffExpr> AffExprField;
typedef ScalarField<double, double> DoubleField;

// one-sided finite differences
template<typename T, S>
void gradient(const ScalarField<T, S>& f, ScalarField<S, S>& g_x, ScalarField<S, S>& g_y) {
  const GridParams& p = f.grid_params();
  assert(p == g_x.grid_params() && p == g_y.grid_params());

  // derivatives in x direction
  for (int i = 0; i < p.nx - 1; ++i) {
    for (int j = 0; j < p.ny; ++j) {
      g_x[i,j] = (f[i+1,j] - f[i,j]) / p.eps_x;
    }
  }
  // copy over last row
  for (int j = 0; j < p.ny; ++j) {
    g_x[p.nx-1, j] = g_x[p.nx-2, j];
  }

  // derivatives in y direction
  for (int i = 0; i < p.nx; ++i) {
    for (int j = 0; j < p.ny - 1; ++j) {
      g_y[i,j] = (f[i,j+1] - f[i,j]) / p.eps_y;
    }
    // copy over last column
    g_y[i, p.ny-1] = g_y[i, p.ny-2];
  }
}


struct TrackingProblemContext {
  const GridParams grid_params;
  VarField phi, u_x, u_y;
  TrackingProblemContext(const GridParams& grid_params_) : grid_params(grid_params_), phi(grid_params_), u_x(grid_params_), u_y(grid_params_) { }
};
typedef boost::shared_ptr<TrackingProblemContext> TrackingProblemContextPtr;



struct FlowRigidityCost : public QuadraticCostFunc {
  FlowRigidityCost(const VarField& u_x, const VarField& u_y) : QuadraticCostFunc("rigidity") {
    const GridParams& p = u_x.params();
    AffExprField u_x_x(p), u_x_y(p), u_y_x(p), u_y_y(p);
    gradient(u_x, u_x_x, u_x_y);
    gradient(u_y, u_y_x, u_y_y);
    QuadExpr expr(0);
    for (int i = 0; i < p.nx; ++i) {
      for (int j = 0; j < p.ny; ++j) {
        // J + J^T
        exprInc(expr, exprSquare(2.*u_x_x[i,j]));
        exprInc(expr, 2*exprSquare(u_x_y[i,j] + u_y_x[i,j]));
        exprInc(expr, exprSquare(2.*u_y_y[i,j]));
      }
    }
    init_from_expr(expr);
  }
};

struct FlowNormCost : public QuadraticCostFunc {
  FlowNormCost(const VarField& u_x, const VarField& u_y) : QuadraticCostFunc("flow_norm") {
    const GridParams& p = u_x.params();
    QuadExpr expr;
    for (int i = 0; i < p.nx; ++i) {
      for (int j = 0; j < p.ny; ++j) {
        exprInc(expr, exprSquare(u_x[i,j]));
        exprInc(expr, exprSquare(u_y[i,j]));
      }
    }
    init_from_expr(expr);
  }
};

struct ObservationCost : public QuadraticCostFunc {
  const VarField& m_phi;
  ObservationCost(const VarField& phi) : m_phi(phi) { }

  void set_obs_pts(const MatrixX2d& pts) {
    for (int i = 0; i < pts.rows(); ++i) {
      AffExpr e = m_phi.eval_xy(pts(i,0), pts(i,1));
    }
  }
};

struct FlowAgreementCost : public CostFunc {
};


struct TrackingProblemCoeffs {
  double flow_norm;
  double flow_rigidity;
  double observation;
  double phi_agreement;
};
typedef boost::shared_ptr<TrackingProblemCoeffs> TrackingProblemCoeffsPtr;

class TrackingProblem {
public:
  TrackingProblem(TrackingProblemContextPtr ctx) { init(ctx); }
  TrackingProblem(double xmin_, double xmax_, double ymin_, double ymax_, int nx_, int ny_) {
    GridParams gp(xmin_, xmax_, ymin_, ymax_, nx_, ny_);
    TrackingProblemContextPtr ctx(new TrackingProblemContext(gp));
    init(ctx);
  }

  TrackingProblemCoeffsPtr coeffs() const { return m_coeffs; }

  void set_observation_points(const MatrixX2d& pts) {
    m_obs_pts = pts;
  }

  void set_prev_phi() {

  }

  void optimize() {
    set_coeffs();

  }

private:
  TrackingProblemContextPtr m_ctx;
  TrackingProblemCoeffsPtr m_coeffs;

  // current problem data
  MatrixX2d m_obs_pts;
  DoubleField m_blah;

  // cost objects
  CostFuncPtr m_flow_norm_cost;
  CostFuncPtr m_flow_rigidity_cost;
  CostFuncPtr m_observation_cost;
  CostFuncPtr m_phi_agreement_cost;

  void init(TrackingProblemContextPtr ctx) {
    m_ctx = ctx;

    // default cost coefficients
    m_coeffs.reset(new TrackingProblemCoeffs);
    m_coeffs->flow_norm = 1e-9;
    m_coeffs->flow_rigidity = 1e-3;
    m_coeffs->observation = 1.;
    m_coeffs->phi_agreement = 1.

    // set up cost objects
    m_flow_norm_cost.reset(new FlowNormCost(m_ctx->u_x, m_ctx->u_y));
    m_flow_rigidity_cost.reset(new FlowRigidityCost(m_ctx->u_x, m_ctx->u_y));
    m_observation_cost.reset(new ObservationCost(m_ctx->phi));
    m_phi_agreement_cost.reset(new PhiAgreementCost(m_ctx->phi, m_ctx->u_x, m_ctx->u_y));
    m_opt->add_cost(m_flow_norm_cost);
    m_opt->add_cost(m_flow_rigidity_cost);
    m_opt->add_cost(m_observation_cost);
    m_opt->add_cost(m_phi_agreement_cost);
    set_coeffs();
  }

  // copy cost coefficients over into the optimizer
  void set_coeffs() {
    m_opt->set_cost_coeff(m_flow_norm_cost, m_coeffs->flow_norm);
    m_opt->set_cost_coeff(m_flow_rigidity_cost, m_coeffs->flow_rigidity);
    m_opt->set_cost_coeff(m_observation_cost, m_coeffs->observation);
    m_opt->set_cost_coeff(m_phi_agreement_cost, m_coeffs->phi_agreement);
  }
};
typedef boost::shared_ptr<TrackingProblem> TrackingProblemPtr;
