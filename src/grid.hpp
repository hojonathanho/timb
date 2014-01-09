#pragma once

#include "common.hpp"
#include <vector>

struct GridParams {
  const double xmin, xmax, ymin, ymax; // grid limits (world coordinates)
  const int nx, ny; // number of grid points in each direction
  const double eps_x, eps_y; // distance between points
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
inline bool operator==(const GridParams& a, const GridParams& b) {
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

  // TODO: analytical
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

template<typename Derived>
void from_eigen(const Eigen::DenseBase<Derived> &in, DoubleField& out) {
  assert(out.grid_params().nx == in.rows() && out.grid_params().ny == in.cols());
  for (int i = 0; i < in.rows(); ++i) {
    for (int j = 0; j < in.cols(); ++j) {
      out(i,j) = in(i,j);
    }
  }
}

typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > RowMajorConstDblMapT;
RowMajorConstDblMapT to_eigen(const DoubleField& f) {
  return RowMajorConstDblMapT(f.data(), f.grid_params().nx, f.grid_params().ny);
}

void extract_field_values(const VectorXd& x, const VarField& vars, DoubleField& out) {
  assert(out.grid_params() == vars.grid_params());
  for (int i = 0; i < vars.grid_params().nx; ++i) {
    for (int j = 0; j < vars.grid_params().ny; ++j) {
      out(i,j) = vars(i,j).value(x);
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
