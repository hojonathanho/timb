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

struct GradientCost : public CostFunc {
  const GridParams m_gp;
  VarField m_phi;

  AffExprField m_phi_x, m_phi_y;
  DoubleField m_phi_vals, m_phi_x_vals, m_phi_y_vals;

  GradientCost(const VarField& phi)
    : m_gp(phi.grid_params()),
      m_phi(phi),
      m_phi_x(m_gp), m_phi_y(m_gp),
      m_phi_vals(m_gp), m_phi_x_vals(m_gp), m_phi_y_vals(m_gp)
  {
    gradient(m_phi, m_phi_x, m_phi_y);
  }

  string name() const { return "grad"; }
  int num_residuals() const { return 2 * m_gp.nx * m_gp.ny; }
  bool is_linear() const { return true; }

  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    extract_field_values(x, m_phi, m_phi_vals);
    gradient(m_phi_vals, m_phi_x_vals, m_phi_y_vals);

    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        out(k++) = m_phi_x_vals(i,j);
        out(k++) = m_phi_y_vals(i,j);
      }
    }
    assert(k == num_residuals());
  }

  void linearize(const VectorXd& x, CostFuncLinearization& lin) {
    int k = 0;
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        lin.set_by_expr(k++, m_phi_x(i,j));
        lin.set_by_expr(k++, m_phi_y(i,j));
      }
    }
    assert(k == num_residuals());
  }
};
typedef boost::shared_ptr<GradientCost> GradientCostPtr;

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

struct ObservationZeroCrossingCost : public CostFunc {
  const GridParams m_gp;
  const VarField m_phi; // next timestep TSDF variables
  DoubleField m_phi_vals; // current iterate values of TSDF
  MatrixX2d m_zero_points; // points where phi should be zero

  ObservationZeroCrossingCost(const VarField& phi)
    : m_gp(phi.grid_params()),
      m_phi(phi), m_phi_vals(m_gp)
  { }

  void set_zero_points(const MatrixX2d& zero_points) {
    m_zero_points = zero_points;
  }

  void py_set_zero_points(py::object py_zero_points) {
    util::fromNdarray(py_zero_points, m_zero_points);
  }

  string name() const { return "obs_zc"; }
  int num_residuals() const { return m_zero_points.rows(); }
  bool is_linear() const { return true; }

  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    assert(m_zero_points.rows() > 0);
    extract_field_values(x, m_phi, m_phi_vals);
    for (int i = 0; i < m_zero_points.rows(); ++i) {
      out(i) = m_phi_vals.eval_xy(m_zero_points(i,0), m_zero_points(i,1));
    }
  }

  void linearize(const VectorXd&, CostFuncLinearization& lin) {
    assert(m_zero_points.rows() > 0);
    for (int i = 0; i < m_zero_points.rows(); ++i) {
      lin.set_by_expr(i, m_phi.eval_xy(m_zero_points(i,0), m_zero_points(i,1)));
    }
  }
};
typedef boost::shared_ptr<ObservationZeroCrossingCost> ObservationZeroCrossingCostPtr;

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
        lin.set_by_expr(k++, sqrt(m_weights(i,j)) * (phi_expr - flowed_prev_phi_val));
      }
    }
    assert(k == num_residuals());
  }
};
typedef boost::shared_ptr<AgreementCost> AgreementCostPtr;
