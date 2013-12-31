#pragma once

#include "common.hpp"
#include "expr.hpp"
#include <boost/function.hpp>

#if 0
class QuadFunction {
public:
  typedef Eigen::SparseMatrix<double> SparseMatrixT;
  typedef Eigen::SparseSelfAdjointView<SparseMatrixT, Eigen::Lower> SparseSelfAdjointViewT;

  QuadFunction(const QuadExpr& expr) : m_quad_expr(expr), m_initialized(false) { }

  void init_with_num_vars(int n) { if (!m_initialized) init_from_expr(n, m_quad_expr); }

  const SparseSelfAdjointViewT A() const { assert(m_initialized); return m_A.selfadjointView<Eigen::Lower>(); }
  const SparseMatrixT& A_lower() const { assert(m_initialized); return m_A; } // warning: only lower triangle filled out!
  const Eigen::VectorXd& b() const { assert(m_initialized); return m_b; }
  double c() const { assert(m_initialized); return m_c; }

  double value(const Eigen::VectorXd& x) const {
    // if sparse matrices filled, out, use those
    // otherwise just call the quad expr's evaluate method
    if (m_initialized) {
      assert(x.size() == m_b.size());
      return .5*x.dot(A()*x) + m_b.dot(x) + m_c;
    }
    return m_quad_expr.value(x);
  }

  const QuadExpr& expr() const { return m_quad_expr; }

  void add_to(SparseMatrixT& other_A_lower, VectorXd& other_b, double& other_c, double coeff=1.) const {
    other_A_lower += coeff * A_lower();
    other_b += coeff * b();
    other_c += coeff * c();
  }

private:
  // 1/2 x^T A x + b^T x + c
  SparseMatrixT m_A; // lower triangle only
  Eigen::VectorXd m_b;
  double m_c;

  bool m_initialized; // true if sparse data was filled out
  const QuadExpr m_quad_expr;

  void init_from_expr(int num_vars, const QuadExpr &expr) {
    // quadratic part
    m_A.resize(num_vars, num_vars);
    m_A.setZero();
    typedef Eigen::Triplet<double> T;
    vector<T> triplets;
    triplets.reserve(expr.size());
    for (int i = 0; i < expr.size(); ++i) {
      int ind1 = expr.vars1[i].rep->index, ind2 = expr.vars2[i].rep->index;
      // only fill in lower triangle
      if (ind1 < ind2) { std::swap(ind1, ind2); }
      double coeff = expr.coeffs[i];
      if (ind1 == ind2) { coeff *= 2; }
      triplets.push_back(T(ind1, ind2, coeff));
    }
    m_A.setFromTriplets(triplets.begin(), triplets.end());

    // affine part
    m_b.resize(num_vars);
    m_b.setZero();
    for (int i = 0; i < expr.affexpr.size(); ++i) {
      m_b(expr.affexpr.vars[i].rep->index) += expr.affexpr.coeffs[i];
    }

    m_c = expr.affexpr.constant;

    m_initialized = true;
  }
};
typedef boost::shared_ptr<QuadFunction> QuadFunctionPtr;
#endif

class JacobianContainer {
public:
  typedef Eigen::Triplet<double> TripletT;

  JacobianContainer(vector<TripletT>& triplets, int row_offset, double weight) : m_triplets(triplets), m_row_offset(row_offset), m_weight(weight) { }

  void set_by_expr(int i, const AffExpr &e) {
    // assert(0 <= i && i < m_num_residuals);
    for (int j = 0; j < e.size(); ++j) {
      m_triplets.push_back(TripletT(m_row_offset + i, e.vars[j].rep->index, m_weight*e.coeffs[j]));
    }
  }
  // vector<TripletT>& triplets() { return m_triplets; }

private:
  // const int m_num_residuals;
  const int m_row_offset;
  const double m_weight;
  vector<TripletT>& m_triplets;
};

class CostFunc {
public:
  CostFunc() { }
  virtual ~CostFunc() { }

  virtual string name() const = 0;
  virtual int num_residuals() const = 0;
  virtual bool is_linear() const = 0;

  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd>) = 0;
  virtual void linearize(const VectorXd&, JacobianContainer&) = 0;
};
typedef boost::shared_ptr<CostFunc> CostFuncPtr;


struct OptParams {
  double init_trust_region_size;
  // double trust_shrink_ratio;
  // double trust_expand_ratio;
  double min_trust_region_size;
  double min_approx_improve;
  // double improve_ratio_threshold;
  int max_iter;

  OptParams();
};

enum OptStatus {
  OPT_INCOMPLETE=0, OPT_CONVERGED, OPT_ITER_LIMIT
};

struct OptResult {
  OptStatus status;

  VectorXd x;
  double cost;
  VectorXd cost_vals;
  vector<double> cost_over_iters;

  int n_func_evals, n_qp_solves, n_iters;

  OptResult() : status(OPT_INCOMPLETE), n_func_evals(0), n_qp_solves(0), n_iters(0) { }
};
typedef boost::shared_ptr<OptResult> OptResultPtr;

struct OptimizerImpl;
class Optimizer {
public:
  Optimizer();
  OptParams& params();

  void add_vars(const StrVec& names, vector<Var>& out);
  void add_cost(CostFuncPtr cost, double coeff=1.);
  void set_cost_coeff(CostFuncPtr cost, double coeff);

  typedef boost::function<void(const VectorXd&)> Callback;
  void add_callback(const Callback &fn);

  int num_vars() const;

  OptResultPtr optimize(const VectorXd& start_x);

private:
  boost::shared_ptr<OptimizerImpl> m_impl;
};
typedef boost::shared_ptr<Optimizer> OptimizerPtr;
