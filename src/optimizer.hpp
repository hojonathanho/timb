#pragma once

#include "common.hpp"
#include "expr.hpp"
#include <boost/function.hpp>

class CostFuncLinearization {
public:
  typedef Eigen::Triplet<double> TripletT;

  CostFuncLinearization(
    vector<TripletT>& triplets_ref,
    int num_residuals,
    int row_offset,
    double weight,
    bool store_triplets, // store triplets, for caching linearizations of linear costs
    bool store_exprs) // store expressions, only for checking linearizations

    : m_triplets_ref(triplets_ref)
      m_num_residuals(num_residuals),
      m_row_offset(row_offset),
      m_weight(weight),
      m_store_triplets(store_triplets),
      m_store_exprs(store_exprs) {

    if (m_store_exprs) {
      m_exprs.resize(m_num_residuals);
    }
  }

  void set_by_expr(int i, const AffExpr &e) {
    // assert(0 <= i && i < m_num_residuals);
    for (int j = 0; j < e.size(); ++j) {
      TripletT triplet(m_row_offset + i, e.vars[j].rep->index, m_weight*e.coeffs[j]);
      m_triplets_ref.push_back(triplet);
      if (m_store_triplets) {
        m_stored_triplets.push_back(triplet);
      }
    }

    if (m_store_exprs) {
      m_exprs[i] = e;
    }
  }

  const vector<AffExpr>& exprs() const {
    assert(m_store_exprs);
    return m_exprs;
  }

  const vector<TripletT>& stored_triplets() const {
    assert(m_store_triplets);
    return m_stored_triplets;
  }

private:
  const bool m_store_triplets;
  const bool m_store_exprs;
  const int m_num_residuals;
  const int m_row_offset;
  const double m_weight;
  vector<TripletT>& m_triplets_ref;

  vector<TripletT> m_stored_triplets; // store triplets. used for linear costs only (to avoid relinearizing)
  vector<AffExpr> m_exprs; // only used for numerical checking of gradients
};
typedef boost::shared_ptr<CostFuncLinearization> CostFuncLinearizationPtr;

class CostFunc {
public:
  CostFunc() { }
  virtual ~CostFunc() { }

  virtual string name() const = 0;
  virtual int num_residuals() const = 0;
  virtual bool is_linear() const = 0;

  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd>) = 0;
  virtual void linearize(const VectorXd&, CostFuncLinearization&) = 0;
};
typedef boost::shared_ptr<CostFunc> CostFuncPtr;


struct OptParams {
  double init_trust_region_size;
  // double trust_shrink_ratio;
  // double trust_expand_ratio;
  double min_trust_region_size;
  // double min_approx_improve;
  // double improve_ratio_threshold;
  double grad_convergence_tol;
  double approx_improve_rel_tol;
  int max_iter;
  bool check_linearizations;

  OptParams();
  string str() const;
};
// typedef boost::shared_ptr<OptParams> OptParamsPtr;

enum OptStatus {
  OPT_INCOMPLETE=0, OPT_CONVERGED, OPT_ITER_LIMIT
};

struct OptResult {
  OptStatus status;

  VectorXd x;
  double cost;
  VectorXd cost_vals;
  vector<double> cost_over_iters;

  int n_func_evals, n_jacobian_evals, n_qp_solves, n_iters;

  OptResult() : status(OPT_INCOMPLETE), n_func_evals(0), n_jacobian_evals(0), n_qp_solves(0), n_iters(0) { }
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
