#pragma once

#include "common.hpp"
#include "expr.hpp"

struct CostFunc {
  string m_name;
  CostFunc(const string& name): m_name(name) { }
  const string& name() const { return m_name; }

  virtual double eval(const VectorXd&) = 0;
  virtual QuadFunctionPtr quadratic(const VectorXd&) = 0;
};
typedef boost::shared_ptr<CostFunc> CostFuncPtr;


struct QuadraticCostFunc : public CostFunc {
  const QuadExpr m_quad_expr;
  QuadFunctionPtr m_quad;
  QuadraticCostFunc(const string& name, const QuadExpr& quad_expr) : m_quad_expr(quad_expr), m_quad(new QuadFunction(quad_expr)), CostFunc(name) { }

  virtual double eval(const VectorXd& x) { return m_quad_expr.value(x); }
  virtual QuadFunctionPtr quadratic(const VectorXd&) { return m_quad; }
};
typedef boost::shared_ptr<QuadraticCostFunc> QuadraticCostFuncPtr;


struct OptParams {
  double init_trust_region_size;
  double trust_shrink_ratio;
  double trust_expand_ratio;
  double min_trust_region_size;
  double min_approx_improve;
  double improve_ratio_threshold;
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

  int n_func_evals, n_qp_solves, n_iters;

  OptResult() : status(OPT_INCOMPLETE), n_func_evals(0), n_qp_solves(0), n_iters(0) { }
};
typedef boost::shared_ptr<OptResult> OptResultPtr;
std::ostream& operator<<(std::ostream& o, const OptResult& r);

struct OptimizerImpl;
class Optimizer {
public:
  Optimizer();
  OptParams& params();

  void add_vars(const StrVec& names, vector<Var>& out);
  void add_cost(CostFuncPtr fn);

  int num_vars() const;

  OptResultPtr optimize(const VectorXd& start_x);

private:
  boost::shared_ptr<OptimizerImpl> m_impl;
};
