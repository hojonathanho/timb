#pragma once

#include "common.hpp"
#include "expr.hpp"

struct CostFunc {
  virtual std::string get_name() const = 0;
  virtual double eval();
  virtual QuadExprPtr convex();
};


struct QuadraticCostFunc {
  QuadExprPtr quad_expr;
  QuadraticCostFunc(QuadExpr quad_expr_) : quad_expr(quad_expr_);
};


struct OptParams {
  double init_trust_region_size;
  double trust_shrink_ratio;
  double trust_expand_ratio;
  double min_trust_region_size;
  double min_approx_improve;
  double improve_ratio_threshold;
  int max_iter;

  OptParams() :
    init_trust_region_size(1.),
    trust_shrink_ratio(.1),
    trust_expand_ratio(2.),
    min_trust_region_size(1e-4),
    min_approx_improve(1e-6),
    improve_ratio_threshold(.25),
    max_iter(50)
  { }
};

struct OptResult {
  VectorXd x;
  OptStatus status;
  double cost;

  int n_func_evals, n_qp_solves;
};
std::ostream& operator<<(std::ostream& o, const OptResult& r);

class OptProb {
public:
  OptProb() { }
  OptParams& params() { return m_params; }

  void add_vars(const StrVec& names);
  void add_cost(CostFuncPtr fn) { m_costs.push_back(fn); }

private:
  OptParams m_params;

  vector<Var> m_vars;
  vector<CostFuncPtr> m_costs;

};
