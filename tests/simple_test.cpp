#include "optimizer.hpp"

#include "gtest/gtest.h"

#include <iostream>
using namespace std;

struct Cost : public CostFunc {
  const double m_c;
  Var m_x;
  int m_evals;

  Cost(double c, Var x) : m_c(c), m_x(x), m_evals(0) { }

  string name() const { return "cost1"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return true; }

  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = x(0) - m_c;
    ++m_evals;
  }

  virtual void linearize(const VectorXd& x, CostFuncLinearization& out) {
    out.set_by_expr(0, m_x - m_c);
  }
};

static void run(double c1, double c2, double w1, double w2) {
  Optimizer opt;

  vector<Var> vars;
  opt.add_vars({"x"}, vars);

  CostFuncPtr cost1(new Cost(c1, vars[0]));
  CostFuncPtr cost2(new Cost(c2, vars[0]));
  opt.add_cost(cost1);
  opt.add_cost(cost2);

  opt.set_cost_coeff(cost1, w1);
  opt.set_cost_coeff(cost2, w2);

  VectorXd init_x(1);
  init_x << 1.;
  OptResultPtr result = opt.optimize(init_x);

  double expected = (w1*c1 + w2*c2) / (w1 + w2);
  EXPECT_LT(fabs(result->x(0) - expected), 1e-9);
}

TEST(SimpleTest, WeightTest) {
  run(10., -18., .3, 5.);
  run(0., 3., 1., 1.);
}

TEST(SimpleTest, NegativeWeightTest) {
  EXPECT_NO_THROW({run(-1., 1., 0., 1.);});
  EXPECT_ANY_THROW({run(-1., 1., 3., -5.);}); // negative weights should throw exceptions
}

TEST(SimpleTest, AddAndRemoveCostsTest) {
  Optimizer opt;

  vector<Var> vars;
  opt.add_vars({"x"}, vars);

  const double c1 = -3.4, c2 = 95.;

  CostFuncPtr cost1(new Cost(c1, vars[0]));
  CostFuncPtr cost2(new Cost(c2, vars[0]));
  opt.add_cost(cost1);
  opt.add_cost(cost2);

  VectorXd init_x(1); init_x << 1.;
  OptResultPtr result;

  ((Cost&) *cost1).m_evals = ((Cost&) *cost2).m_evals = 0;
  opt.set_cost_coeff(cost1, 0.);
  result = opt.optimize(init_x);
  EXPECT_LT(fabs(result->x(0) - c2), 1e-8);
  EXPECT_EQ(0, ((Cost&) *cost1).m_evals);

  ((Cost&) *cost1).m_evals = ((Cost&) *cost2).m_evals = 0;
  opt.set_cost_coeff(cost1, 1.);
  opt.set_cost_coeff(cost2, 0.);
  result = opt.optimize(init_x);
  EXPECT_LT(fabs(result->x(0) - c1), 1e-8);
  EXPECT_EQ(0, ((Cost&) *cost2).m_evals);

  opt.set_cost_coeff(cost1, 1.);
  opt.set_cost_coeff(cost2, 1.);
  result = opt.optimize(init_x);
  EXPECT_LT(fabs(result->x(0) - .5*(c1+c2)), 1e-8);
}