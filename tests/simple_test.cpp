#include "optimizer.hpp"

#include "gtest/gtest.h"

#include <iostream>
using namespace std;

struct Cost : public CostFunc {
  const double m_c;
  Var m_x;

  Cost(double c, Var x) : m_c(c), m_x(x) { }

  string name() const { return "cost1"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return true; }

  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = x(0) - m_c;
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
