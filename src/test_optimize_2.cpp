#include "problem.hpp"

#include <iostream>
#include <boost/format.hpp>
using namespace std;

#define XDIM 1000000

static inline double sq(double x) { return x*x; }

struct Cost : public CostFunc {
  vector<Var> m_vars;
  QuadFunctionPtr m_quad;
  Cost(const vector<Var>& vars) : m_vars(vars), CostFunc("cost") {
    QuadExpr expr;
    for (int i = 0; i < m_vars.size(); ++i) {
      exprInc(expr, exprSquare(m_vars[i] - 1.0));
    }
    m_quad.reset(new QuadFunction(expr));
  }
  virtual double eval(const VectorXd& x) {
    double out = 0.;
    for (int i = 0; i < x.size(); ++i) {
      out += sq(x[i] - 1.0);
    }
    return out;
  }

  virtual QuadFunctionPtr quadratic(const VectorXd&) {
    return m_quad;
  }
};
int main() {
  Optimizer opt;
  opt.params().min_approx_improve = 1e-10;

  vector<string> var_names;
  for (int i = 0; i < XDIM; ++i) {
    var_names.push_back((boost::format("x_%d") % i).str());
  }
  vector<Var> varvec;
  opt.add_vars(var_names, varvec);
  opt.add_cost(CostFuncPtr(new Cost(varvec)));

  VectorXd init_x(XDIM);
  init_x.setZero();
  OptResultPtr result = opt.optimize(init_x);
  //cout << "x: " << result->x.transpose() << endl;
  cout << "cost: " << result->cost << endl;

  return 0;
}

