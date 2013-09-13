#include "problem.hpp"

#include <iostream>
using namespace std;

static inline double sq(double x) { return x*x; }

struct PowellProbVars {
  Var x, y, z, w;
};

struct PowellCost1 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost1(PowellProbVars &vars) : m_vars(vars), CostFunc("cost1") { }
  virtual double eval(const VectorXd& x) {
    return sq(x(0) + 10*x(1));
  }
  virtual QuadFunctionPtr quadratic(const VectorXd&) {
    return QuadFunctionPtr(new QuadFunction(exprSquare(m_vars.x + 10*m_vars.y)));
  }
};

struct PowellCost2 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost2(PowellProbVars &vars) : m_vars(vars), CostFunc("cost2") { }
  virtual double eval(const VectorXd& x) {
    return 5.*sq(x(2) - x(3));
  }
  virtual QuadFunctionPtr quadratic(const VectorXd&) {
    return QuadFunctionPtr(new QuadFunction(5.*exprSquare(m_vars.z - m_vars.w)));
  }
};

struct PowellCost3 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost3(PowellProbVars &vars) : m_vars(vars), CostFunc("cost3") { }
  virtual double eval(const VectorXd& x) {
    return pow(x(1) - 2*x(2), 4.);
  }
  virtual QuadFunctionPtr quadratic(const VectorXd& x) {
    return QuadFunctionPtr(new QuadFunction(exprSquare(2.*(x(1) - 2*x(2))*(m_vars.y - x(1)) - 4.*(x(1) - 2*x(2))*(m_vars.z - x(2)) + pow((x(1) - 2*x(2)),2))));
  }
};

struct PowellCost4 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost4(PowellProbVars &vars) : m_vars(vars), CostFunc("cost4") { }
  virtual double eval(const VectorXd& x) {
    return 10.*pow(x(0) - x(3), 4.);
  }
  virtual QuadFunctionPtr quadratic(const VectorXd& x) {
    return QuadFunctionPtr(new QuadFunction(exprSquare( sqrt(10.)*2.*(x(0) - x(3))*(m_vars.x-x(0)) - sqrt(10.)*2.*(x(0) - x(3))*(m_vars.w-x(3)) + sqrt(10.)*pow(x(0) - x(3), 2.))));
  }
};

int main() {
  Optimizer opt;
  opt.params().min_approx_improve = 1e-10;

  vector<string> var_names;
  var_names.push_back("x");
  var_names.push_back("y");
  var_names.push_back("z");
  var_names.push_back("w");
  vector<Var> varvec;
  opt.add_vars(var_names, varvec);
  PowellProbVars vars = { varvec[0], varvec[1], varvec[2], varvec[3] };

  opt.add_cost(CostFuncPtr(new PowellCost1(vars)));
  opt.add_cost(CostFuncPtr(new PowellCost2(vars)));
  opt.add_cost(CostFuncPtr(new PowellCost3(vars)));
  opt.add_cost(CostFuncPtr(new PowellCost4(vars)));

  VectorXd init_x(4);
  init_x << 3, -1, 0, 1;
  OptResultPtr result = opt.optimize(init_x);
  cout << "x: " << result->x.transpose() << endl;
  cout << "cost: " << result->cost << endl;

  return 0;
}
