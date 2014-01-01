#include "optimizer.hpp"

#include <iostream>
using namespace std;

static inline double sq(double x) { return x*x; }

struct PowellProbVars {
  Var x, y, z, w;
};

struct PowellCost1 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost1(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost1"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = x(0) + 10*x(1);
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0, m_vars.x + 10*m_vars.y);
  }
};

struct PowellCost2 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost2(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost2"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = sqrt(5.)*(x(2) - x(3));
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0, sqrt(5.)*(m_vars.z - m_vars.w));
  }
};

struct PowellCost3 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost3(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost3"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = sq(x(1) - 2*x(2));
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0, 2.*(x(1) - 2*x(2))*(m_vars.y - x(1)) - 4.*(x(1) - 2*x(2))*(m_vars.z - x(2)) + pow((x(1) - 2*x(2)),2));
  }
};

struct PowellCost4 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost4(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost4"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = sqrt(10.)*sq(x(0) - x(3));
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0, sqrt(10.)*2.*(x(0) - x(3))*(m_vars.x-x(0)) - sqrt(10.)*2.*(x(0) - x(3))*(m_vars.w-x(3)) + sqrt(10.)*pow(x(0) - x(3), 2.));
  }
};

int main() {
  Optimizer opt;

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
