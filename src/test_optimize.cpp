#include "optimizer.hpp"

#include <iostream>
using namespace std;

static inline double sq(double x) { return x*x; }

struct PowellProbVars {
  Var x, y, z, w;
  double sx, sy, sz, sw;
};

struct PowellCost1 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost1(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost1"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = m_vars.sx*x(0) + 10*m_vars.sy*x(1);
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0, m_vars.sx*m_vars.x + 10*m_vars.sy*m_vars.y);
  }
};

struct PowellCost2 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost2(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost2"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = sqrt(5.)*(m_vars.sz*x(2) - m_vars.sw*x(3));
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0, sqrt(5.)*(m_vars.sz*m_vars.z - m_vars.sw*m_vars.w));
  }
};

struct PowellCost3 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost3(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost3"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = sq(m_vars.sy*x(1) - 2*m_vars.sz*x(2));
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0,
      2.*(m_vars.sy*x(1) - 2*m_vars.sz*x(2))*m_vars.sy*(m_vars.y - x(1)) - 4.*(m_vars.sy*x(1)
      - 2*m_vars.sz*x(2))*m_vars.sz*(m_vars.z - x(2))
      + sq(m_vars.sy*x(1) - 2*m_vars.sz*x(2))
    );
  }
};

struct PowellCost4 : public CostFunc {
  PowellProbVars m_vars;
  PowellCost4(PowellProbVars &vars) : m_vars(vars) { }
  string name() const { return "cost4"; }
  int num_residuals() const { return 1; }
  bool is_linear() const { return false; }
  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out[0] = sqrt(10.)*sq(m_vars.sx*x(0) - m_vars.sw*x(3));
  }
  virtual void linearize(const VectorXd& x, JacobianContainer& out) {
    out.set_by_expr(0,
      sqrt(10.)*2.*(m_vars.sx*x(0) - m_vars.sw*x(3))*m_vars.sx*(m_vars.x-x(0))
      - sqrt(10.)*2.*(m_vars.sx*x(0) - m_vars.sw*x(3))*m_vars.sw*(m_vars.w-x(3))
      + sqrt(10.)*sq(m_vars.sx*x(0) - m_vars.sw*x(3))
    );
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
  PowellProbVars vars = { varvec[0], varvec[1], varvec[2], varvec[3], 1, 10, 1, 1 };

  opt.add_cost(CostFuncPtr(new PowellCost1(vars)));
  opt.add_cost(CostFuncPtr(new PowellCost2(vars)));
  opt.add_cost(CostFuncPtr(new PowellCost3(vars)));
  opt.add_cost(CostFuncPtr(new PowellCost4(vars)));

  VectorXd init_x(4);
  init_x << 3./vars.sx, -1./vars.sy, 0./vars.sz, 1./vars.sw;
  OptResultPtr result = opt.optimize(init_x);
  cout << "x: " << result->x.transpose() << endl;
  cout << "cost: " << result->cost << endl;

  return 0;
}
