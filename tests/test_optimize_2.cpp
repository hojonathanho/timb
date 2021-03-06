#include "optimizer.hpp"

#include <iostream>
#include <boost/format.hpp>
using namespace std;

#define XDIM 1000000


struct Cost : public CostFunc {
  vector<Var> m_vars;
  Cost(const vector<Var>& vars) : m_vars(vars) { }

  string name() const { return "cost"; }
  int num_residuals() const { return m_vars.size(); }
  bool is_linear() const { return true; }

  virtual void eval(const VectorXd& x, Eigen::Ref<VectorXd> out_residuals) {
    assert(out_residuals.size() == num_residuals());
    std::cout << "eval" << std::endl;
    for (int i = 0; i < x.size(); ++i) {
      out_residuals[i] = x[i] - 1.0;
    }
  }

  virtual void linearize(const VectorXd&, CostFuncLinearization& out) {
    // assert(out.size() == num_residuals());
    std::cout << "linearize" << std::endl;

    for (int i = 0; i < m_vars.size(); ++i) {
      out.set_by_expr(i, m_vars[i] - 1.0);
    }
  }
};

int main() {
  Optimizer opt;
  // opt.params().min_approx_improve = 1e-10;

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
  cout << "x: " << result->x.maxCoeff() << ' ' << result->x.minCoeff() << endl;
  cout << "cost: " << result->cost << endl;

  return 0;
}
