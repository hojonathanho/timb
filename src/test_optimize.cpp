#include "problem.hpp"

#include <iostream>
using namespace std;

int main() {
  Optimizer opt;

  vector<string> var_names;
  var_names.push_back("x");
  var_names.push_back("y");
  var_names.push_back("z");
  vector<Var> vars;
  opt.add_vars(var_names, vars);
  Var x = vars[0], y = vars[1], z = vars[2];

  QuadraticCostFuncPtr cost(new QuadraticCostFunc("cost", exprSquare(x-1) + exprSquare(y-2) + exprSquare(z+3) + 2.4));
  opt.add_cost(cost);

  OptResultPtr result = opt.optimize(Eigen::Vector3d(0,0,0));
  cout << result->x.transpose() << endl;
  cout << result->cost << endl;

  return 0;
}
