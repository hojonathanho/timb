#include <iostream>
using namespace std;

#include "expr.hpp"
#include <Eigen/IterativeLinearSolvers>

class VarFactory {
public:
  VarFactory() : m_curr_index(0) { }
  ~VarFactory() {
    for (int i = 0; i < m_reps.size(); ++i) {
      delete m_reps[i];
    }
  }
  Var make_var(const string& name) {
    VarRep* rep = new VarRep(m_curr_index++, name, this);
    m_reps.push_back(rep);
    return Var(rep);
  }

private:
  int m_curr_index;
  vector<VarRep*> m_reps;
};

int main() {
  VarFactory factory;
  Var x = factory.make_var("x");
  Var y = factory.make_var("y");
  Var z = factory.make_var("z");
  QuadExpr q = 5*exprSquare(x-2.) + exprSquare(z + .5*y) + 3.8;

  for (int i = 0; i < q.size(); ++i) {
    cout << q.vars1[i].rep->name << ' ' << q.vars2[i].rep->name << ' ' << q.coeffs[i] << endl;
  }

  MatrixQuadExpr mq(3, q);
  // cout << "A:\n" << Eigen::MatrixXd(mq.A) << endl;
  // cout << "b: " << mq.b.transpose() << endl;
  // cout << "c: " << mq.c << endl;

  Eigen::Vector3d v(17, -38, 2);
  cout << mq.value(v) << endl;
  cout << q.value(v) << endl;

  Eigen::ConjugateGradient<MatrixQuadExpr::SparseMatrixT, Eigen::Lower> solver;
  Eigen::VectorXd x0 = solver.compute(mq.A_lower()).solve(-mq.b());
  cout << mq.value(x0) << ' ' << q.value(x0) << endl;
  cout << x0.transpose() << endl;

  return 0;
}
