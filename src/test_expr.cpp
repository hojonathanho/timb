#include <iostream>
using namespace std;

#include "expr.hpp"

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
  QuadExpr q = 5*x + exprSquare(z + .5*y) + 3;

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
  return 0;
}
