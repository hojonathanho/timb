#pragma once

#include "common.hpp"
#include <Eigen/Sparse>

struct VarRep {
  VarRep(int _index, const std::string& _name, void* _creator) : index(_index), name(_name), removed(false), creator(_creator) {}
  int index;
  std::string name;
  bool removed;
  void* creator;
};

struct Var {
  VarRep* rep;
  Var() : rep(NULL) {}
  explicit Var(VarRep* rep) : rep(rep) {}
  Var(const Var& other) : rep(other.rep) {}
  double value(const Eigen::VectorXd& x) const {return x(rep->index);}
};

struct AffExpr { // affine expression
  double constant;
  std::vector<double> coeffs;
  std::vector<Var> vars;
  AffExpr() : constant(0) {}
  explicit AffExpr(double a) : constant(a) {}
  explicit AffExpr(const Var& v) : constant(0), coeffs(1, 1), vars(1, v) {}
  AffExpr(const AffExpr& other) :
    constant(other.constant), coeffs(other.coeffs), vars(other.vars) {}
  size_t size() const {return coeffs.size();}
  double value(const Eigen::VectorXd& x) const {
    double out = constant;
    for (size_t i=0; i < size(); ++i) {
      out += coeffs[i] * vars[i].value(x);
    }
    return out;
  }
};
typedef boost::shared_ptr<AffExpr> AffExprPtr;

struct QuadExpr {
  AffExpr affexpr;
  std::vector<double> coeffs;
  std::vector<Var> vars1;
  std::vector<Var> vars2;
  QuadExpr() {}
  explicit QuadExpr(double a) : affexpr(a) {}
  explicit QuadExpr(const Var& v) : affexpr(v) {}
  explicit QuadExpr(const AffExpr& aff) : affexpr(aff) {}
  size_t size() const {return coeffs.size();}
  double value(const Eigen::VectorXd& x) const {
    double out = affexpr.value(x);
    for (size_t i=0; i < size(); ++i) {
      out += coeffs[i] * vars1[i].value(x) * vars2[i].value(x);
    }
    return out;
  }
};
typedef boost::shared_ptr<QuadExpr> QuadExprPtr;

class MatrixQuadExpr {
public:
  typedef Eigen::SparseMatrix<double> SparseMatrixT;

  MatrixQuadExpr(int num_vars) : m_A(num_vars, num_vars), m_b(Eigen::VectorXd::Zero(num_vars)), m_c(0.) { }
  MatrixQuadExpr(int num_vars, const QuadExpr &qe) : m_A(num_vars, num_vars), m_b(Eigen::VectorXd::Zero(num_vars)), m_c(0.) {
    set_from_expr(qe);
  }

  const Eigen::SparseSelfAdjointView<SparseMatrixT, Eigen::Lower> A() const { return m_A.selfadjointView<Eigen::Lower>(); }
  const Eigen::VectorXd& b() const { return m_b; }
  double c() const { return m_c; }

  void set_from_expr(const QuadExpr &qe) {
    // quadratic part
    m_A.setZero();
    typedef Eigen::Triplet<double> T;
    vector<T> triplets;
    triplets.reserve(qe.size());
    for (int i = 0; i < qe.size(); ++i) {
      int ind1 = qe.vars1[i].rep->index, ind2 = qe.vars2[i].rep->index;
      // only fill in lower diagonal
      if (ind1 < ind2) { std::swap(ind1, ind2); }
      double coeff = qe.coeffs[i];
      if (ind1 == ind2) { coeff *= 2; }
      triplets.push_back(T(ind1, ind2, coeff));
    }
    m_A.setFromTriplets(triplets.begin(), triplets.end());

    // affine part
    m_b.setZero();
    for (int i = 0; i < qe.affexpr.size(); ++i) {
      m_b(qe.affexpr.vars[i].rep->index) = qe.affexpr.coeffs[i];
    }
    m_c = qe.affexpr.constant;
  }
  void set_from_expr(QuadExprPtr qe) { set_from_expr(*qe); }

  double value(const Eigen::VectorXd& x) const {
    assert(x.size() == m_b.size());
    return .5*x.dot(A()*x) + m_b.dot(x) + m_c;
  }

private:
  // 1/2 x^T A x + b^T x + c
  SparseMatrixT m_A; // lower diagonal only
  Eigen::VectorXd m_b;
  double m_c;
};
typedef boost::shared_ptr<MatrixQuadExpr> MatrixQuadExprPtr;


////// In-place operations ///////

// multiplication
inline void exprScale(AffExpr& v, double a) {
  v.constant *= a;
  for (int i=0; i < v.coeffs.size(); ++i) v.coeffs[i] *= a;
}
inline void exprScale(QuadExpr& q, double a) {
  exprScale(q.affexpr, a);
  for (int i=0; i < q.coeffs.size(); ++i) q.coeffs[i] *= a;
}

// addition
inline void exprInc(AffExpr& a, double b) {
  a.constant += b;
}
inline void exprInc(AffExpr& a, const AffExpr& b) {
  a.constant += b.constant;
  a.coeffs.insert(a.coeffs.end(), b.coeffs.begin(), b.coeffs.end());
  a.vars.insert(a.vars.end(), b.vars.begin(), b.vars.end());
}
inline void exprInc(AffExpr& a, const Var& b) {
  exprInc(a, AffExpr(b));
}
inline void exprInc(QuadExpr& a, double b) {
  exprInc(a.affexpr, b);
}
inline void exprInc(QuadExpr& a, const Var& b) {
  exprInc(a.affexpr, AffExpr(b));
}
inline void exprInc(QuadExpr& a, const AffExpr& b) {
  exprInc(a.affexpr, b);
}
inline void exprInc(QuadExpr& a, const QuadExpr& b) {
  exprInc(a.affexpr, b.affexpr);
  a.coeffs.insert(a.coeffs.end(), b.coeffs.begin(), b.coeffs.end());
  a.vars1.insert(a.vars1.end(), b.vars1.begin(), b.vars1.end());
  a.vars2.insert(a.vars2.end(), b.vars2.begin(), b.vars2.end());
}


// subtraction
inline void exprDec(AffExpr& a, double b) {
  a.constant -= b;
}
inline void exprDec(AffExpr& a, AffExpr b) {
  exprScale(b, -1);
  exprInc(a, b);
}
inline void exprDec(AffExpr& a, const Var& b) {
  exprDec(a, AffExpr(b));
}
inline void exprDec(QuadExpr& a, double b) {
  exprDec(a.affexpr, b);
}
inline void exprDec(QuadExpr& a, const Var& b) {
  exprDec(a.affexpr, b);
}
inline void exprDec(QuadExpr& a, const AffExpr& b) {
  exprDec(a.affexpr, b);
}
inline void exprDec(QuadExpr& a, QuadExpr b) {
  exprScale(b, -1);
  exprInc(a, b);
}

/////////////////////

inline AffExpr exprMult(const Var& a, double b) {
  AffExpr c(a);
  exprScale(c,b);
  return c;
}
// multiplication
inline AffExpr exprMult(AffExpr a, double b) {
  exprScale(a,b);
  return a;
}
inline QuadExpr exprMult(QuadExpr a, double b) {
  exprScale(a, b);
  return a;
}



inline AffExpr exprAdd(AffExpr a, double b) {
  exprInc(a, b);
  return a;
}
inline AffExpr exprAdd(AffExpr a, const Var& b) {
  exprInc(a, b);
  return a;
}
inline AffExpr exprAdd(AffExpr a, const AffExpr& b) {
  exprInc(a, b);
  return a;
}
inline QuadExpr exprAdd(QuadExpr a, double b) {
  exprInc(a, b);
  return a;
}
inline QuadExpr exprAdd(QuadExpr a, const Var& b) {
  exprInc(a, b);
  return a;
}
inline QuadExpr exprAdd(QuadExpr a, const AffExpr& b) {
  exprInc(a, b);
  return a;
}
inline QuadExpr exprAdd(QuadExpr a, const QuadExpr& b) {
  exprInc(a, b);
  return a;
}

inline AffExpr exprSub(AffExpr a, double b) {
  exprDec(a, b);
  return a;
}
inline AffExpr exprSub(AffExpr a, const Var& b) {
  exprDec(a, b);
  return a;
}
inline AffExpr exprSub(AffExpr a, const AffExpr& b) {
  exprDec(a, b);
  return a;
}
inline QuadExpr exprSub(QuadExpr a, double b) {
  exprDec(a, b);
  return a;
}
inline QuadExpr exprSub(QuadExpr a, const Var& b) {
  exprDec(a, b);
  return a;
}
inline QuadExpr exprSub(QuadExpr a, const AffExpr& b) {
  exprDec(a, b);
  return a;
}
inline QuadExpr exprSub(QuadExpr a, const QuadExpr& b) {
  exprDec(a, b);
  return a;
}



inline AffExpr operator+(const Var& x, double y) {
  return exprAdd(AffExpr(x), y);
}
inline AffExpr operator+(const AffExpr& x, double y) {
  return exprAdd(x, y);
}
inline QuadExpr operator+(const QuadExpr& x, double y) {
  return exprAdd(x, y);
}

inline AffExpr operator+(const Var& x, const Var& y) {
  return exprAdd(AffExpr(x), y);
}
inline AffExpr operator+(const AffExpr& x, const Var& y) {
  return exprAdd(x, y);
}
inline QuadExpr operator+(const QuadExpr& x, const Var& y) {
  return exprAdd(x, y);
}

inline AffExpr operator+(const Var& x, const AffExpr& y) {
  return exprAdd(AffExpr(x), y);
}
inline AffExpr operator+(const AffExpr& x, const AffExpr& y) {
  return exprAdd(x, y);
}
inline QuadExpr operator+(const QuadExpr& x, const AffExpr& y) {
  return exprAdd(x, y);
}

inline QuadExpr operator+(const Var& x, const QuadExpr& y) {
  return exprAdd(QuadExpr(x), y);
}
inline QuadExpr operator+(const AffExpr& x, const QuadExpr& y) {
  return exprAdd(QuadExpr(x), y);
}
inline QuadExpr operator+(const QuadExpr& x, const QuadExpr& y) {
  return exprAdd(x, y);
}




inline AffExpr operator-(const Var& x, double y) {
  return exprSub(AffExpr(x), y);
}
inline AffExpr operator-(const AffExpr& x, double y) {
  return exprSub(x, y);
}
inline QuadExpr operator-(const QuadExpr& x, double y) {
  return exprSub(x, y);
}

inline AffExpr operator-(const Var& x, const Var& y) {
  return exprSub(AffExpr(x), y);
}
inline AffExpr operator-(const AffExpr& x, const Var& y) {
  return exprSub(x, y);
}
inline QuadExpr operator-(const QuadExpr& x, const Var& y) {
  return exprSub(x, y);
}

inline AffExpr operator-(const Var& x, const AffExpr& y) {
  return exprSub(AffExpr(x), y);
}
inline AffExpr operator-(const AffExpr& x, const AffExpr& y) {
  return exprSub(x, y);
}
inline QuadExpr operator-(const QuadExpr& x, const AffExpr& y) {
  return exprSub(x, y);
}

inline QuadExpr operator-(const Var& x, const QuadExpr& y) {
  return exprSub(QuadExpr(x), y);
}
inline QuadExpr operator-(const AffExpr& x, const QuadExpr& y) {
  return exprSub(QuadExpr(x), y);
}
inline QuadExpr operator-(const QuadExpr& x, const QuadExpr& y) {
  return exprSub(x, y);
}

///////////////



inline AffExpr operator*(double a, const Var& b) {
  return exprMult(b, a);
}
inline AffExpr operator*(double a, const AffExpr& b) {
  return exprMult(b, a);
}
inline QuadExpr operator*(double a, const QuadExpr& b) {
  return exprMult(b, a);
}

inline AffExpr operator*(const Var& a, double b) {
  return exprMult(a, b);
}
inline AffExpr operator*(const AffExpr& a, double b) {
  return exprMult(a, b);
}
inline QuadExpr operator*(const QuadExpr& a, double b) {
  return exprMult(a, b);
}

inline AffExpr operator-(const Var& a) {
  return exprMult(a, -1);
}
inline AffExpr operator-(const AffExpr& a) {
  return exprMult(a, -1);
}


QuadExpr exprSquare(const Var&);
QuadExpr exprSquare(const AffExpr&);

AffExpr cleanupAff(const AffExpr&);
QuadExpr cleanupQuad(const QuadExpr&); //warning: might make it non-psd!


