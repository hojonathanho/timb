#pragma once

#include "common.hpp"
#include <Eigen/Sparse>
#include <ostream>

///////// Symbolic quadratic expressions /////////

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
  string name() const { return rep->name; }
};

class VarFactory {
public:
  VarFactory() : m_curr_index(0) { }
  ~VarFactory();
  Var make_var(const string& name);
  int num_vars() const { return m_reps.size(); }
  const vector<Var>& vars() const { return m_vars; }

private:
  int m_curr_index;
  vector<VarRep*> m_reps;
  vector<Var> m_vars; // redundant but convenient
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

////// Stream operators ///////

std::ostream& operator<<(std::ostream& o, const Var& v);
std::ostream& operator<<(std::ostream& o, const AffExpr& e);
std::ostream& operator<<(std::ostream& o, const QuadExpr& e);


////// Equality checks (approximate) ///////
inline bool close(double a, double b, double rtol=1e-05, double atol=1e-08) { return fabs(a - b) <= (atol + rtol*fabs(b)); }
bool close(const AffExpr& a, const AffExpr& b, double rtol=1e-05, double atol=1e-08);

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

AffExpr reduceAff(const AffExpr&);
