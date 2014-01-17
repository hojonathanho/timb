#include "expr.hpp"

#define ZERO_THRESH 1e-13

static inline double sq(double x) { return x*x; }

VarFactory::~VarFactory() {
  for (int i = 0; i < m_reps.size(); ++i) {
    delete m_reps[i];
  }
}

Var VarFactory::make_var(const string& name) {
  VarRep* rep = new VarRep(m_curr_index++, name, this);
  m_reps.push_back(rep);
  m_vars.push_back(Var(rep));
  return m_vars.back();
}

std::ostream& operator<<(std::ostream& o, const Var& v) {
  if (v.rep != NULL)
    o << v.rep->name;
  else
    o << "nullvar";
  return o;
}
std::ostream& operator<<(std::ostream& o, const AffExpr& e) {
  o << e.constant;
  for (size_t i=0; i < e.size(); ++i) {
    o << " + " << e.coeffs[i] << "*" << e.vars[i];
  }
  return o;
}
std::ostream& operator<<(std::ostream& o, const QuadExpr& e) {
  o << e.affexpr;
  for (size_t i=0; i < e.size(); ++i) {
    o << " + " << e.coeffs[i] << "*" << e.vars1[i] << "*" << e.vars2[i];
  }
  return o;
}

// Sorts an AffExpr's variables and coeffs by the variable name (in-place)
static void sort_vars_by_name(AffExpr& a) {
  typedef std::pair<Var, double> P;
  std::vector<P> var2coeff;
  const int n = a.coeffs.size();
  assert(a.coeffs.size() == a.vars.size());
  for (int i = 0; i < n; ++i) {
    var2coeff.push_back(std::make_pair(a.vars[i], a.coeffs[i]));
  }
  std::sort(var2coeff.begin(), var2coeff.end(), [](const P& x, const P& y) {
    return x.first.rep->name < y.first.rep->name;
  });
  for (int i = 0; i < n; ++i) {
    a.vars[i] = var2coeff[i].first;
    a.coeffs[i] = var2coeff[i].second;
  }
}

bool close(const AffExpr& a_, const AffExpr& b_, double rtol, double atol) {
  AffExpr a = reduceAff(a_), b = reduceAff(b_);
  if (a.coeffs.size() != b.coeffs.size()) return false;
  if (!close(a.constant, b.constant)) return false;
  for (int i = 0; i < a.coeffs.size(); ++i) {
    if (a.vars[i].rep->name != b.vars[i].rep->name) return false;
    if (!close(a.coeffs[i], b.coeffs[i], rtol, atol)) return false;
  }
  return true;
}

QuadExpr exprSquare(const Var& a) {
  QuadExpr out;
  out.coeffs.push_back(1);
  out.vars1.push_back(a);
  out.vars2.push_back(a);
  return out;
}

QuadExpr exprSquare(const AffExpr& affexpr) {
  QuadExpr out;
  size_t naff = affexpr.coeffs.size();
  size_t nquad = (naff*(naff+1))/2;

  out.affexpr.constant = sq(affexpr.constant);

  out.affexpr.vars = affexpr.vars;
  out.affexpr.coeffs.resize(naff);
  for (size_t i=0; i < naff; ++i) out.affexpr.coeffs[i] = 2*affexpr.constant*affexpr.coeffs[i];

  out.coeffs.reserve(nquad);
  out.vars1.reserve(nquad);
  out.vars2.reserve(nquad);
  for (size_t i=0; i < naff; ++i) {
    out.vars1.push_back(affexpr.vars[i]);
    out.vars2.push_back(affexpr.vars[i]);
    out.coeffs.push_back(sq(affexpr.coeffs[i]));
    for (size_t j=i+1; j < naff; ++j) {
      out.vars1.push_back(affexpr.vars[i]);
      out.vars2.push_back(affexpr.vars[j]);
      out.coeffs.push_back(2 * affexpr.coeffs[i] * affexpr.coeffs[j]);
    }
  }
  return out;
}

AffExpr cleanupAff(const AffExpr& a) {
  AffExpr out;
  for (size_t i=0; i < a.size(); ++i) {
    if (fabs(a.coeffs[i]) > ZERO_THRESH) {
      out.coeffs.push_back(a.coeffs[i]);
      out.vars.push_back(a.vars[i]);
    }
  }
  out.constant = a.constant;
  return out;
}

QuadExpr cleanupQuad(const QuadExpr& q) {
  QuadExpr out;
  out.affexpr = cleanupAff(q.affexpr);
  for (size_t i=0; i < q.size(); ++i) {
    if (fabs(q.coeffs[i]) > ZERO_THRESH) {
      out.coeffs.push_back(q.coeffs[i]);
      out.vars1.push_back(q.vars1[i]);
      out.vars2.push_back(q.vars2[i]);
    }
  }
  return out;
}

AffExpr reduceAff(const AffExpr& a) {
  AffExpr b = cleanupAff(a);
  sort_vars_by_name(b);
  AffExpr out(b.constant);
  int k = -1;
  for (int i = 0; i < b.size(); ++i) {
    if (k != -1 && b.vars[i].name() == b.vars[i-1].name()) {
      out.coeffs[k] += b.coeffs[i];
    } else {
      out.vars.push_back(b.vars[i]);
      out.coeffs.push_back(b.coeffs[i]);
      ++k;
    }
  }
  return out;
}
