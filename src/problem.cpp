#include "problem.hpp"

#include <cstdio>
using std::printf;

OptParams::OptParams() :
  init_trust_region_size(1e3),
  trust_shrink_ratio(.1),
  trust_expand_ratio(2.),
  min_trust_region_size(1e-4),
  min_approx_improve(1e-6),
  improve_ratio_threshold(.25),
  max_iter(50)
{ }

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
    m_vars.push_back(Var(rep));
    return m_vars.back();
  }
  int num_vars() const { return m_reps.size(); }
  const vector<Var>& vars() const { return m_vars; }

private:
  int m_curr_index;
  vector<VarRep*> m_reps;
  vector<Var> m_vars; // redundant but convenient
};

static string status_to_str(OptStatus s) {
  switch (s) {
  case OPT_INCOMPLETE:
    return "incomplete";
  case OPT_CONVERGED:
    return "converged";
  case OPT_ITER_LIMIT:
    return "iteration limit";
   default:
    assert(false);
  }
}

// TODO: anisotropic trust region
struct TrustRegion {
  // trust region params
  double m_cost_coeff;

  vector<Var> m_vars;
  QuadFunctionPtr m_quad;

  TrustRegion(const vector<Var>& vars) : m_vars(vars) { }

  void set_center(const VectorXd& center) {
    assert(center.size() == m_vars.size());
    QuadExpr expr;
    for (int i = 0; i < m_vars.size(); ++i) {
      exprInc(expr, exprSquare(m_vars[i] - center[i]));
    }
    m_quad.reset(new QuadFunction(expr));
    m_quad->init_with_num_vars(m_vars.size());
  }
  void set_size(double size) {
    m_cost_coeff = 1./(size + 1e-10);
  }

  void add_to(QuadFunction::SparseMatrixT& other_A_lower, VectorXd& other_b, double& other_c) const {
    other_A_lower += m_cost_coeff * m_quad->A_lower();
    other_b += m_cost_coeff * m_quad->b();
    other_c += m_cost_coeff * m_quad->c();
  }

  double cost_coeff() const { return m_cost_coeff; }
};

struct OptimizerImpl {
  OptParams m_params;
  VarFactory m_var_factory;

  vector<CostFuncPtr> m_costs;
  vector<double> m_cost_coeffs;
  map<CostFuncPtr, int> m_cost2idx;

  int num_vars() const { return m_var_factory.num_vars(); }

  void add_vars(const StrVec& names, vector<Var>& out) {
    for (int i = 0; i < names.size(); ++i) {
      out.push_back(m_var_factory.make_var(names[i]));
    }
  }
  void add_cost(CostFuncPtr cost, double coeff) {
    assert(m_costs.size() == m_cost_coeffs.size());
    m_costs.push_back(cost);
    m_cost_coeffs.push_back(coeff);
    m_cost2idx[cost] = m_costs.size() - 1;
  }
  void set_cost_coeff(CostFuncPtr cost, double coeff) {
    assert(m_cost2idx.find(cost) != m_cost2idx.end());
    m_cost_coeffs[m_cost2idx[cost]] = coeff;
  }

  void convexify_costs(const VectorXd& x, vector<QuadFunctionPtr>& out) {
    out.resize(m_costs.size());
    for (int i = 0; i < m_costs.size(); ++i) {
      out[i] = m_costs[i]->quadratic(x);
      out[i]->init_with_num_vars(num_vars());
    }
  }

  void eval_quad_costs(const vector<QuadFunctionPtr>& quad_costs, const VectorXd& x, VectorXd& out) {
    assert(out.size() == m_costs.size() && out.size() == quad_costs.size());
    for (int i = 0; i < quad_costs.size(); ++i) {
      out[i] = m_cost_coeffs[i] * quad_costs[i]->value(x);
    }
  }

  void eval_true_costs(const VectorXd& x, VectorXd& out) {
    assert(out.size() == m_costs.size());
    for (int i = 0; i < m_costs.size(); ++i) {
      out[i] = m_cost_coeffs[i] * m_costs[i]->eval(x);
    }
  }

  void print_cost_info(const VectorXd& old_cost_vals, const VectorXd& quad_cost_vals, const VectorXd& new_cost_vals,
                       double old_merit, double approx_merit_improve, double exact_merit_improve, double merit_improve_ratio) {
    assert(m_costs.size() == quad_cost_vals.size() && m_costs.size() == new_cost_vals.size() && m_costs.size() == old_cost_vals.size());

    LOG_INFO("%10s | %10s | %10s | %10s | %10s", "", "oldexact", "dapprox", "dexact", "ratio");
    LOG_INFO("%10s | %10s---%10s---%10s---%10s", "COSTS", "----------", "----------", "----------", "----------");
    for (int i = 0; i < old_cost_vals.size(); ++i) {
      double approx_improve = old_cost_vals[i] - quad_cost_vals[i];
      double exact_improve = old_cost_vals[i] - new_cost_vals[i];
      double ratio = exact_improve / approx_improve;
      if (fabs(approx_improve) > 1e-8) {
        LOG_INFO("%10s | %10.3e | %10.3e | %10.3e | %10.3e", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, ratio);
      } else {
        LOG_INFO("%10s | %10.3e | %10.3e | %10.3e | %10s", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, "  ------  ");
      }
    }
    LOG_INFO("%10s | %10.3e | %10.3e | %10.3e | %10.3e", "TOTAL", old_merit, approx_merit_improve, exact_merit_improve, merit_improve_ratio);
  }

  OptResultPtr optimize(const VectorXd& start_x) {
    assert(start_x.size() == num_vars());
    OptResultPtr result(new OptResult);
    result->status = OPT_INCOMPLETE;
    result->x = start_x;
    result->cost_vals = VectorXd::Zero(m_costs.size());
    eval_true_costs(start_x, result->cost_vals);
    result->cost = result->cost_vals.sum();

    VectorXd new_x;
    double trust_region_size = m_params.init_trust_region_size;

    vector<QuadFunctionPtr> quad_costs(m_costs.size(), QuadFunctionPtr());
    VectorXd quad_cost_vals(VectorXd::Zero(m_costs.size()));
    VectorXd new_cost_vals(VectorXd::Zero(m_costs.size()));
    Eigen::ConjugateGradient<QuadFunction::SparseMatrixT, Eigen::Lower> solver;
    QuadFunction::SparseMatrixT quad_A_lower(num_vars(), num_vars());
    VectorXd quad_b(VectorXd::Zero(num_vars()));
    double quad_c = 0.;

    TrustRegion trust_region(m_var_factory.vars());

    int iter = 0;
    while (true) {
      convexify_costs(result->x, quad_costs);
      trust_region.set_center(result->x);

      while (trust_region_size >= m_params.min_trust_region_size) {
        // set trust region
        trust_region.set_size(trust_region_size);

        // build the quadratic subproblem
        quad_A_lower.setZero(); quad_b.setZero(); quad_c = 0.;
        for (int i = 0; i < quad_costs.size(); ++i) {
          quad_costs[i]->add_to(quad_A_lower, quad_b, quad_c, m_cost_coeffs[i]);
        }
        trust_region.add_to(quad_A_lower, quad_b, quad_c);
        quad_A_lower.makeCompressed(); // TODO: every iteration?


        // solve the quadratic subproblem
        solver.compute(quad_A_lower);
        new_x = solver.solve(-quad_b); // TODO: warm start and early termination
        ++result->n_qp_solves;

        eval_quad_costs(quad_costs, new_x, quad_cost_vals);
        eval_true_costs(new_x, new_cost_vals);
        ++result->n_func_evals;

        double old_merit = result->cost;
        double model_merit = quad_cost_vals.sum();
        double new_merit = new_cost_vals.sum();
        double approx_merit_improve = old_merit - model_merit;
        double exact_merit_improve = old_merit - new_merit;
        double merit_improve_ratio = exact_merit_improve / approx_merit_improve;

        if (util::GetLogLevel() >= util::LevelInfo) {
          LOG_INFO("Iteration %d", iter + 1);
          print_cost_info(result->cost_vals, quad_cost_vals, new_cost_vals, old_merit, approx_merit_improve, exact_merit_improve, merit_improve_ratio);
        }

        if (approx_merit_improve < -1e-5) {
          LOG_ERROR("approximate merit function got worse (%.3e). (convexification is probably wrong to zeroth order)", approx_merit_improve);
        }
        if (approx_merit_improve < m_params.min_approx_improve) {
          LOG_INFO("converged because improvement was small (%.3e < %.3e)", approx_merit_improve, m_params.min_approx_improve);
          result->status = OPT_CONVERGED;
          goto out;
        }
        if (exact_merit_improve < 0 || merit_improve_ratio < m_params.improve_ratio_threshold) {
          trust_region_size *= m_params.trust_shrink_ratio;
          LOG_INFO("shrunk trust region. new radius: %.3e, cost: %.3e", trust_region_size, trust_region.cost_coeff());
        } else {
          result->x = new_x;
          result->cost_vals = new_cost_vals;
          result->cost = new_cost_vals.sum();
          trust_region_size *= m_params.trust_expand_ratio;
          LOG_INFO("expanded trust region. new radius: %.3e, cost: %.3e", trust_region_size, trust_region.cost_coeff());
          break;
        }
      }

      if (trust_region_size < m_params.min_trust_region_size) {
        LOG_INFO("converged because trust region is tiny");
        result->status = OPT_CONVERGED;
        goto out;
      } else if (iter >= m_params.max_iter) {
        LOG_INFO("iteration limit");
        result->status = OPT_ITER_LIMIT;
        goto out;
      }

      ++iter;
    }

  out:
    result->n_iters = iter;
    LOG_INFO("Final results:\n\tStatus: %s\n\tCost: %.4e\n\tIterations: %d", status_to_str(result->status).c_str(), result->cost, result->n_iters);
    return result;
  }

};


Optimizer::Optimizer() : m_impl(new OptimizerImpl()) { }
OptParams& Optimizer::params() { return m_impl->m_params; }
void Optimizer::add_vars(const vector<string>& names, vector<Var>& out) { m_impl->add_vars(names, out); }
void Optimizer::add_cost(CostFuncPtr cost, double coeff) { m_impl->add_cost(cost, coeff); }
OptResultPtr Optimizer::optimize(const VectorXd& start_x) { return m_impl->optimize(start_x); }