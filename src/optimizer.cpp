#include "optimizer.hpp"

#include <map>
#include <cstdio>
using std::printf;
using std::map;

#include <boost/timer.hpp>

#include <Eigen/CholmodSupport>

OptParams::OptParams() :
  init_trust_region_size(1e4),
  // trust_shrink_ratio(.1),
  // trust_expand_ratio(2.),
  min_trust_region_size(1e-4),
  // min_approx_improve(1e-6),
  // improve_ratio_threshold(.25),
  grad_convergence_tol(1e-8),
  approx_improve_rel_tol(1e-8),
  max_iter(100),
  check_linearizations(false)
{ }

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

typedef Eigen::SparseMatrix<double> SparseMatrixT;
typedef Eigen::SparseSelfAdjointView<SparseMatrixT, Eigen::Lower> SparseSelfAdjointViewT;

#if 0
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
  // void set_size(double size) {
  //   m_cost_coeff = 1./(size + 1e-10);
  // }
  void set_coeff(double coeff) {
    m_cost_coeff = coeff;
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

  vector<Optimizer::Callback> m_callbacks;

  int num_vars() const { return m_var_factory.num_vars(); }

  int num_residuals() const {
    int n = 0;
    for (CostFuncPtr c : m_costs) {
      n += c->num_residuals();
    }
    return n;
  }

  void add_vars(const StrVec& names, vector<Var>& out) {
    out.clear();
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

  void add_callback(Optimizer::Callback fn) {
    m_callbacks.push_back(fn);
  }

  void convexify_costs(const VectorXd& x, vector<QuadFunctionPtr>& out) {
    LOG_DEBUG("Convexifying costs:");
    out.resize(m_costs.size());
    for (int i = 0; i < m_costs.size(); ++i) {
      LOG_DEBUG("\t%s", m_costs[i]->name().c_str());
      out[i] = m_costs[i]->quadratic(x);
      out[i]->init_with_num_vars(num_vars());
    }
    LOG_DEBUG("Done");
  }

  void eval_quad_costs(const vector<QuadFunctionPtr>& quad_costs, const VectorXd& x, VectorXd& out) {
    LOG_DEBUG("Evaluating model costs:");
    assert(out.size() == m_costs.size() && out.size() == quad_costs.size());
    for (int i = 0; i < quad_costs.size(); ++i) {
      out[i] = m_cost_coeffs[i] * quad_costs[i]->value(x);
      LOG_DEBUG("\t%s: %f", m_costs[i]->name().c_str(), out[i]);
    }
    LOG_DEBUG("Done");
  }

  void eval_true_costs(const VectorXd& x, VectorXd& out) {
    LOG_DEBUG("Evaluating true costs:");
    assert(out.size() == m_costs.size());
    for (int i = 0; i < m_costs.size(); ++i) {
      out[i] = m_cost_coeffs[i] * m_costs[i]->eval(x);
      LOG_DEBUG("\t%s: %f", m_costs[i]->name().c_str(), out[i]);
    }
    LOG_DEBUG("Done");
  }

  void linearize_costs(const VectorXd& x, SparseMatrixT& A, VectorXd& b) {
    A.resize(num_residuals(), num_vars());
    A.setZero();
    b.resize(num_residuals());
    b.setZero();

    typedef Eigen::Triplet<double> T;
    vector<T> triplets; //triplets.reserve(expr.size());

    int start_row = 0;
    for (CostFuncPtr c : m_costs) {
      vector<AffExpr> res_exprs(c->num_residuals());
      c->linearize(x, res_exprs);
      for (int i = 0; i < c->num_residuals(); ++i) {
        const AffExpr& expr = res_exprs[i];
        int row = start_row + i;
        for (int j = 0; j < expr.size(); ++j) {
          int col = expr.vars[j].rep->index;
          triplets.push_back(T(row, col, expr.coeffs[i]));
        }
        b(row) = expr.constant;
      }

      start_row += c->num_residuals();
    }



  }

  void print_cost_info(const VectorXd& old_cost_vals, const VectorXd& quad_cost_vals, const VectorXd& new_cost_vals,
                       double old_merit, double approx_merit_improve, double exact_merit_improve, double merit_improve_ratio) {
    assert(m_costs.size() == quad_cost_vals.size() && m_costs.size() == new_cost_vals.size() && m_costs.size() == old_cost_vals.size());

    LOG_INFO("%15s | %10s | %10s | %10s | %10s", "", "oldexact", "dapprox", "dexact", "ratio");
    LOG_INFO("%15s | %10s---%10s---%10s---%10s", "COSTS", "----------", "----------", "----------", "----------");
    for (int i = 0; i < old_cost_vals.size(); ++i) {
      double approx_improve = old_cost_vals[i] - quad_cost_vals[i];
      double exact_improve = old_cost_vals[i] - new_cost_vals[i];
      double ratio = exact_improve / approx_improve;
      if (fabs(approx_improve) > 1e-8) {
        LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10.3e", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, ratio);
      } else {
        LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10s", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, "  ------  ");
      }
    }
    LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10.3e", "TOTAL", old_merit, approx_merit_improve, exact_merit_improve, merit_improve_ratio);
  }

  OptResultPtr optimize(const VectorXd& start_x) {
    assert(start_x.size() == num_vars());
    OptResultPtr result(new OptResult);
    result->status = OPT_INCOMPLETE;
    result->x = start_x;
    result->cost_vals = VectorXd::Zero(m_costs.size());
    eval_true_costs(start_x, result->cost_vals);
    result->cost = result->cost_vals.sum();
    result->cost_over_iters.push_back(result->cost);

    VectorXd new_x;
    VectorXd delta_x;

    vector<QuadFunctionPtr> quad_costs(m_costs.size(), QuadFunctionPtr());
    VectorXd quad_cost_vals(VectorXd::Zero(m_costs.size()));
    VectorXd new_cost_vals(VectorXd::Zero(m_costs.size()));

    // Subproblem solver and data
    //Eigen::ConjugateGradient<QuadFunction::SparseMatrixT, Eigen::Lower> solver;
    // Eigen::SimplicialLDLT<QuadFunction::SparseMatrixT, Eigen::Lower> solver;
    Eigen::CholmodSupernodalLLT<QuadFunction::SparseMatrixT, Eigen::Lower> solver;
    QuadFunction::SparseMatrixT quad_A_lower(num_vars(), num_vars());
    VectorXd quad_b(VectorXd::Zero(num_vars()));
    double quad_c = 0.;

    TrustRegion trust_region(m_var_factory.vars());
    double trust_region_coeff = 1e-3;
    double trust_region_shrink_factor = 2.;

    int iter = 0;
    while (true) {
      convexify_costs(result->x, quad_costs);

      for (auto fn : m_callbacks) {
        fn(result->x);
      }

      // build the quadratic subproblem
      LOG_DEBUG("Building quadratic problem");
      quad_A_lower.setZero(); quad_b.setZero(); quad_c = 0.;
      for (int i = 0; i < quad_costs.size(); ++i) {
        quad_costs[i]->add_to(quad_A_lower, quad_b, quad_c, m_cost_coeffs[i]);
      }

      // Build trust region
      trust_region.set_center(result->x);
      trust_region.set_coeff(trust_region_coeff);
      trust_region.add_to(quad_A_lower, quad_b, quad_c);
      //quad_A_lower.makeCompressed(); // TODO: every iteration?
      LOG_DEBUG("Done building quadratic problem");

      // Solve the quadratic subproblem
      LOG_DEBUG("Solving subproblem");
      solver.compute(quad_A_lower);
      //new_x = solver.solveWithGuess(-quad_b, result->x); // TODO: warm start and early termination
      quad_b = -quad_b; new_x = solver.solve(quad_b);
      delta_x = new_x - result->x;
      ++result->n_qp_solves;
      LOG_DEBUG("Done solving subproblem");

      double delta_x_norm = delta_x.norm();
      double delta_x_norm_thresh = 1e-10*(result->x.norm() + 1e-10);
      if (delta_x_norm < delta_x_norm_thresh) {
        LOG_INFO("converged because improvement was small (%.3e < %.3e)", delta_x_norm, delta_x_norm_thresh);
        result->status = OPT_CONVERGED;
        goto out;
      }

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

      if (merit_improve_ratio > 0) {
        result->x = new_x;
        result->cost_vals = new_cost_vals;
        result->cost = new_cost_vals.sum();
        result->cost_over_iters.push_back(result->cost);
        trust_region_coeff *= fmax(1./3., 1. - pow(2.*merit_improve_ratio - 1., 3));//m_params.trust_expand_ratio;
        trust_region_shrink_factor = 2.;
        LOG_INFO("expanded trust region. new coeff: %.3e", trust_region_coeff);
      } else {
        trust_region_coeff *= trust_region_shrink_factor;
        trust_region_shrink_factor *= 2.;
        LOG_INFO("shrunk trust region. new coeff: %.3e", trust_region_coeff);
      }

      if (trust_region_coeff > 1./m_params.min_trust_region_size) {
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
#endif

struct OptimizerImpl {
  OptParams m_params;
  VarFactory m_var_factory;

  vector<CostFuncPtr> m_costs;
  vector<double> m_cost_coeffs;
  map<CostFuncPtr, int> m_cost2idx;

  vector<Optimizer::Callback> m_callbacks;

  int num_vars() const { return m_var_factory.num_vars(); }

  int num_residuals() const {
    int n = 0;
    for (CostFuncPtr c : m_costs) {
      n += c->num_residuals();
    }
    return n;
  }

  void add_vars(const StrVec& names, vector<Var>& out) {
    out.clear();
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

  void add_callback(Optimizer::Callback fn) {
    m_callbacks.push_back(fn);
  }

  void eval_true_costs(const VectorXd& x, VectorXd& out) {
    LOG_DEBUG("Evaluating true costs:");
    assert(out.size() == m_costs.size());
    for (int i = 0; i < m_costs.size(); ++i) {
      VectorXd residuals(m_costs[i]->num_residuals());
      m_costs[i]->eval(x, residuals);
      out[i] = m_cost_coeffs[i] * residuals.squaredNorm();
      LOG_DEBUG("\t%s: %f", m_costs[i]->name().c_str(), out[i]);
    }
    LOG_DEBUG("Done");
  }

  // void print_cost_info(const VectorXd& old_cost_vals, const VectorXd& quad_cost_vals, const VectorXd& new_cost_vals,
  //                      double old_merit, double approx_merit_improve, double exact_merit_improve, double merit_improve_ratio) {
  //   assert(m_costs.size() == quad_cost_vals.size() && m_costs.size() == new_cost_vals.size() && m_costs.size() == old_cost_vals.size());

  //   LOG_INFO("%15s | %10s | %10s | %10s | %10s", "", "oldexact", "dapprox", "dexact", "ratio");
  //   LOG_INFO("%15s | %10s---%10s---%10s---%10s", "COSTS", "----------", "----------", "----------", "----------");
  //   for (int i = 0; i < old_cost_vals.size(); ++i) {
  //     double approx_improve = old_cost_vals[i] - quad_cost_vals[i];
  //     double exact_improve = old_cost_vals[i] - new_cost_vals[i];
  //     double ratio = exact_improve / approx_improve;
  //     if (fabs(approx_improve) > 1e-8) {
  //       LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10.3e", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, ratio);
  //     } else {
  //       LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10s", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, "  ------  ");
  //     }
  //   }
  //   LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10.3e", "TOTAL", old_merit, approx_merit_improve, exact_merit_improve, merit_improve_ratio);
  // }

  void numdiff_cost(CostFuncPtr cost, const VectorXd& x0, vector<AffExpr>& out_exprs) {
    VectorXd x = x0;
    VectorXd y2(cost->num_residuals());
    VectorXd y1(cost->num_residuals());
    VectorXd dy(cost->num_residuals());
    const double eps = 1e-6;

    out_exprs.clear();
    out_exprs.resize(cost->num_residuals());

    VectorXd y0(cost->num_residuals());
    cost->eval(x0, y0);
    for (int i = 0; i < y0.size(); ++i) {
      exprInc(out_exprs[i], y0[i]);
    }

    for (int i = 0; i < num_vars(); ++i) {
      x(i) += eps;
      cost->eval(x, y2);
      x(i) = x0(i) - eps;
      cost->eval(x, y1);
      x(i) = x0(i);
      dy = (y2 - y1) / (2.*eps);
      for (int j = 0; j < dy.size(); ++j) {
        if (fabs(dy(j)) > 1e-10) {
          exprInc(out_exprs[j], dy(j)*(m_var_factory.vars()[i] - x0(i)));
        }
      }
    }
  }

  void eval_residuals(const VectorXd& x, VectorXd& fvec) {
    fvec.setZero();
    int pos = 0;
    for (int i = 0; i < m_costs.size(); ++i) {
      CostFuncPtr cost = m_costs[i];
      double coeff = m_cost_coeffs[i];

      auto seg = fvec.segment(pos, cost->num_residuals());
      cost->eval(x, seg);
      seg *= coeff;

      pos += cost->num_residuals();
    }
    assert(pos == num_residuals());
  }

  void eval_jacobian(const VectorXd& x, SparseMatrixT& fjac) {
    fjac.resize(num_residuals(), num_vars());
    fjac.setZero();

    vector<Eigen::Triplet<double> > triplets;
    int pos = 0;
    for (int i = 0; i < m_costs.size(); ++i) {
      CostFuncPtr cost = m_costs[i];
      double coeff = m_cost_coeffs[i];

      JacobianContainer jc(triplets, cost->num_residuals(), pos, coeff, m_params.check_linearizations);
      cost->linearize(x, jc);

      if (m_params.check_linearizations) {
        LOG_DEBUG("Evaluating numerical derivatives for cost %s", cost->name());
        vector<AffExpr> numdiff_exprs;
        numdiff_cost(cost, x, numdiff_exprs);
        FAIL_IF_FALSE(numdiff_exprs.size() == jc.exprs().size());
        for (int z = 0; z < numdiff_exprs.size(); ++z) {
          FAIL_IF_FALSE(close(numdiff_exprs[z], jc.exprs()[z]));
        }
        LOG_DEBUG("Cost %s passed derivative check", cost->name());
      }

      pos += cost->num_residuals();
    }
    assert(pos == num_residuals());

    fjac.setFromTriplets(triplets.begin(), triplets.end());
    fjac.makeCompressed();
  }

  OptResultPtr optimize(const VectorXd& start_x) {
    assert(start_x.size() == num_vars());

    boost::timer total_timer;

    bool converged = false;
    int iter = 0;

    double damping = 1e-5;
    double damping_increase_factor = 2.;

    OptResultPtr result(new OptResult);
    result->x = start_x;
    result->status = OPT_INCOMPLETE;

    // Data for the current linearization
    VectorXd fvec(num_residuals());
    SparseMatrixT fjac(num_residuals(), num_vars());
    SparseMatrixT scaling(num_vars(), num_vars());

    // Eigen::CholmodSupernodalLLT<SparseMatrixT> solver;
    Eigen::SimplicialLDLT<SparseMatrixT> solver;
    // Temporary per-iteration data
    SparseMatrixT jtj(num_vars(), num_vars());
    SparseMatrixT lin_lhs(num_vars(), num_vars());
    // SparseMatrixT eye(num_vars(), num_vars()); eye.setIdentity();
    VectorXd lin_rhs(num_vars());
    VectorXd delta_x(num_vars());
    VectorXd tmp_residuals(num_residuals());
    VectorXd new_x(num_vars());

    bool x_changed; // whether recomputing linear model is needed, after evaluation

    while (!converged && iter < m_params.max_iter) {
      ++iter;

      boost::timer iteration_timer;

      // Initialization (only on the first iteration)
      if (iter == 1) {
        eval_residuals(result->x, fvec); ++result->n_func_evals;
        eval_jacobian(result->x, fjac); ++result->n_jacobian_evals;
        result->cost = fvec.squaredNorm();
        result->cost_over_iters.push_back(result->cost);
        x_changed = true;

        for (int i = 0; i < num_vars(); ++i) {
          scaling.coeffRef(i,i) = fjac.col(i).norm();
        }
        scaling.makeCompressed();
      }

      const double starting_cost = result->cost;

      for (auto& fn : m_callbacks) {
        fn(result->x);
      }

      // Form and solve linear model
      if (x_changed) {
        jtj = fjac.transpose()*fjac;
        lin_rhs = -fjac.transpose()*fvec;

        x_changed = false;
      }

      lin_lhs = jtj + damping*scaling.transpose()*scaling;
      solver.compute(lin_lhs);
      delta_x = solver.solve(lin_rhs);

      // Check gradient convergence condition
      double grad_max = (lin_rhs.cwiseQuotient(scaling.diagonal())).lpNorm<Eigen::Infinity>();
      double delta_x_norm;
      if (grad_max <= m_params.grad_convergence_tol) {
        result->status = OPT_CONVERGED;
        LOG_INFO("converged because gradient was small (%.3e < %.3e)", grad_max, m_params.grad_convergence_tol);
        converged = true;
      } else {
        // Check delta_x (approx improvement) relative convergence condition
        delta_x_norm = (delta_x.cwiseQuotient(scaling.diagonal())).norm();
        double delta_x_norm_thresh = m_params.approx_improve_rel_tol*(result->x.norm() + m_params.approx_improve_rel_tol);
        if (delta_x_norm <= delta_x_norm_thresh) {
          result->status = OPT_CONVERGED;
          LOG_INFO("converged because improvement was small (%.3e < %.3e)", delta_x_norm, delta_x_norm_thresh);
          converged = true;
        }
      }

      if (!converged) {
        new_x = result->x + delta_x;

        // Calculate improvement ratio
        // eval_residuals(result->x, tmp_residuals); double true_old_cost = tmp_residuals.squaredNorm();
        // double true_old_cost = fvec.squaredNorm();
        double true_old_cost = result->cost;
#ifndef NDEBUG
        eval_residuals(result->x, tmp_residuals); ++result->n_func_evals;
        assert(fabs(tmp_residuals.squaredNorm() - true_old_cost) < 1e-10);
#endif
        eval_residuals(new_x, tmp_residuals); ++result->n_func_evals; double true_new_cost = tmp_residuals.squaredNorm();
        double true_improvement = true_old_cost - true_new_cost;
        double model_improvement = delta_x.dot(damping*scaling.transpose()*scaling*delta_x + lin_rhs);
        double ratio = true_improvement / model_improvement;

        // Adjust damping
        bool expanded_trust_region = false;
        if (ratio > 0) {
          result->x = new_x;
          fvec = tmp_residuals;
#ifndef NDEBUG
          LOG_DEBUG("test evaluation");
          eval_residuals(result->x, tmp_residuals); ++result->n_func_evals;
          assert((tmp_residuals - fvec).isMuchSmallerThan(1e-10));
#endif
          eval_jacobian(result->x, fjac); ++result->n_jacobian_evals;
          result->cost = fvec.squaredNorm();
          result->cost_over_iters.push_back(result->cost);
          x_changed = true;
          for (int i = 0; i < num_vars(); ++i) {
            scaling.coeffRef(i,i) = fmax(scaling.coeffRef(i,i), fjac.col(i).norm());
          }

          damping *= fmax(1/3., 1. - pow(2*ratio - 1, 3));
          damping_increase_factor = 2.;
          expanded_trust_region = true;

        } else {
          damping *= damping_increase_factor;
          damping_increase_factor *= 2.;
          expanded_trust_region = false;
        }

        LOG_INFO(
          "% 4d: f:% 8e d:% 3.2e g:% 3.2e h:% 3.2e rho:% 3.2e mu:% 3.2e (%c) li:% 3d it:% 3.2e tt:% 3.2e",
          iter,
          result->cost,
          starting_cost - result->cost,
          grad_max,
          delta_x_norm,
          ratio,
          1./(damping+1e-15),
          expanded_trust_region ? 'E' : 'S',
          0,
          iteration_timer.elapsed(),
          total_timer.elapsed()
        );
      }
    }

    if (iter == m_params.max_iter && !converged) {
      result->status = OPT_ITER_LIMIT;
    }

    result->n_iters = iter;
    return result;

  }

};


Optimizer::Optimizer() : m_impl(new OptimizerImpl()) { }
OptParams& Optimizer::params() { return m_impl->m_params; }
int Optimizer::num_vars() const { return m_impl->num_vars(); }
void Optimizer::add_vars(const vector<string>& names, vector<Var>& out) { m_impl->add_vars(names, out); }
void Optimizer::add_cost(CostFuncPtr cost, double coeff) { m_impl->add_cost(cost, coeff); }
void Optimizer::set_cost_coeff(CostFuncPtr cost, double coeff) { m_impl->set_cost_coeff(cost, coeff); }
void Optimizer::add_callback(const Callback &fn) { m_impl->add_callback(fn); }
OptResultPtr Optimizer::optimize(const VectorXd& start_x) { return m_impl->optimize(start_x); }
