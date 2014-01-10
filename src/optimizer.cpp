#include "optimizer.hpp"

#include <map>
#include <cstdio>
using std::printf;
using std::map;

#include <boost/timer.hpp>
#include <boost/format.hpp>

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
  check_linearizations(false),
  keep_results_over_iterations(false)
{ }

string OptParams::str() const {
  return (boost::format(
    "init_trust_region_size:% 3.2e\n"
    "min_trust_region_size:% 3.2e\n"
    "grad_convergence_tol:% 3.2e\n"
    "approx_improve_rel_tol:% 3.2e\n"
    "max_iter:% d\n"
    "check_linearizations:% d"
    )
    % init_trust_region_size
    % min_trust_region_size
    % grad_convergence_tol
    % approx_improve_rel_tol
    % max_iter
    % check_linearizations
  ).str();
}

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

struct OptimizerImpl {
  OptParams m_params;
  VarFactory m_var_factory;

  vector<CostFuncPtr> m_costs;
  vector<double> m_cost_coeffs;
  map<CostFuncPtr, int> m_cost2idx;

  // store cost linearizations for linear costs only
  // (nonlinear ones are recomputed at each step)
  map<CostFuncPtr, CostFuncLinearizationPtr> m_cost2lin;

  vector<Optimizer::Callback> m_callbacks;

  int num_vars() const { return m_var_factory.num_vars(); }

  int num_residuals() const {
    int n = 0;
    for (CostFuncPtr c : m_costs) {
      n += c->num_residuals();
    }
    return n;
  }

  int num_costs() const {
    return m_costs.size();
  }

  void add_vars(const StrVec& names, vector<Var>& out) {
    out.clear();
    for (int i = 0; i < names.size(); ++i) {
      out.push_back(m_var_factory.make_var(names[i]));
    }
  }
  void add_cost(CostFuncPtr cost, double coeff) {
    assert(m_costs.size() == m_cost_coeffs.size());
    FAIL_IF_FALSE(coeff >= 0);
    m_costs.push_back(cost);
    m_cost_coeffs.push_back(coeff);
    m_cost2idx[cost] = m_costs.size() - 1;
  }
  void set_cost_coeff(CostFuncPtr cost, double coeff) {
    assert(m_cost2idx.find(cost) != m_cost2idx.end());
    FAIL_IF_FALSE(coeff >= 0);
    m_cost_coeffs[m_cost2idx[cost]] = coeff;
  }

  void add_callback(Optimizer::Callback fn) {
    m_callbacks.push_back(fn);
  }

  void print_cost_info(const VectorXd& old_cost_vals, const VectorXd& model_cost_vals, const VectorXd& new_cost_vals) {
                       // double old_merit, double approx_merit_improve, double exact_merit_improve, double merit_improve_ratio) {
    assert(m_costs.size() == model_cost_vals.size() && m_costs.size() == new_cost_vals.size() && m_costs.size() == old_cost_vals.size());

    LOG_INFO("%15s | %10s | %10s | %10s | %10s", "", "oldexact", "dapprox", "dexact", "ratio");
    LOG_INFO("%15s | %10s---%10s---%10s---%10s", "COSTS", "----------", "----------", "----------", "----------");
    for (int i = 0; i < old_cost_vals.size(); ++i) {
      double approx_improve = old_cost_vals[i] - model_cost_vals[i];
      double exact_improve = old_cost_vals[i] - new_cost_vals[i];
      double ratio = exact_improve / approx_improve;
      if (fabs(approx_improve) > 1e-8) {
        LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10.3e", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, ratio);
      } else {
        LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10s", m_costs[i]->name().c_str(), old_cost_vals[i], approx_improve, exact_improve, "  ------  ");
      }
    }
    // LOG_INFO("%15s | %10.3e | %10.3e | %10.3e | %10.3e", "TOTAL", old_merit, approx_merit_improve, exact_merit_improve, merit_improve_ratio);
  }

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
      seg *= sqrt(coeff);

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

      // If the cost is linear and its jacobian hasn't yet been computed,
      // or if the cost is nonlinear, then compute the jacobian
      // and cache if it's linear
      CostFuncLinearizationPtr lin;
      if ((cost->is_linear() && m_cost2lin.find(cost) == m_cost2lin.end()) || !cost->is_linear()) {
        double coeff = m_cost_coeffs[i];
        lin.reset(new CostFuncLinearization(
          triplets, cost->num_residuals(), pos, sqrt(coeff),
          cost->is_linear(), m_params.check_linearizations
        ));
        cost->linearize(x, *lin);

        if (cost->is_linear()) {
          m_cost2lin[cost] = lin;
        }

      // If the cost is linear and the jacobian has been computed, then use that
      } else {
        lin = m_cost2lin[cost];
        triplets.reserve(triplets.size() + lin->stored_triplets().size());
        triplets.insert(triplets.end(), lin->stored_triplets().begin(), lin->stored_triplets().end());
      }

      if (m_params.check_linearizations) {
        LOG_DEBUG("Evaluating numerical derivatives for cost %s", cost->name().c_str());
        vector<AffExpr> numdiff_exprs;
        numdiff_cost(cost, x, numdiff_exprs);
        FAIL_IF_FALSE(numdiff_exprs.size() == lin->exprs().size());
        for (int z = 0; z < numdiff_exprs.size(); ++z) {
          if (!close(numdiff_exprs[z], lin->exprs()[z])) {
            std::stringstream s1, s2;
            s1 << numdiff_exprs[z]; s2 << lin->exprs()[z];
            PRINT_AND_THROW((boost::format("Cost %s: numdiff %s, analytical %s not close") % cost->name() % s1.str() % s2.str()).str());
          }
        }
        LOG_DEBUG("Cost %s passed derivative check", cost->name().c_str());
      }

      pos += cost->num_residuals();
    }
    assert(pos == num_residuals());

    fjac.setFromTriplets(triplets.begin(), triplets.end());
    fjac.makeCompressed();
  }

  void update_result(OptResultPtr result, const VectorXd& fvec) {
    result->cost = fvec.squaredNorm();

    result->cost_detail.resize(num_costs());
    int pos = 0;
    // TODO: if there are a lot of costs, just do this, and set result->cost to the sum
    for (int i = 0; i < num_costs(); ++i) {
      int num_resid = m_costs[i]->num_residuals();
      result->cost_detail(i) = fvec.segment(pos, num_resid).squaredNorm();
      pos += num_resid;
    }
    assert(pos == num_residuals() && close(result->cost_detail.sum(), result->cost));
  }

  OptResultPtr optimize(const VectorXd& start_x) {
    assert(start_x.size() == num_vars());

    boost::timer total_timer;
    LOG_INFO("Running optimizer with parameters:\n%s", m_params.str().c_str());
    if (m_params.check_linearizations) {
      LOG_WARN("Numerical derivative checking enabled!");
    }
    if (m_params.keep_results_over_iterations) {
      LOG_WARN("Saving values over iterations!")
    }

    m_cost2lin.clear();

    bool converged = false;
    int iter = 0;

    double min_scaling = 1e-5;

    double damping = 1e-5;
    double damping_increase_factor = 2.;

    OptResultPtr result(new OptResult);
    result->x = start_x;
    result->status = OPT_INCOMPLETE;

    // Data for the current linearization
    VectorXd fvec(num_residuals());
    SparseMatrixT fjac(num_residuals(), num_vars());
    SparseMatrixT scaling(num_vars(), num_vars());

    Eigen::CholmodDecomposition<SparseMatrixT> solver;
    // Eigen::SimplicialLDLT<SparseMatrixT> solver;
    // Temporary per-iteration data
    SparseMatrixT jtj(num_vars(), num_vars());
    SparseMatrixT lin_lhs(num_vars(), num_vars());
    VectorXd lin_rhs(num_vars());
    VectorXd delta_x(num_vars());
    VectorXd tmp_residuals(num_residuals());
    VectorXd new_x(num_vars());

    bool x_changed = true; // whether recomputing linear model is needed (true if a step is made)

    while (!converged && iter < m_params.max_iter) {
      if (x_changed) {
        ++iter;
      }

      boost::timer iteration_timer;

      // Initialization (only on the first iteration)
      if (iter == 1) {
        eval_residuals(result->x, fvec); ++result->n_func_evals;
        eval_jacobian(result->x, fjac); ++result->n_jacobian_evals;
        update_result(result, fvec);
        x_changed = true;

        for (int i = 0; i < num_vars(); ++i) {
          scaling.coeffRef(i,i) = fmax(min_scaling, fjac.col(i).norm());
        }
        scaling.makeCompressed();
      }

      const double starting_cost = result->cost;
      const VectorXd starting_cost_detail = result->cost_detail;

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
        double delta_x_norm_thresh = m_params.approx_improve_rel_tol*((result->x.cwiseQuotient(scaling.diagonal())).norm() + m_params.approx_improve_rel_tol);
        LOG_INFO("dx norm thresh %3.10e", delta_x_norm_thresh);
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
          update_result(result, fvec);
          x_changed = true;
          for (int i = 0; i < num_vars(); ++i) {
            scaling.coeffRef(i,i) = fmax(min_scaling, fmax(scaling.coeffRef(i,i), fjac.col(i).norm()));
          }

          damping *= fmax(1/3., 1. - pow(2*ratio - 1, 3));
          damping_increase_factor = 2.;
          expanded_trust_region = true;

          result->cost_over_iters.push_back(result->cost);
          if (m_params.keep_results_over_iterations) {
            result->x_over_iters.push_back(result->x);
          }
          print_cost_info(starting_cost_detail, result->cost_detail, result->cost_detail);

        } else {
          damping *= damping_increase_factor;
          damping_increase_factor *= 2.;
          expanded_trust_region = false;
        }

        LOG_INFO(
          "% 4d: f:% 8e d:% 3.10e g:% 3.2e h:% 3.2e rho:% 3.2e mu:% 3.2e (%c) li:% 3d it:% 3.2e tt:% 3.2e",
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
