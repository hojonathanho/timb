#pragma once

#include "common.hpp"
#include "ceres/ceres.h"
#include <iostream>
#include <vector>
#include "grid.hpp"
#include "grid_numpy_utils.hpp"
#include <boost/format.hpp>
#include <boost/multi_array.hpp>
#include <boost/python.hpp>

namespace ceres {
class ScalableLoss : public LossFunction {
 public:
  // Constructs a ScaledLoss wrapping another loss function. Takes
  // ownership of the wrapped loss function or not depending on the
  // ownership parameter.
  ScalableLoss(const LossFunction* rho, double* aptr, Ownership ownership=TAKE_OWNERSHIP) :
      rho_(rho), aptr_(aptr), ownership_(ownership) { }

  virtual ~ScalableLoss() {
    if (ownership_ == DO_NOT_TAKE_OWNERSHIP) {
      rho_.release();
    }
  }

  void Evaluate(double s, double rho[3]) const {
    double a = *aptr_;
    if (rho_.get() == NULL) {
      rho[0] = a * s;
      rho[1] = a;
      rho[2] = 0.0;
    } else {
      rho_->Evaluate(s, rho);
      rho[0] *= a;
      rho[1] *= a;
      rho[2] *= a;
    }
  }

 private:
  internal::scoped_ptr<const LossFunction> rho_;
  double* aptr_;
  const Ownership ownership_;
  CERES_DISALLOW_COPY_AND_ASSIGN(ScalableLoss);
};
} // namespace ceres


namespace timb_ceres {

// Flow norm
// Flow rigidity
// Observation
// Agreement

using ceres::ScalableLoss;

struct CeresCost {
  CeresCost() : m_coeff(1.) { }
  virtual ~CeresCost() { }
  virtual void add_to_problem(ceres::Problem& problem) = 0;

  void set_coeff(double coeff) {
    m_coeff = coeff;
  }


  struct DistanceFunctor {
    double m_c;
    DistanceFunctor(double c) : m_c(c) { }
    void set_center(double c) { m_c = c; }

    template<typename T>
    bool operator()(const T* const x, T* residual) const {
      residual[0] = x[0] - T(m_c);
      return true;
    }
  };
  DistanceFunctor* add_block(ceres::Problem& problem, ceres::LossFunction* loss_function, double* var, double c) {
    DistanceFunctor* f = new DistanceFunctor(c);
    auto cost = new ceres::AutoDiffCostFunction<DistanceFunctor, 1, 1>(f);
    problem.AddResidualBlock(cost, new ScalableLoss(loss_function, &m_coeff, ceres::TAKE_OWNERSHIP), var);
    return f;
  }


  struct ExprFunctor {
    ExprFunctor(const AffExpr& e) : m_expr(reduceAff(e)) { }

    template <typename T>
    bool operator()(const T* const * parameters, T* residual) const {
      residual[0] = T(m_expr.constant);
      for (int i = 0; i < m_expr.size(); ++i) {
        residual[0] += T(m_expr.coeffs[i]) * parameters[i][0];
      }
      return true;
    }

    void reset_expr(const AffExpr& e2) {
      AffExpr e = reduceAff(e2);
#ifndef NDEBUG
      assert(e.size() == m_expr.size());
      for (int i = 0; i < e.size(); ++i) {
        assert(e.vars[i].name() == m_expr.vars[i].name());
      }
#endif
      m_expr = e;
    }

  private:
    AffExpr m_expr;
  };

  void add_block_for_expr(ceres::Problem& problem, ceres::LossFunction* loss_function, const AffExpr& orig_expr) {
    AffExpr expr = reduceAff(orig_expr);
    auto cost = new ceres::DynamicAutoDiffCostFunction<ExprFunctor, 4>(new ExprFunctor(expr));
    std::vector<double*> param_block(expr.size());
    for (int i = 0; i < expr.size(); ++i) {
      param_block[i] = (double*) expr.vars[i].rep->data;
      assert(param_block[i] != NULL); // should've been filled in by make_field_vars
      cost->AddParameterBlock(1);
    }
    cost->SetNumResiduals(1);
    problem.AddResidualBlock(cost, new ScalableLoss(loss_function, &m_coeff, ceres::TAKE_OWNERSHIP), param_block);
  }

protected:
  double m_coeff;
};
typedef boost::shared_ptr<CeresCost> CeresCostPtr;

class FlowNormCost : public CeresCost {
public:
  FlowNormCost(DoubleField& u_x, DoubleField& u_y) : m_gp(u_x.grid_params()), m_u_x(u_x), m_u_y(u_y) { }
  void add_to_problem(ceres::Problem& problem) {
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        add_block(problem, NULL, &m_u_x(i,j), 0.);
        add_block(problem, NULL, &m_u_y(i,j), 0.);
      }
    }
  }

private:
  GridParams m_gp;
  DoubleField& m_u_x;
  DoubleField& m_u_y;
};
typedef boost::shared_ptr<FlowNormCost> FlowNormCostPtr;


void make_field_vars(VarFactory& vf, const string& prefix, DoubleField& g, VarField& f) {
  assert(g.grid_params() == f.grid_params());

  for (int i = 0; i < f.grid_params().nx; ++i) {
    for (int j = 0; j < f.grid_params().ny; ++j) {
      string name = (boost::format("%s_%d_%d") % prefix % i % j).str();
      f(i,j) = vf.make_var(name);
      f(i,j).rep->data = &g(i,j);
    }
  }
}


class FlowRigidityCost : public CeresCost {
public:
  FlowRigidityCost(DoubleField& u_x, DoubleField& u_y) : m_gp(u_x.grid_params()), m_u_x(u_x), m_u_y(u_y) { }

  void add_to_problem(ceres::Problem& problem) {
    VarFactory vf;
    VarField u_x_vars(m_gp), u_y_vars(m_gp);
    make_field_vars(vf, "u_x", m_u_x, u_x_vars);
    make_field_vars(vf, "u_y", m_u_y, u_y_vars);

    AffExprField u_x_x(m_gp), u_x_y(m_gp), u_y_x(m_gp), u_y_y(m_gp);
    gradient(u_x_vars, u_x_x, u_x_y);
    gradient(u_y_vars, u_y_x, u_y_y);

    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        add_block_for_expr(problem, NULL, 2.*u_x_x(i,j));
        add_block_for_expr(problem, NULL, SQRT2*(u_x_y(i,j) + u_y_x(i,j)));
        add_block_for_expr(problem, NULL, 2.*u_y_y(i,j));
      }
    }
  }

private:
  const GridParams m_gp;
  DoubleField& m_u_x;
  DoubleField& m_u_y;
};
typedef boost::shared_ptr<FlowRigidityCost> FlowRigidityCostPtr;


class ObservationCost : public CeresCost {
public:
  ObservationCost(DoubleField &phi)
    : m_gp(phi.grid_params()), m_phi(phi),
      m_vals(m_gp), m_weights(m_gp),
      m_functors(boost::extents[m_gp.nx][m_gp.ny])
  { }

  void set_observation(const DoubleField& vals, const DoubleField& weights) {
    m_vals = vals;
    m_weights = weights;
    update_vals();
  }

  void py_set_observation(boost::python::object py_vals, boost::python::object py_weights) {
    from_numpy(py_vals, m_vals);
    from_numpy(py_weights, m_weights);
    update_vals();
  }

  void add_to_problem(ceres::Problem& problem) {
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        m_functors[i][j] = add_block(problem, new ScalableLoss(NULL, &m_weights(i,j)), &m_phi(i,j), m_vals(i,j));
      }
    }
  }

private:
  const GridParams m_gp;
  DoubleField& m_phi;
  DoubleField m_vals, m_weights;
  boost::multi_array<CeresCost::DistanceFunctor*, 2> m_functors;

  void update_vals() {
    for (int i = 0; i < m_gp.nx; ++i) {
      for (int j = 0; j < m_gp.ny; ++j) {
        m_functors[i][j]->set_center(m_vals(i,j));
      }
    }
  }
};


class CeresOptimizer {
public:
  void add_cost(CeresCostPtr cost) { m_costs.push_back(cost); }

  void optimize() {
    ceres::Problem problem;
    for (CeresCostPtr cost : m_costs) {
      cost->add_to_problem(problem);
    }
    
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
  }

protected:
  std::vector<CeresCostPtr> m_costs;
};


class CeresTrackingProblem {
public:
  double m_flow_norm_coeff;
  double m_flow_rigidity_coeff;
  double m_observation_coeff;
  double m_agreement_coeff;

  CeresTrackingProblem(const GridParams& gp)
    : m_gp(gp),
      m_phi(gp), m_u_x(gp), m_u_y(gp)
  {
    m_flow_norm_cost.reset(new FlowNormCost(m_u_x, m_u_y));
    m_optimizer.add_cost(m_flow_norm_cost);

    m_flow_rigidity_cost.reset(new FlowRigidityCost(m_u_x, m_u_y));
    m_optimizer.add_cost(m_flow_rigidity_cost);

    // m_observation_cost.reset(new ObservationCost(m_phi));
    // m_optimizer.add_cost(m_observation_cost);

    // m_agreement_cost.reset(new AgreementCost(m_phi, m_u_x, m_u_y));
    // m_optimizer.add_cost(m_agreement_cost);
  }

  void optimize() {
    m_flow_norm_cost->set_coeff(m_flow_norm_coeff);
    m_flow_rigidity_cost->set_coeff(m_flow_rigidity_coeff);
    // m_observation_cost->set_coeff(m_observation_coeff);
    // m_agreement_cost->set_coeff(m_agreement_coeff);

    m_optimizer.optimize();
  }

protected:
  const GridParams m_gp;
  CeresOptimizer m_optimizer;

  DoubleField m_phi, m_u_x, m_u_y;

  FlowNormCostPtr m_flow_norm_cost;
  FlowRigidityCostPtr m_flow_rigidity_cost;
  // ObservationCostPtr m_observation_cost;
  // AgreementCostPtr m_agreement_cost;
};
typedef boost::shared_ptr<CeresTrackingProblem> CeresTrackingProblemPtr;


} // namespace timb_ceres
