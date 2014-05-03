#pragma once

#include "common.hpp"
#include "grid.hpp"

#include <Eigen/LU> // for determinant
#include <math.h>

static inline double square(double x) { return x*x; }

template<typename ElemT, typename ExprT>
void apply_flow_to_weights(const ScalarField<ElemT, ExprT>& phi, const DoubleField& u_x, const DoubleField& u_y, ScalarField<ExprT, ExprT>& out) {
  const GridParams& gp = phi.grid_params();
  assert(u_x.grid_params() == gp && u_y.grid_params() == gp && out.grid_params() == gp);

  apply_flow(phi, u_x, u_y, out);

  typedef Eigen::Matrix<double, 2, 2> Matrix22;

  for (int i = 0; i < gp.nx; ++i) {
    for (int j = 0; j < gp.ny; ++j) {
      auto g_u_x = u_x.grad_ij(i, j);
      auto g_u_y = u_y.grad_ij(i, j);
      Matrix22 jac_u;
      jac_u <<
        g_u_x.x, g_u_x.y,
        g_u_y.x, g_u_y.y
      ;
      out(i,j) *= fabs((Matrix22::Identity() - jac_u).determinant());
    }
  }
}

inline double timb_problem_eval_objective(
  const DoubleField& phi,
  const DoubleField& u,
  const DoubleField& v,

  // observation values and weights
  const DoubleField& z,
  const DoubleField& w_z,

  // prev phi
  const DoubleField& prev_phi,
  // flowed weights
  const DoubleField& wtilde,

  double strain_cost_coeff,
  double norm_cost_coeff
) {

  // Evaluate displacement derivatives for the strain cost
  // TODO: cache this somewhere?
  DoubleField du_dx(u.grid_params()), du_dy(u.grid_params());
  // deriv_x_central(u, du_dx);
  // deriv_y_central(u, du_dy);
  deriv_x(u, du_dx);
  deriv_y(u, du_dy);
  DoubleField dv_dx(v.grid_params()), dv_dy(v.grid_params());
  // deriv_x_central(v, dv_dx);
  // deriv_y_central(v, dv_dy);
  deriv_x(v, dv_dx);
  deriv_y(v, dv_dy);
  // Evaluate flowed phi for agreement cost (TODO: cache?)
  DoubleField flowed_prev_phi(phi.grid_params());
  apply_flow(prev_phi, u, v, flowed_prev_phi);

  double obs_cost = 0.;
  double agreement_cost = 0.;
  double strain_cost = 0.;
  double norm_cost = 0.;

  for (int i = 0; i < phi.grid_params().nx; ++i) {
    for (int j = 0; j < phi.grid_params().ny; ++j) {

      // observation cost
      obs_cost += w_z(i,j) * square(z(i,j) - phi(i,j));

      // flow agreement cost
      agreement_cost += wtilde(i,j) * square(phi(i,j) - flowed_prev_phi(i,j));

      // displacement strain cost
      strain_cost += strain_cost_coeff * (4.*square(du_dx(i,j)) + 2.*square(dv_dx(i,j) + du_dy(i,j)) + 4.*square(dv_dy(i,j)));

      // displacement norm cost
      norm_cost += norm_cost_coeff * (square(u(i,j)) + square(v(i,j)));
    }
  }

  std::cout << "obs_cost: " << obs_cost
    << "\nagreement_cost: " << agreement_cost
    << "\nstrain_cost: " << strain_cost
    << "\nnorm_cost: " << norm_cost << std::endl;

  return obs_cost + agreement_cost + strain_cost + norm_cost;
}


inline double timb_problem_eval_model_objective(
  const DoubleField& phi,
  const DoubleField& u,
  const DoubleField& v,

  // observation values and weights
  const DoubleField& z,
  const DoubleField& w_z,

  // linearized flow
  const DoubleField& mu_0,
  const DoubleField& mu_u,
  const DoubleField& mu_v,
  // flowed weights
  const DoubleField& wtilde,

  double strain_cost_coeff,
  double norm_cost_coeff
) {
  // Evaluate displacement derivatives for the strain cost
  // TODO: cache this somewhere?
  DoubleField du_dx(u.grid_params()), du_dy(u.grid_params());
  // deriv_x_central(u, du_dx);
  // deriv_y_central(u, du_dy);
  deriv_x(u, du_dx);
  deriv_y(u, du_dy);
  DoubleField dv_dx(v.grid_params()), dv_dy(v.grid_params());
  // deriv_x_central(v, dv_dx);
  // deriv_y_central(v, dv_dy);
  deriv_x(v, dv_dx);
  deriv_y(v, dv_dy);

  double obs_cost = 0.;
  double agreement_cost = 0.;
  double strain_cost = 0.;
  double norm_cost = 0.;

  for (int i = 0; i < phi.grid_params().nx; ++i) {
    for (int j = 0; j < phi.grid_params().ny; ++j) {

      // observation cost
      obs_cost += w_z(i,j) * square(z(i,j) - phi(i,j));

      // linearized flow agreement cost
      agreement_cost += wtilde(i,j) * square(phi(i,j) - mu_0(i,j) - mu_u(i,j)*u(i,j) - mu_v(i,j)*v(i,j));

      // displacement strain cost
      strain_cost += strain_cost_coeff * (4.*square(du_dx(i,j)) + 2.*square(dv_dx(i,j) + du_dy(i,j)) + 4.*square(dv_dy(i,j)));

      // displacement norm cost
      norm_cost += norm_cost_coeff * (square(u(i,j)) + square(v(i,j)));
    }
  }

  std::cout << "obs_cost: " << obs_cost
    << "\nagreement_cost: " << agreement_cost
    << "\nstrain_cost: " << strain_cost
    << "\nnorm_cost: " << norm_cost << std::endl;

  return obs_cost + agreement_cost + strain_cost + norm_cost;
}


inline void timb_linearize_flowed_prev_phi(
  const DoubleField& prev_phi,
  const DoubleField& u, const DoubleField& v,
  DoubleField& out_0, DoubleField& out_u, DoubleField& out_v
) {

  const GridParams& gp = prev_phi.grid_params();

  for (int i = 0; i < gp.nx; ++i) {
    for (int j = 0; j < gp.ny; ++j) {
      auto xy = gp.to_xy(i, j);
      double flowed_x = xy.first - u(i,j), flowed_y = xy.second - v(i,j);
      double flowed_prev_phi_val = prev_phi.eval_xy(flowed_x, flowed_y);
      auto prev_phi_grad = prev_phi.grad_xy(flowed_x, flowed_y);

      out_0(i,j) = flowed_prev_phi_val + prev_phi_grad.x*u(i,j) + prev_phi_grad.y*v(i,j);
      out_u(i,j) = -prev_phi_grad.x;
      out_v(i,j) = -prev_phi_grad.y;
    }
  }
}


inline void timb_solve_model_problem_gauss_seidel(
  DoubleField& phi,
  DoubleField& u,
  DoubleField& v,

  // observation values and weights
  const DoubleField& z,
  const DoubleField& w_z,

  // linearized flow
  const DoubleField& mu_0,
  const DoubleField& mu_u,
  const DoubleField& mu_v,
  // flowed weights
  const DoubleField& wtilde,

  double alpha, // strain cost coeff
  double beta, // norm cost coeff

  double gamma, // soft trust region cost
  // trust region centers
  const DoubleField& phi_0,
  const DoubleField& u_0,
  const DoubleField& v_0,

  double h, // grid spacing
  int grid_size,
  int num_iters
) {

  for (int iter = 0; iter < num_iters; ++iter) {

    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        if (i == 0) {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j+1) + 2*alpha*v(i+1,j) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i,j+1) - 2*alpha*u(i,j) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j+1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-2) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j-1) + 4*alpha*u(i,j+1) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i+1,j+1) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) - 2*alpha*v(i,j+1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(10*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (-2*alpha*u(i,j) + 2*alpha*u(i,j+1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j-1) + 8*alpha*v(i,j+1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(14*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i+1,j) + 4*alpha*u(i,j-1) - 2*alpha*v(i+1,j-1) - 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(8*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (-2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*v(i+1,j) + 8*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(10*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j-1) + 2*alpha*u(i,j+1) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(8*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (-2*alpha*u(i,j) + 2*alpha*u(i,j+1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j-1) + 4*alpha*v(i,j+1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(10*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else if (i == grid_size-2) {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 2*alpha*u(i,j+1) + 8*alpha*u(i+1,j) - 2*alpha*v(i,j) + 2*alpha*v(i+1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(14*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (-2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) + 2*alpha*u(i,j+1) - 2*alpha*u(i,j) + 2*alpha*u(i+1,j+1) - 2*alpha*u(i+1,j) + 2*alpha*v(i-1,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(10*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-2) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 2*alpha*u(i,j-1) + 4*alpha*u(i,j+1) + 8*alpha*u(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) - 2*alpha*v(i,j+1) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i+1,j+1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(18*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j) - 2*alpha*u(i-1,j+1) - 2*alpha*u(i,j) + 2*alpha*u(i,j+1) - 2*alpha*u(i+1,j) + 2*alpha*u(i+1,j+1) + 2*alpha*v(i-1,j) + 4*alpha*v(i,j-1) + 8*alpha*v(i,j+1) + 4*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(18*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 4*alpha*u(i,j-1) + 8*alpha*u(i+1,j) + 2*alpha*v(i,j-1) + 2*alpha*v(i,j) - 2*alpha*v(i+1,j-1) - 2*alpha*v(i+1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(16*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 2*alpha*u(i,j-1) + 2*alpha*u(i,j) - 2*alpha*u(i+1,j-1) + 2*alpha*u(i+1,j) + 2*alpha*v(i-1,j) + 8*alpha*v(i,j-1) + 4*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(14*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 2*alpha*u(i,j-1) + 2*alpha*u(i,j+1) + 8*alpha*u(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(16*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j) - 2*alpha*u(i-1,j+1) - 2*alpha*u(i,j) + 2*alpha*u(i,j+1) - 2*alpha*u(i+1,j) + 2*alpha*u(i+1,j+1) + 2*alpha*v(i-1,j) + 4*alpha*v(i,j-1) + 4*alpha*v(i,j+1) + 4*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(14*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else if (i == grid_size-1) {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (8*alpha*u(i-1,j) + 2*alpha*u(i,j+1) - 2*alpha*v(i-1,j) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(10*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (-2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) - 2*alpha*u(i,j+1) + 2*alpha*u(i,j) + 4*alpha*v(i-1,j) + 4*alpha*v(i,j+1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(8*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-2) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (8*alpha*u(i-1,j) + 2*alpha*u(i,j-1) + 4*alpha*u(i,j+1) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) - 2*alpha*v(i-1,j+1) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + 2*alpha*v(i,j+1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(14*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i,j) - 2*alpha*u(i,j+1) + 4*alpha*v(i-1,j) + 4*alpha*v(i,j-1) + 8*alpha*v(i,j+1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(16*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (8*alpha*u(i-1,j) + 4*alpha*u(i,j-1) + 2*alpha*v(i-1,j-1) + 2*alpha*v(i-1,j) - 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) + 2*alpha*u(i,j-1) - 2*alpha*u(i,j) + 4*alpha*v(i-1,j) + 8*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (8*alpha*u(i-1,j) + 2*alpha*u(i,j-1) + 2*alpha*u(i,j+1) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i,j) - 2*alpha*u(i,j+1) + 4*alpha*v(i-1,j) + 4*alpha*v(i,j-1) + 4*alpha*v(i,j+1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 2*alpha*u(i,j+1) + 4*alpha*u(i+1,j) - 2*alpha*v(i,j) + 2*alpha*v(i+1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(10*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (-2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) + 2*alpha*u(i,j+1) - 2*alpha*u(i,j) + 2*alpha*v(i-1,j) + 4*alpha*v(i,j+1) + 2*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(8*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-2) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 2*alpha*u(i,j-1) + 4*alpha*u(i,j+1) + 4*alpha*u(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) - 2*alpha*v(i,j+1) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i+1,j+1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(14*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j) - 2*alpha*u(i-1,j+1) - 2*alpha*u(i,j) + 2*alpha*u(i,j+1) + 2*alpha*v(i-1,j) + 4*alpha*v(i,j-1) + 8*alpha*v(i,j+1) + 2*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(16*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 4*alpha*u(i,j-1) + 4*alpha*u(i+1,j) + 2*alpha*v(i,j-1) + 2*alpha*v(i,j) - 2*alpha*v(i+1,j-1) - 2*alpha*v(i+1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*v(i-1,j) + 8*alpha*v(i,j-1) + 2*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            u(i,j) = (4*alpha*u(i-1,j) + 2*alpha*u(i,j-1) + 2*alpha*u(i,j+1) + 4*alpha*u(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            v(i,j) = (2*alpha*u(i-1,j) - 2*alpha*u(i-1,j+1) - 2*alpha*u(i,j) + 2*alpha*u(i,j+1) + 2*alpha*v(i-1,j) + 4*alpha*v(i,j-1) + 4*alpha*v(i,j+1) + 2*alpha*v(i+1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        }
      }
    } // end iteration over grid
  } // end gauss-seidel iterations

}


#if 0
inline void timb_solve_model_problem_jacobi(
  DoubleField& phi_input,
  DoubleField& u_input,
  DoubleField& v_input,
  DoubleField& mirror_phi_input,
  DoubleField& mirror_u_input,
  DoubleField& mirror_v_input,

  // observation values and weights
  const DoubleField& z,
  const DoubleField& w_z,

  // linearized flow
  const DoubleField& mu_0,
  const DoubleField& mu_u,
  const DoubleField& mu_v,
  // flowed weights
  const DoubleField& wtilde,

  double alpha, // strain cost coeff
  double beta, // norm cost coeff

  double gamma, // soft trust region cost
  // trust region centers
  const DoubleField& phi_0,
  const DoubleField& u_0,
  const DoubleField& v_0,

  double h, // grid spacing
  int grid_size,
  int num_iters
) {

  for (int iter = 0; iter < num_iters; ++iter) {
    DoubleField& phi = (iter % 2 == 0) ? phi_input : mirror_phi_input;
    DoubleField& u = (iter % 2 == 0) ? u_input : mirror_u_input;
    DoubleField& v = (iter % 2 == 0) ? v_input : mirror_v_input;

    DoubleField& mirror_phi = (iter % 2 == 1) ? phi_input : mirror_phi_input;
    DoubleField& mirror_u = (iter % 2 == 1) ? u_input : mirror_u_input;
    DoubleField& mirror_v = (iter % 2 == 1) ? v_input : mirror_v_input;

    static const double eps = 1e-10;

    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        if (i == 0) {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 2*alpha*v(i+1,j+1) - 2*alpha*v(i+1,j) - 2*alpha*v(i,j+1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = (2*alpha*u(i+1,j+1) - 2*alpha*u(i+1,j) - 2*alpha*u(i,j+1) + 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = (-2*alpha*u(i+1,j-1) + 2*alpha*u(i+1,j) + 2*alpha*u(i,j-1) - 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + alpha*v(i+1,j+1) - alpha*v(i+1,j-1) - alpha*v(i,j+1) + alpha*v(i,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)) + eps);

            mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j-1) - alpha*u(i,j+1) + alpha*u(i,j-1) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          }

        } else if (i == grid_size-1) {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (-4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + 2*alpha*v(i,j+1) - 2*alpha*v(i,j) - 2*alpha*v(i-1,j+1) + 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = (2*alpha*u(i,j+1) - 2*alpha*u(i,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (-4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = (-2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + alpha*v(i,j+1) - alpha*v(i,j-1) - alpha*v(i-1,j+1) + alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)) + eps);

            mirror_v(i,j) = (alpha*u(i,j+1) - alpha*u(i,j-1) - alpha*u(i-1,j+1) + alpha*u(i-1,j-1) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          }

        } else {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 4*alpha*u(i-1,j) + alpha*v(i+1,j+1) - alpha*v(i+1,j) - alpha*v(i-1,j+1) + alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j) - alpha*u(i-1,j+1) + alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)) + eps);

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) + 4*alpha*u(i-1,j) - alpha*v(i+1,j-1) + alpha*v(i+1,j) + alpha*v(i-1,j-1) - alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = (-alpha*u(i+1,j-1) + alpha*u(i+1,j) + alpha*u(i-1,j-1) - alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)) + eps);

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            mirror_u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + 4*alpha*u(i-1,j) + (1.0L/2.0L)*alpha*v(i+1,j+1) - 1.0L/2.0L*alpha*v(i+1,j-1) - 1.0L/2.0L*alpha*v(i-1,j+1) + (1.0L/2.0L)*alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            mirror_v(i,j) = ((1.0L/2.0L)*alpha*u(i+1,j+1) - 1.0L/2.0L*alpha*u(i+1,j-1) - 1.0L/2.0L*alpha*u(i-1,j+1) + (1.0L/2.0L)*alpha*u(i-1,j-1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          }

        }
      }
    } // end iteration over grid


    double val = timb_problem_eval_model_objective(mirror_phi, mirror_u, mirror_v, z, w_z, mu_0, mu_u, mu_v, wtilde, alpha, beta);
    // int nz_phi = 0, nz_mirror_phi = 0;
    // for (int i = 0; i < grid_size; ++i) {
    //   for (int j = 0; j < grid_size; ++j) {
    //     if (phi_input(i,j) == 0 || isnan(phi_input(i,j))) ++nz_phi;
    //     if (mirror_phi_input(i,j) == 0 || isnan(mirror_phi_input(i,j))) ++nz_mirror_phi;
    //   }
    // }
    printf("=== jacobi iter %d done, val: %f ===\n", iter, val);
  }
}


inline void timb_solve_model_problem_gauss_seidel(
  DoubleField& phi,
  DoubleField& u,
  DoubleField& v,

  // observation values and weights
  const DoubleField& z,
  const DoubleField& w_z,

  // linearized flow
  const DoubleField& mu_0,
  const DoubleField& mu_u,
  const DoubleField& mu_v,
  // flowed weights
  const DoubleField& wtilde,

  double alpha, // strain cost coeff
  double beta, // norm cost coeff

  double gamma, // soft trust region cost
  // trust region centers
  const DoubleField& phi_0,
  const DoubleField& u_0,
  const DoubleField& v_0,

  double h, // grid spacing
  int grid_size,
  int num_iters
) {

  for (int iter = 0; iter < num_iters; ++iter) {
    static const double eps = 1e-10;

    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        if (i == 0) {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 2*alpha*v(i+1,j+1) - 2*alpha*v(i+1,j) - 2*alpha*v(i,j+1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = (2*alpha*u(i+1,j+1) - 2*alpha*u(i+1,j) - 2*alpha*u(i,j+1) + 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = (-2*alpha*u(i+1,j-1) + 2*alpha*u(i+1,j) + 2*alpha*u(i,j-1) - 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + alpha*v(i+1,j+1) - alpha*v(i+1,j-1) - alpha*v(i,j+1) + alpha*v(i,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)) + eps);

            v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j-1) - alpha*u(i,j+1) + alpha*u(i,j-1) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          }

        } else if (i == grid_size-1) {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (-4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + 2*alpha*v(i,j+1) - 2*alpha*v(i,j) - 2*alpha*v(i-1,j+1) + 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = (2*alpha*u(i,j+1) - 2*alpha*u(i,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (-4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = (-2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + alpha*v(i,j+1) - alpha*v(i,j-1) - alpha*v(i-1,j+1) + alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)) + eps);

            v(i,j) = (alpha*u(i,j+1) - alpha*u(i,j-1) - alpha*u(i-1,j+1) + alpha*u(i-1,j-1) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          }

        } else {

          if (j == 0) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 4*alpha*u(i-1,j) + alpha*v(i+1,j+1) - alpha*v(i+1,j) - alpha*v(i-1,j+1) + alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j) - alpha*u(i-1,j+1) + alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)) + eps);

          } else if (j == grid_size-1) {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) + 4*alpha*u(i-1,j) - alpha*v(i+1,j-1) + alpha*v(i+1,j) + alpha*v(i-1,j-1) - alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = (-alpha*u(i+1,j-1) + alpha*u(i+1,j) + alpha*u(i-1,j-1) - alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)) + eps);

          } else {

            phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j) + eps);

            u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + 4*alpha*u(i-1,j) + (1.0L/2.0L)*alpha*v(i+1,j+1) - 1.0L/2.0L*alpha*v(i+1,j-1) - 1.0L/2.0L*alpha*v(i-1,j+1) + (1.0L/2.0L)*alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j) + eps);

            v(i,j) = ((1.0L/2.0L)*alpha*u(i+1,j+1) - 1.0L/2.0L*alpha*u(i+1,j-1) - 1.0L/2.0L*alpha*u(i-1,j+1) + (1.0L/2.0L)*alpha*u(i-1,j-1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j) + eps);

          }

        }
      }
    } // end iteration over grid


    double val = timb_problem_eval_model_objective(phi, u, v, z, w_z, mu_0, mu_u, mu_v, wtilde, alpha, beta);
    printf("=== gauss-seidel iter %d done, val: %f ===\n", iter, val);
  }
}
#endif