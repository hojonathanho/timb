#pragma once

#include "common.hpp"
#include "grid.hpp"

#include <Eigen/LU> // for determinant

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

  double alpha, // strain cost coeff
  double beta // norm cost coeff
) {

  // Evaluate displacement derivatives for the strain cost
  // TODO: cache this somewhere?
  DoubleField du_dx(u.grid_params()), du_dy(u.grid_params());
  deriv_x_central(u, du_dx);
  deriv_y_central(u, du_dy);
  DoubleField dv_dx(v.grid_params()), dv_dy(v.grid_params());
  deriv_x_central(v, dv_dx);
  deriv_y_central(v, dv_dy);

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
      strain_cost += alpha * (4.*square(du_dx(i,j)) + 2.*square(dv_dx(i,j) + du_dy(i,j)) + 4.*square(dv_dy(i,j)));

      // displacement norm cost
      norm_cost += beta * (square(u(i,j)) + square(v(i,j)));
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

  double alpha, // strain cost coeff
  double beta // norm cost coeff
) {
  // Evaluate displacement derivatives for the strain cost
  // TODO: cache this somewhere?
  DoubleField du_dx(u.grid_params()), du_dy(u.grid_params());
  deriv_x_central(u, du_dx);
  deriv_y_central(u, du_dy);
  DoubleField dv_dx(v.grid_params()), dv_dy(v.grid_params());
  deriv_x_central(v, dv_dx);
  deriv_y_central(v, dv_dy);

  double val = 0.;

  for (int i = 0; i < phi.grid_params().nx; ++i) {
    for (int j = 0; j < phi.grid_params().ny; ++j) {

      // observation cost
      val += w_z(i,j) * square(z(i,j) - phi(i,j));

      // linearized flow agreement cost
      val += wtilde(i,j) * square(phi(i,j) - mu_0(i,j) - mu_u(i,j)*u(i,j) - mu_v(i,j)*v(i,j));

      // displacement strain cost
      val += alpha * (4.*square(du_dx(i,j)) + 2.*square(dv_dx(i,j) + du_dy(i,j)) + 4.*square(dv_dy(i,j)));

      // displacement norm cost
      val += beta * (square(u(i,j)) + square(v(i,j)));
    }
  }

  return val;
}


inline void jacobi(
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

    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        if (i == 0) {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 2*alpha*v(i+1,j+1) - 2*alpha*v(i+1,j) - 2*alpha*v(i,j+1) + 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (2*alpha*u(i+1,j+1) - 2*alpha*u(i+1,j) - 2*alpha*u(i,j+1) + 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 2*alpha*v(i+1,j-1) + 2*alpha*v(i+1,j) + 2*alpha*v(i,j-1) - 2*alpha*v(i,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (-2*alpha*u(i+1,j-1) + 2*alpha*u(i+1,j) + 2*alpha*u(i,j-1) - 2*alpha*u(i,j) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-8*alpha*u(i+1,j) + 4*alpha*u(i+2,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + alpha*v(i+1,j+1) - alpha*v(i+1,j-1) - alpha*v(i,j+1) + alpha*v(i,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)));

            mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j-1) - alpha*u(i,j+1) + alpha*u(i,j-1) - 4*alpha*v(i+1,j) + 2*alpha*v(i+2,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else if (i == grid_size-1) {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + 2*alpha*v(i,j+1) - 2*alpha*v(i,j) - 2*alpha*v(i-1,j+1) + 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (2*alpha*u(i,j+1) - 2*alpha*u(i,j) - 2*alpha*u(i-1,j+1) + 2*alpha*u(i-1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (-4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) - 2*alpha*v(i,j-1) + 2*alpha*v(i,j) + 2*alpha*v(i-1,j-1) - 2*alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (-2*alpha*u(i,j-1) + 2*alpha*u(i,j) + 2*alpha*u(i-1,j-1) - 2*alpha*u(i-1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(-6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) - 8*alpha*u(i-1,j) + 4*alpha*u(i-2,j) + alpha*v(i,j+1) - alpha*v(i,j-1) - alpha*v(i-1,j+1) + alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_u(i,j)*mu_u(i,j))*wtilde(i,j)));

            mirror_v(i,j) = (alpha*u(i,j+1) - alpha*u(i,j-1) - alpha*u(i-1,j+1) + alpha*u(i-1,j-1) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) - 4*alpha*v(i-1,j) + 2*alpha*v(i-2,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        } else {

          if (j == 0) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j+1) + 2*alpha*u(i,j+2) + 4*alpha*u(i-1,j) + alpha*v(i+1,j+1) - alpha*v(i+1,j) - alpha*v(i-1,j+1) + alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (alpha*u(i+1,j+1) - alpha*u(i+1,j) - alpha*u(i-1,j+1) + alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j+1) + 4*alpha*v(i,j+2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)));

          } else if (j == grid_size-1) {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (4*alpha*u(i+1,j) - 4*alpha*u(i,j-1) + 2*alpha*u(i,j-2) + 4*alpha*u(i-1,j) - alpha*v(i+1,j-1) + alpha*v(i+1,j) + alpha*v(i-1,j-1) - alpha*v(i-1,j) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(6*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = (-alpha*u(i+1,j-1) + alpha*u(i+1,j) + alpha*u(i-1,j-1) - alpha*u(i-1,j) + 2*alpha*v(i+1,j) - 8*alpha*v(i,j-1) + 4*alpha*v(i,j-2) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/((h*h)*(beta + gamma + (mu_v(i,j)*mu_v(i,j))*wtilde(i,j)));

          } else {

            mirror_phi(i,j) = (gamma*phi_0(i,j) + mu_0(i,j)*wtilde(i,j) + mu_u(i,j)*u(i,j)*wtilde(i,j) + mu_v(i,j)*v(i,j)*wtilde(i,j) + w_z(i,j)*z(i,j))/(gamma + wtilde(i,j) + w_z(i,j));

            mirror_u(i,j) = (4*alpha*u(i+1,j) + 2*alpha*u(i,j+1) + 2*alpha*u(i,j-1) + 4*alpha*u(i-1,j) + (1.0L/2.0L)*alpha*v(i+1,j+1) - 1.0L/2.0L*alpha*v(i+1,j-1) - 1.0L/2.0L*alpha*v(i-1,j+1) + (1.0L/2.0L)*alpha*v(i-1,j-1) + gamma*(h*h)*u_0(i,j) - (h*h)*mu_0(i,j)*mu_u(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*v(i,j)*wtilde(i,j) + (h*h)*mu_u(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_u(i,j)*mu_u(i,j))*wtilde(i,j));

            mirror_v(i,j) = ((1.0L/2.0L)*alpha*u(i+1,j+1) - 1.0L/2.0L*alpha*u(i+1,j-1) - 1.0L/2.0L*alpha*u(i-1,j+1) + (1.0L/2.0L)*alpha*u(i-1,j-1) + 2*alpha*v(i+1,j) + 4*alpha*v(i,j+1) + 4*alpha*v(i,j-1) + 2*alpha*v(i-1,j) + gamma*(h*h)*v_0(i,j) - (h*h)*mu_0(i,j)*mu_v(i,j)*wtilde(i,j) - (h*h)*mu_u(i,j)*mu_v(i,j)*u(i,j)*wtilde(i,j) + (h*h)*mu_v(i,j)*phi(i,j)*wtilde(i,j))/(12*alpha + beta*(h*h) + gamma*(h*h) + (h*h)*(mu_v(i,j)*mu_v(i,j))*wtilde(i,j));

          }

        }
      }
    } // end iteration over grid

    // printf("iter %d done\n", iter);
  }
}
