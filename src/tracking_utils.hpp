#pragma once

#include "common.hpp"
#include "grid.hpp"

#if 0
void march_from_zero_crossing(const MatrixXd& phi, bool propagate_sign, const MatrixXi* pignore_mask, MatrixXd& out);

void make_flow_operator(const DoubleField& u_x, const DoubleField& u_y, SparseMatrixT& out);

void compute_flowed_precision(const VectorXd& precision_diag, const DoubleField& u_x, const DoubleField& u_y, VectorXd& out_diag);

void compute_flowed_precision_direct(const VectorXd& precision_diag, const DoubleField& u_x, const DoubleField& u_y, VectorXd& out_diag);
#endif

#include <Eigen/LU>
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
