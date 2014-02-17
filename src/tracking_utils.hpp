#pragma once

#include "common.hpp"
#include "grid.hpp"

void march_from_zero_crossing(const MatrixXd& phi, bool propagate_sign, const MatrixXi* pignore_mask, MatrixXd& out);

void make_flow_operator(const DoubleField& u_x, const DoubleField& u_y, SparseMatrixT& out);

void compute_flowed_precision(const VectorXd& precision_diag, const DoubleField& u_x, const DoubleField& u_y, VectorXd& out_diag);

void compute_flowed_precision_direct(const VectorXd& precision_diag, const DoubleField& u_x, const DoubleField& u_y, VectorXd& out_diag);
