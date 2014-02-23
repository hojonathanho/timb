#include "tracking_utils.hpp"

#include "expr.hpp"
#include <vector>
#include <boost/multi_array.hpp>
//#include <boost/heap/fibonacci_heap.hpp>

static inline double square(double x) { return x*x; }
#if 0
void march_from_zero_crossing(const MatrixXd& phi, bool propagate_sign, const MatrixXi* pignore_mask, MatrixXd& out) {
  Eigen::MatrixXi started_as_zero(Eigen::MatrixXi::Zero(phi.rows(), phi.cols()));
  Eigen::MatrixXd sign_multiplier(Eigen::MatrixXd::Ones(phi.rows(), phi.cols()));

  out.resize(phi.rows(), phi.cols());
  out.fill(std::numeric_limits<double>::max());

  struct PointAndDist {
    int i, j;
    double d;
  };

  struct PointAndDistCmp {
    bool operator()(const PointAndDist& a, const PointAndDist& b) const {
      return a.d > b.d;
    }
  };
  typedef boost::heap::fibonacci_heap<PointAndDist, boost::heap::compare<PointAndDistCmp> > Heap;
  Heap heap;

  boost::multi_array<Heap::handle_type, 2> handles(boost::extents[phi.rows()][phi.cols()]);

  const auto make_neighbors = [](int i, int j) {
    return std::vector<std::pair<int, int> >({
      std::make_pair(i-1, j),
      std::make_pair(i+1, j),
      std::make_pair(i, j-1),
      std::make_pair(i, j+1)
    });
  };

#define IN_RANGE(I,J) (0 <= (I) && (I) < phi.rows() && 0 <= (J) && (J) < phi.cols())
  // find zero crossing of phi
  for (int i = 0; i < phi.rows(); ++i) {
    for (int j = 0; j < phi.cols(); ++j) {
      if (pignore_mask && (*pignore_mask)(i,j)) {
        continue;
      }

      if (phi(i,j) == 0.0) {
      // if (fabs(phi(i,j)) < 1e-3) {
        out(i,j) = 0.;
        started_as_zero(i,j) = 1;
      } else {
        for (const auto& nbd : make_neighbors(i,j)) {
          if (!IN_RANGE(nbd.first, nbd.second)) continue;
          if (pignore_mask && (*pignore_mask)(nbd.first, nbd.second)) continue;
          if (phi(nbd.first,nbd.second)*phi(i,j) >= 0) continue;
          double dist_to_zero = phi(i,j) / (phi(i,j) - phi(nbd.first,nbd.second));
          out(i,j) = std::min(out(i,j), dist_to_zero);
          if (phi(i,j) < 0) sign_multiplier(i,j) = -1.;
        }
      }
    }
  }

  for (int i = 0; i < phi.rows(); ++i) {
    for (int j = 0; j < phi.cols(); ++j) {
      handles[i][j] = heap.push({i, j, out(i,j)});
    }
  }

  while (!heap.empty()) {
    PointAndDist top = heap.top(); heap.pop();
    const int i = top.i, j = top.j;
    for (const auto& nbd : make_neighbors(i,j)) {
      const int i2 = nbd.first, j2 = nbd.second;
      if (!IN_RANGE(i2, j2)) continue;

      // eikonal update
      if (true) {
        double new_d;
        if (IN_RANGE(i2-1,j2) && IN_RANGE(i2+1,j2) && IN_RANGE(i2,j2-1) && IN_RANGE(i2,j2+1)) {
          double dx = fmin(out(i2-1,j2), out(i2+1,j2));
          double dy = fmin(out(i2,j2-1), out(i2,j2+1));
          double delta = 2. - square(dx - dy);
          if (delta >= 0) {
            new_d = (dx + dy + sqrt(delta)) / 2.;
          } else {
            new_d = fmin(dx, dy) + 1.;
          }
        } else {
          new_d = fmin(out(i2,j2), out(i,j) + 1.);
        }

        // std::cout << "new_d " << new_d << std::endl;
        if (new_d < out(i2,j2)) {
          out(i2,j2) = new_d;
          if (!started_as_zero(i,j)) sign_multiplier(i2,j2) = sign_multiplier(i,j);
          heap.update(handles[i2][j2], {i2,j2,new_d});
        }

      } else {
        double new_d = out(i,j) + 1.;
        if (new_d < out(i2,j2)) {
          out(i2,j2) = new_d;
          if (!started_as_zero(i,j)) sign_multiplier(i2,j2) = sign_multiplier(i,j);
          heap.update(handles[i2][j2], {i2,j2,new_d});
        }
      }

    }
  }

  if (propagate_sign) {
    out = out.cwiseProduct(sign_multiplier);
  }

#undef IN_RANGE
}


void make_flow_operator(const DoubleField& u_x, const DoubleField& u_y, SparseMatrixT& out) {
  const GridParams& gp = u_x.grid_params();
  assert(u_y.grid_params() == gp);

  // Make field dummy variables
  AffExprField phi(gp);
  VarFactory factory;
  for (int i = 0; i < gp.nx; ++i) {
    for (int j = 0; j < gp.ny; ++j) {
      phi(i,j) = AffExpr(factory.make_var(""));
    }
  }

  // Flow the variables
  AffExprField flowed_phi(gp);
  apply_flow(phi, u_x, u_y, flowed_phi);

  // Build flow operator by looking at the resulting expressions after flowing
  typedef Eigen::Triplet<double> Triplet;
  std::vector<Triplet> triplets;
  triplets.reserve(4 * gp.nx * gp.ny);

  int k = 0;
  for (int i = 0; i < gp.nx; ++i) {
    for (int j = 0; j < gp.ny; ++j) {
      const AffExpr &e = flowed_phi(i,j);
      std::cout << i << ' ' << j << ": " << e << std::endl;
      assert(close(e.constant, 0)); // better be linear

      for (int z = 0; z < e.size(); ++z) {
        int row = k;
        int col = e.vars[z].rep->index;
        double val = e.coeffs[z];
        triplets.push_back(Triplet(row, col, val));
      }

      ++k;
    }
  }
  out.setZero();
  out.resize(gp.nx*gp.ny, gp.nx*gp.ny);
  out.setFromTriplets(triplets.begin(), triplets.end());
}

void compute_flowed_precision(const VectorXd& precision_diag, const DoubleField& u_x, const DoubleField& u_y, VectorXd& out_diag) {
  const GridParams& gp = u_x.grid_params();
  assert(u_y.grid_params() == gp);

  // negate flow
  DoubleField uinv_x(gp), uinv_y(gp);
  for (int i = 0; i < gp.nx; ++i) {
    for (int j = 0; j < gp.ny; ++j) {
      uinv_x(i,j) = -u_x(i,j);
      uinv_y(i,j) = -u_y(i,j);
    }
  }

  SparseMatrixT F;
  make_flow_operator(uinv_x, uinv_y, F);

  // out_diag.resize(precision_diag.size());
  // for (int i = 0; i < precision_diag.size(); ++i) {
  //   out_diag(i) = F.col(i).dot(F.col(i).cwiseProduct(precision_diag));
  // }

  out_diag = (F.transpose() * precision_diag.asDiagonal() * F).eval().diagonal();
}

void compute_flowed_precision_direct(const VectorXd& precision_diag, const DoubleField& u_x, const DoubleField& u_y, VectorXd& out_diag) {
  const GridParams& gp = u_x.grid_params();
  assert(u_y.grid_params() == gp);

  SparseMatrixT F;
  make_flow_operator(u_x, u_y, F);

  out_diag = (F * precision_diag.asDiagonal() * F.transpose()).eval().diagonal();
}
#endif
