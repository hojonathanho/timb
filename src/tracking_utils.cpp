#include "tracking_utils.hpp"

#include "common.hpp"

#include <boost/multi_array.hpp>
#include <boost/heap/fibonacci_heap.hpp>


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

#define IN_RANGE(I,J) (0 <= (I) && (I) < phi.rows() && 0 <= (J) && (J) < phi.cols())
  // find zero crossing of phi
  for (int i = 0; i < phi.rows(); ++i) {
    for (int j = 0; j < phi.cols(); ++j) {
      if (pignore_mask && (*pignore_mask)(i,j)) {
        continue;
      }

      if (fabs(phi(i,j)) < 1e-3) {
        out(i,j) = 0.;
        started_as_zero(i,j) = 1;
      } else {
        const auto neighbors = {
          std::make_pair(i-1, j),
          std::make_pair(i+1, j),
          std::make_pair(i, j-1),
          std::make_pair(i, j+1)
        };
        for (const auto& nbd : neighbors) {
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
    const auto neighbors = {
      std::make_pair(i-1, j),
      std::make_pair(i+1, j),
      std::make_pair(i, j-1),
      std::make_pair(i, j+1)
    };
    for (const auto& nbd : neighbors) {
      if (!IN_RANGE(nbd.first, nbd.second)) continue;
      double new_d = out(i,j) + 1.;
      if (new_d < out(nbd.first,nbd.second)) {
        out(nbd.first,nbd.second) = new_d;
        if (!started_as_zero(i,j)) sign_multiplier(nbd.first,nbd.second) = sign_multiplier(i,j);
        heap.update(handles[nbd.first][nbd.second], {nbd.first,nbd.second,new_d});
      }
    }
  }

  if (propagate_sign) {
    out = out.cwiseProduct(sign_multiplier);
  }

#undef IN_RANGE
}
