#include <iostream>
#include <ceres/ceres.h>
#include <glog/logging.h>
using namespace std;

#define XDIM 1000000

//struct Cost {
//  template<typename T> bool operator()(const T* const x, T* residual) const {
//    cout << "enter" << endl;
//    for (int i = 0; i < XDIM; ++i) {
//      residual[i] = x[i] - T(1.0);
//    }
//    cout << "called" << endl;
//    return true;
//  }
//};

struct SingleCost {
  template<typename T> bool operator()(const T* const x, T* residual) const {
    residual[0] = x[0] - T(1.0);
    return true;
  }
};

// struct DynamicSingleCost {
//   template<typename T> bool operator(T const* const* parameters, T* residuals) const {
    
//   }
// };

string to_str(const double* x, int n) {
  stringstream ss;
  for (int i = 0; i < n; ++i) {
    ss << x[i] << ' ';
  }
  return ss.str();
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  double *x = new double[XDIM];
  for (int i = 0; i < XDIM; ++i) { x[i] = .5; }
  double *init_x = new double[XDIM];
  memcpy(init_x, x, sizeof(double)*XDIM);

  ceres::Problem prob;
//  ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<Cost, XDIM, XDIM>(new Cost);
//  prob.AddResidualBlock(cost, NULL, x);
  vector<ceres::CostFunction*> costs;
  for (int i = 0; i < XDIM; ++i) {
    ceres::CostFunction* c = new ceres::AutoDiffCostFunction<SingleCost, 1, 1>(new SingleCost);
    prob.AddResidualBlock(c, NULL, x+i);
    costs.push_back(c);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
//  options.linear_solver_type = ceres::CGNR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &prob, &summary);

  std::cout << summary.FullReport() << std::endl;
//  std::cout << "x : " << to_str(init_x, XDIM)
//            << " -> " << to_str(x, XDIM) << "\n";
  delete[] x;
  delete[] init_x;
  return 0;
}
