#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include "logging.hpp"

using std::vector;
using std::string;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixX2d;

typedef vector<string> StrVec;


#define PRINT_AND_THROW(s) do {\
  std::cerr << "\033[1;31mERROR " << s << "\033[0m\n";\
  std::cerr << "at " << __FILE__ << ":" << __LINE__ << std::endl;\
  std::stringstream ss;\
  ss << s;\
  throw std::runtime_error(ss.str());\
} while (0)
#define FAIL_IF_FALSE(expr) if (!(expr)) {\
  PRINT_AND_THROW( "expected true: " #expr);\
}


template<typename T>
inline T clip(T x, T lo, T hi) {
  return std::max(std::min(x, hi), lo);
}

template<typename T>
inline T square(const T& x) { return x*x; }
