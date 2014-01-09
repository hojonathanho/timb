#include "grid_numpy_utils.hpp"

#include "gtest/gtest.h"
#include "expr.hpp" // for close()
#include <iostream>
TEST(NumpyUtilsTest, Grid) {
  Py_Initialize();
  util::PythonInit();

  const int nx = 4, ny = 3;
  GridParams gp(-1, 1, -2, 3, nx, ny);
  DoubleField f(gp);
  int k = 0;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      f(i,j) = k++;
    }
  }

  DoubleField g(gp);
  from_numpy(to_numpy(f), g);

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      EXPECT_TRUE(close(f(i,j), g(i,j)));
    }
  }
}
