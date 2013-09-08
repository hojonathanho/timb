# add_library(sqp
#   expr.cpp
#   lm.cpp
# )
# target_link_libraries()

set(EIGEN_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/ext/eigen-3.2.0)

include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(test_expr
  test_expr.cpp
  expr.cpp
)

add_executable(test_optimize
  test_optimize.cpp
  expr.cpp
  problem.cpp
  logging.cpp
  )
