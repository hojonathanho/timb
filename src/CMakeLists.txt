include_directories(
  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(test_expr
  test_expr.cpp
  expr.cpp
)

add_library(timb
  expr.cpp
  problem.cpp
  logging.cpp
)

add_executable(test_optimize test_optimize.cpp)
target_link_libraries(test_optimize timb)

boost_python_module(ctimbpy python_interface.cpp numpy_utils.cpp)
target_link_libraries(ctimbpy timb)
