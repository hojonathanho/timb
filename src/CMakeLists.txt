include_directories(
  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${CHOLMOD_LIBDIR}
)

#add_executable(test_expr
#  test_expr.cpp
#  expr.cpp
#)

add_library(timb
  expr.cpp
  optimizer.cpp
  logging.cpp
  numpy_utils.cpp
  tracking_utils.cpp
)
target_link_libraries(timb ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES})

# add_executable(test_optimize test_optimize.cpp)
# target_link_libraries(test_optimize timb)

# add_executable(test_optimize_2 test_optimize_2.cpp)
# target_link_libraries(test_optimize_2 timb)

boost_python_module(ctimbpy python_interface.cpp)
target_link_libraries(ctimbpy timb)

#add_executable(test_ceres test_ceres.cpp)
#target_link_libraries(test_ceres ceres)
