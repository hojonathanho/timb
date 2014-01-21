include_directories(
  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CHOLMOD_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

link_directories(
  ${CHOLMOD_LIBDIR}
)

add_library(timb
  expr.cpp
  optimizer.cpp
  logging.cpp
  numpy_utils.cpp
  tracking_utils.cpp
)
target_link_libraries(timb ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES})

boost_python_module(ctimbpy python_interface.cpp)
target_link_libraries(ctimbpy timb)

boost_python_module(ctimb_ceres python_interface_ceres.cpp)
target_link_libraries(ctimb_ceres timb ceres)
