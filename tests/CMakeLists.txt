enable_testing()

find_package(PythonLibs REQUIRED)

include_directories(
  ${gtest_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR}

  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src

  ${PYTHON_INCLUDE_PATH}
)

set(GTEST_LIBRARIES gtest gtest_main)

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test timb ${GTEST_LIBRARIES})

add_executable(powell_test powell_test.cpp)
target_link_libraries(powell_test timb ${GTEST_LIBRARIES})

add_executable(numpy_utils_test numpy_utils_test.cpp)
target_link_libraries(numpy_utils_test timb ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARY} ${GTEST_LIBRARIES})

add_executable(test_optimize_2 test_optimize_2.cpp)
target_link_libraries(test_optimize_2 timb ${GTEST_LIBRARIES})

#add_executable(test_ceres test_ceres.cpp)
#target_link_libraries(test_ceres ceres)
